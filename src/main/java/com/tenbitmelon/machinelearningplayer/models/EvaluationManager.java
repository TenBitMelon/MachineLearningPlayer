package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.MachineLearningPlayer;
import com.tenbitmelon.machinelearningplayer.debugger.Debugger;
import com.tenbitmelon.machinelearningplayer.debugger.SystemStats;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.*;
import com.tenbitmelon.machinelearningplayer.environment.*;
import net.kyori.adventure.text.Component;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.phys.Vec3;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.craftbukkit.CraftWorld;
import org.bukkit.craftbukkit.entity.CraftEntity;
import org.bukkit.entity.ArmorStand;
import org.bukkit.entity.EntityType;
import org.bukkit.entity.Mob;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.cuda.DeviceStats;
import org.bytedeco.pytorch.global.torch;
import org.jetbrains.annotations.NotNull;

import java.util.Set;

import static com.tenbitmelon.machinelearningplayer.MachineLearningPlayer.CURRENT_MODE;
import static com.tenbitmelon.machinelearningplayer.MachineLearningPlayer.LOGGER;

public class EvaluationManager {

    static public boolean runEvaluation = false;
    static public boolean sprint = false;
    public static Device device;
    static ExperimentConfig args = ExperimentConfig.getInstance();
    //
    static MinecraftEnvironment environment;
    static MinecraftRL model;
    static MinecraftRL.LSTMState nextLstmState;
    static ResetResult resetResult;
    static String logText = "";
    /** Shape: [numEnvs, Observation.OBSERVATION_SPACE_SIZE] */
    private static Tensor nextObs;
    /** Shape: [numEnvs] */
    private static Tensor nextDone;
    private static boolean needsPostTickStep = false;

    // Basic evaluation metrics
    private static int currentStep = 0;
    private static double episodeReturn = 0.0;
    private static int episodeLength = 0;
    private static int totalEpisodes = 0;
    private static double averageReturn = 0.0;
    private static long evaluationStartTime = System.currentTimeMillis();

    public static void setup() {
        device = new Device("cuda:0");

        // Use single environment for evaluation
        args.numEnvs = 1;

        Entity armorStand = ((CraftEntity) Debugger.WORLD.spawnEntity(new Location(Debugger.WORLD, 16.0, 8.0, 16.0), EntityType.ARMOR_STAND)).getHandle();
        armorStand.setInvulnerable(true);
        environment = new MinecraftEnvironment(args);
        environment.setTarget(armorStand);
        model = new MinecraftRL(device);
        model.loadCheckpoint(args.startingCheckpoint);
        model.to(device, false);

        TensorOptions deviceTensorOptions = new TensorOptions(device);
        nextDone = torch.zeros(new long[]{args.numEnvs}, deviceTensorOptions);
        nextLstmState = new MinecraftRL.LSTMState(
            torch.zeros(new long[]{model.getLSTMLayers(), args.numEnvs, model.getLSTMHiddenSize()}, deviceTensorOptions),
            torch.zeros(new long[]{model.getLSTMLayers(), args.numEnvs, model.getLSTMHiddenSize()}, deviceTensorOptions));

        // Add evaluation-specific debug controls
        Debugger.mainDebugWindow.addControl(new TextControl("Evaluation Manager", "--------------------------------"));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Status"), () -> logText));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Env. Ready"), () -> environment.isReady()));
        Debugger.mainDebugWindow.addControl(new BooleanControl(Component.text("Run Evaluation"), () -> runEvaluation, (value) -> runEvaluation = value));
        Debugger.mainDebugWindow.addControl(new ButtonControl(Component.text("Run Step"), () -> {
            runEvaluation = true;
            evaluationStep();
            runEvaluation = false;
        }));
        Debugger.mainDebugWindow.addControl(new BooleanControl(Component.text("Sprint"), () -> sprint, (value) -> sprint = value));
        Debugger.mainDebugWindow.addControl(new TextControl(""));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Current Step"), () -> currentStep));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Episode Length"), () -> episodeLength));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Episode Return"), () -> String.format("%.2f", episodeReturn)));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Total Episodes"), () -> totalEpisodes));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Avg Return"), () -> String.format("%.2f", averageReturn)));

        // DeviceStats deviceStats = torch.cuda_device_count() > 0 ?
        //     new DeviceStats() : null;

        System.out.println("torch.cuda_is_available() = " + torch.cuda_is_available());
        System.out.println("torch.cuda_device_count() = " + torch.cuda_device_count());
        System.out.println("torch.hasCUDA() = " + torch.hasCUDA());
    }

    public static void shutdown() {
        // Cleanup resources if needed
        if (nextObs != null) {
            nextObs.close();
        }
        if (nextDone != null) {
            nextDone.close();
        }
        if (nextLstmState != null) {
            nextLstmState.close();
        }
    }

    public static void evaluationStep() {
        if (CURRENT_MODE != MachineLearningPlayer.Mode.EVALUATION) return;
        if (!runEvaluation) return;

        if (!environment.isReady()) {
            Bukkit.broadcast(Component.text("Environment is not ready for evaluation."));
            LOGGER.warn("Attempted to run evaluation step, but environment is not ready.");
            runEvaluation = false;
            return;
        }

        if (resetResult == null) {
            LOGGER.info("Initial environment reset for evaluation...");
            resetResult = environment.reset();
            nextObs = resetResult.observation().tensor().to(device, torch.ScalarType.Float);
            evaluationStartTime = System.currentTimeMillis();
        }

        if (sprint) {
            Bukkit.getServerTickManager().requestGameToSprint(200);
        } else {
            Bukkit.getServerTickManager().stopSprinting();
        }

        try {
            runEvaluationStep();
        } catch (Exception e) {
            LOGGER.error("Exception during evaluation step: {}", e.getMessage());
            e.printStackTrace();
            runEvaluation = false;
        }
    }

    private static void runEvaluationStep() {
        if (!needsPostTickStep) {
            // Get action from model (inference mode)
            logText = "Getting action for evaluation...";

            PointerScope scope = new PointerScope();
            try {
                AutogradState.get_tls_state().set_grad_mode(false); // with torch.no_grad():

                MinecraftRL.ActionAndValue actionResult = model.getActionAndValue(nextObs, nextLstmState, nextDone);
                nextLstmState.copy_(actionResult.lstmState());

                Tensor actionTensor = actionResult.action().cpu();

                // Step environment
                logText = "Stepping environment...";
                environment.preTickStep(actionTensor);
                actionTensor.close();

                needsPostTickStep = true;
            } finally {
                scope.close();
            }
        } else {
            // Process environment step results
            logText = "Processing step results...";

            PointerScope scope = new PointerScope();
            try {
                StepResult stepResult = environment.postTickStep();
                Observation observation = stepResult.observation();

                if (stepResult.logicalOrTerminationAndTruncation() > 0) {
                    ResetResult resetResult = environment.reset();
                    observation = resetResult.observation();

                    Vec3 location = environment.centerPosition.add(0.0, 0.1, -MinecraftEnvironment.CIRCLE_RADIUS / 2.0);

                    environment.targetEntity.teleportTo(((CraftWorld) environment.roomLocation.getWorld()).getHandle(), location.x(), location.y(), location.z(), Set.of(), 0, 0, true);
                    environment.targetEntity.snapTo(location.x(), location.y(), location.z(), 0, 0);
                }

                // Update observations
                if (nextObs != null) {
                    nextObs.close();
                }
                nextObs = observation.tensor().to(device, torch.ScalarType.Float);
                nextObs.retainReference();

                // Update done flags
                if (nextDone != null) {
                    nextDone.close();
                }
                nextDone = Tensor.create(stepResult.logicalOrTerminationAndTruncation()).to(device, torch.ScalarType.Float);
                nextDone.retainReference();

                // Update metrics
                updateEvaluationMetrics(stepResult);

                currentStep++;
            } finally {
                scope.close();
                needsPostTickStep = false;
            }
        }
    }

    private static void updateEvaluationMetrics(StepResult stepResult) {
        double reward = stepResult.reward();
        int terminationOrTruncation = stepResult.logicalOrTerminationAndTruncation();

        // Add rewards to current episode

        episodeReturn += reward;
        episodeLength++;

        // Check for episode completion
        if (terminationOrTruncation > 0) {
            totalEpisodes++;
            averageReturn = (averageReturn * (totalEpisodes - 1) + episodeReturn) / totalEpisodes;

            LOGGER.info("Episode {} completed: Return = {:.2f}, Length = {}, Avg Return = {:.2f}",
                totalEpisodes, episodeReturn, episodeLength, averageReturn);

            // Reset episode metrics
            episodeReturn = 0.0;
            episodeLength = 0;
        }
    }

    public static void resetEvaluationMetrics() {
        currentStep = 0;
        episodeReturn = 0.0;
        episodeLength = 0;
        totalEpisodes = 0;
        averageReturn = 0.0;
        evaluationStartTime = System.currentTimeMillis();
    }

    public static String getEvaluationSummary() {
        double elapsedMinutes = (System.currentTimeMillis() - evaluationStartTime) / (1000.0 * 60.0);
        double stepsPerMinute = elapsedMinutes > 0 ? currentStep / elapsedMinutes : 0;

        return String.format("Evaluation Summary: %d episodes, Avg Return: %.2f, %d steps, %.1f steps/min",
            totalEpisodes, averageReturn, currentStep, stepsPerMinute);
    }

    public static MinecraftEnvironment getEnvironment() {
        return environment;
    }
}
