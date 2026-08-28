package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.debugger.Debugger;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.BooleanControl;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.ButtonControl;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.TextControl;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.VariableControl;
import com.tenbitmelon.machinelearningplayer.environment.MinecraftEnvironment;
import com.tenbitmelon.machinelearningplayer.environment.Observation;
import com.tenbitmelon.machinelearningplayer.environment.StepResult;
import net.kyori.adventure.text.Component;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.phys.Vec3;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.craftbukkit.CraftWorld;
import org.bukkit.craftbukkit.entity.CraftEntity;
import org.bukkit.craftbukkit.inventory.CraftItemStack;
import org.bukkit.entity.EntityType;
import org.bukkit.inventory.ItemStack;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.AutogradState;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;

import java.util.Set;

import static com.tenbitmelon.machinelearningplayer.MachineLearningPlayer.*;
import static com.tenbitmelon.machinelearningplayer.environment.MinecraftEnvironment.getRandomPointInCircle;

public class EvaluationManager {

    static public boolean runEvaluation = false;
    static public boolean sprint = false;
    public static Device device;
    static ExperimentConfig args = ExperimentConfig.getInstance();
    //
    static MinecraftEnvironment environment;
    static MinecraftRL model;
    static MinecraftRL.LSTMState nextLstmState;
    static Observation resetResult;
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
    private static double averageEpisodeLength = 0.0;
    private static double bestReturn = Double.NEGATIVE_INFINITY;
    private static double worstReturn = Double.POSITIVE_INFINITY;
    private static double lastEpisodeReturn = 0.0;
    private static int lastEpisodeLength = 0;
    private static String lastEpisodeOutcome = "n/a";
    private static int wins = 0;
    private static int losses = 0;
    private static int draws = 0;
    private static int truncations = 0;
    private static float episodeDamageTaken = 0.0f;
    private static float episodeDamageDealt = 0.0f;
    private static float lastEpisodeDamageTaken = 0.0f;
    private static float lastEpisodeDamageDealt = 0.0f;
    private static double lastDistanceToTarget = 0.0;
    private static long evaluationStartTime = System.currentTimeMillis();

    public static void setup() {
        device = new Device("cuda:0");

        // Use single environment for evaluation
        args.numEnvs = 1;

        LivingEntity armorStand = (LivingEntity) ((CraftEntity) WORLD.spawnEntity(new Location(WORLD, 16.0, 8.0, 16.0), EntityType.ARMOR_STAND)).getHandle();
        armorStand.setInvulnerable(true);
        environment = new MinecraftEnvironment(args);
        environment.setTarget(armorStand);
        model = new MinecraftRL(device);
        model.loadCheckpoint(args.startingCheckpoint);
        model.to(device, false);
        resetEvaluationMetrics();

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
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Avg Length"), () -> String.format("%.2f", averageEpisodeLength)));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Best Return"), () -> totalEpisodes > 0 ? String.format("%.2f", bestReturn) : "n/a"));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Worst Return"), () -> totalEpisodes > 0 ? String.format("%.2f", worstReturn) : "n/a"));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Last Outcome"), () -> lastEpisodeOutcome));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("W/L/D/T"), () -> wins + "/" + losses + "/" + draws + "/" + truncations));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Last Damage"), () -> String.format("deal %.2f / take %.2f", lastEpisodeDamageDealt, lastEpisodeDamageTaken)));

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
        if (CURRENT_MODE != Mode.EVALUATION) return;
        if (!runEvaluation) return;

        if (!environment.isReady()) {
            Bukkit.broadcast(Component.text("Environment is not ready for evaluation."));
            LOGGER.warn("Attempted to run evaluation step, but environment is not ready.");
            runEvaluation = false;
            return;
        }

        if (resetResult == null) {
            LOGGER.info("Initial environment reset for evaluation...");
            environment.reset();
            resetResult = environment.getObservation();
            nextObs = resetResult.tensor().to(device, torch.ScalarType.Float);
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
                actionResult.close();

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

                if (stepResult.terminated()) {
                    environment.reset();
                    observation = environment.getObservation();

                    double minRadius = 1.0;
                    double maxRadius = 1.0;

                    if (CURRENT_MODE == Mode.TRAINING) {
                        minRadius += 1.0 / 3000.0 * TrainingManager.iteration;
                        maxRadius += 6.0 / 3000.0 * TrainingManager.iteration;
                    } else {
                        minRadius = 3.0;
                        maxRadius = 8.0;
                    }

                    double[] randomPointInCircle = getRandomPointInCircle(minRadius, maxRadius);
                    Vec3 agentLocation = environment.centerPosition.add(randomPointInCircle[0], 0, randomPointInCircle[1]);


                    environment.targetEntity.teleportTo(((CraftWorld) WORLD).getHandle(), agentLocation.x(), agentLocation.y(), agentLocation.z(), Set.of(), 0, 0, true);
                    environment.targetEntity.snapTo(agentLocation.x(), agentLocation.y(), agentLocation.z(), 0, 0);

                    if (environment.targetEntity instanceof Player player) {
                        player.getInventory().setSelectedSlot(0);
                        ItemStack itemStack = Material.WOODEN_SWORD.asItemType().createItemStack();
                        player.setItemInHand(InteractionHand.MAIN_HAND, ((CraftItemStack) itemStack).handle);

                        // if (player instanceof ServerPlayer serverPlayer) {
                        //     serverPlayer.connection.send(new ServerboundClientCommandPacket(ServerboundClientCommandPacket.Action.PERFORM_RESPAWN));
                        // }
                    }
                    environment.targetEntity.setHealth(environment.targetEntity.getMaxHealth());
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
        episodeDamageTaken += stepResult.damageTaken();
        episodeDamageDealt += stepResult.damageDealt();
        lastDistanceToTarget = stepResult.distanceToTarget();

        // Check for episode completion
        if (terminationOrTruncation > 0) {
            totalEpisodes++;
            averageReturn = (averageReturn * (totalEpisodes - 1) + episodeReturn) / totalEpisodes;
            averageEpisodeLength = (averageEpisodeLength * (totalEpisodes - 1) + episodeLength) / totalEpisodes;
            bestReturn = Math.max(bestReturn, episodeReturn);
            worstReturn = Math.min(worstReturn, episodeReturn);
            lastEpisodeReturn = episodeReturn;
            lastEpisodeLength = episodeLength;
            lastEpisodeDamageTaken = episodeDamageTaken;
            lastEpisodeDamageDealt = episodeDamageDealt;

            String outcome = "unknown";
            if (stepResult.truncated()) {
                truncations++;
                outcome = "truncated";
            } else if (stepResult.myHealth() > 0 && stepResult.targetHealth() <= 0) {
                wins++;
                outcome = "win";
            } else if (stepResult.myHealth() <= 0 && stepResult.targetHealth() > 0) {
                losses++;
                outcome = "loss";
            } else if (stepResult.myHealth() <= 0 && stepResult.targetHealth() <= 0) {
                draws++;
                outcome = "draw";
            }

            lastEpisodeOutcome = outcome;

            LOGGER.info("Episode {} completed: outcome={}, return={}, length={}, avgReturn={}, avgLength={}, W/L/D/T={}/{}/{}/{}",
                totalEpisodes, outcome, episodeReturn, episodeLength, averageReturn, averageEpisodeLength, wins, losses, draws, truncations);

            // Reset episode metrics
            episodeReturn = 0.0;
            episodeLength = 0;
            episodeDamageTaken = 0.0f;
            episodeDamageDealt = 0.0f;
        }
    }

    public static void resetEvaluationMetrics() {
        currentStep = 0;
        episodeReturn = 0.0;
        episodeLength = 0;
        totalEpisodes = 0;
        averageReturn = 0.0;
        averageEpisodeLength = 0.0;
        bestReturn = Double.NEGATIVE_INFINITY;
        worstReturn = Double.POSITIVE_INFINITY;
        lastEpisodeReturn = 0.0;
        lastEpisodeLength = 0;
        lastEpisodeOutcome = "n/a";
        wins = 0;
        losses = 0;
        draws = 0;
        truncations = 0;
        episodeDamageTaken = 0.0f;
        episodeDamageDealt = 0.0f;
        lastEpisodeDamageTaken = 0.0f;
        lastEpisodeDamageDealt = 0.0f;
        lastDistanceToTarget = 0.0;
        evaluationStartTime = System.currentTimeMillis();
    }


    public static String getDetailedEvaluationSummary() {
        double elapsedMinutes = (System.currentTimeMillis() - evaluationStartTime) / (1000.0 * 60.0);
        double stepsPerMinute = elapsedMinutes > 0 ? currentStep / elapsedMinutes : 0;

        return String.format(
            "episodes=%d avgReturn=%.2f avgLength=%.2f best=%.2f worst=%.2f last=%s(%.2f/%d) W/L/D/T=%d/%d/%d/%d steps=%d stepsPerMin=%.1f lastDamage=deal %.2f take %.2f lastDist=%.2f",
            totalEpisodes,
            averageReturn,
            averageEpisodeLength,
            totalEpisodes > 0 ? bestReturn : 0.0,
            totalEpisodes > 0 ? worstReturn : 0.0,
            lastEpisodeOutcome,
            lastEpisodeReturn,
            lastEpisodeLength,
            wins,
            losses,
            draws,
            truncations,
            currentStep,
            stepsPerMinute,
            lastEpisodeDamageDealt,
            lastEpisodeDamageTaken,
            lastDistanceToTarget
        );
    }

    public static MinecraftEnvironment getEnvironment() {
        return environment;
    }
}
