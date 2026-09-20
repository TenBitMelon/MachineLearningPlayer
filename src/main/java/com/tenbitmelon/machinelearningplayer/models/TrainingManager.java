package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.ExperimentConfig;
import com.tenbitmelon.machinelearningplayer.MachineLearningPlayer;
import com.tenbitmelon.machinelearningplayer.debugger.Debugger;
import com.tenbitmelon.machinelearningplayer.debugger.SystemStats;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.BooleanControl;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.ButtonControl;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.CounterControl;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.VariableControl;
import com.tenbitmelon.machinelearningplayer.environment.Action;
import com.tenbitmelon.machinelearningplayer.environment.Observation;
import net.kyori.adventure.text.Component;
import org.bukkit.Bukkit;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.javacpp.tools.NativeAllocationTracer;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.cuda.CUDAAllocator;
import org.bytedeco.pytorch.cuda.DeviceStats;
import org.bytedeco.pytorch.cuda.SnapshotInfo;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch_cuda;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;

import static com.tenbitmelon.machinelearningplayer.MachineLearningPlayer.CURRENT_MODE;
import static com.tenbitmelon.machinelearningplayer.MachineLearningPlayer.LOGGER;

public class TrainingManager {

    private static final Scalar SCALAR_ONE = new Scalar(1.0);
    private static final Scalar SCALAR_1E_8 = new Scalar(1e-8);
    private static final Scalar SCALAR_0_5 = new Scalar(0.5);
    static public boolean runTraining = false;
    static public boolean sprint = false;
    public static Device device;
    public static int iteration = 1;
    /** Shape: [numEnvs] */
    public static Tensor zerosLikeNumEnvs;
    /** Shape: [numEnvs] */
    public static Tensor onesLikeNumEnvs;
    public static ExperimentConfig args = new ExperimentConfig();
    // Pre computes:
    private static final Scalar SCALAR_GAMMA = new Scalar(args.gamma);
    private static final Scalar SCALAR_GAMMA_GAE_LAMBDA = new Scalar(args.gamma * args.gaeLambda);
    private static final Scalar SCALAR_NUM_ENVS = new Scalar(args.numEnvs);
    private static final Scalar SCALAR_BATCH_SIZE = new Scalar(args.batchSize);
    private static final Scalar SCALAR_CLIP_COEF = new Scalar(args.clipCoef);
    private static final ScalarOptional SCALAR_1_SUB_CLIP_COEF = new ScalarOptional(new Scalar(1.0 - args.clipCoef));
    private static final ScalarOptional SCALAR_1_ADD_CLIP_COEF = new ScalarOptional(new Scalar(1.0 + args.clipCoef));
    private static final ScalarOptional SCALAR_NEG_CLIP_COEF = new ScalarOptional(new Scalar(-args.clipCoef));
    private static final ScalarOptional SCALAR_OPT_CLIP_COEF = new ScalarOptional(new Scalar(args.clipCoef));
    //
    private static final Scalar SCALAR_ENT_COEF = new Scalar(args.entCoef);
    private static final Scalar SCALAR_VF_COEF = new Scalar(args.vfCoef);
    static SyncedVectorEnvironment environment;
    static MinecraftRL model;
    static MinecraftRL.LSTMState initialLSTMState;
    static MinecraftRL.LSTMState nextLstmState;
    static VectorResetResult resetResult;
    static AdamOptions adamOptions;
    static Adam optimizer;
    static long iterationStartTime = System.currentTimeMillis();
    static int step = 0;
    static String logText = "";
    static TrainingLogger trainingLogger;
    static TensorVector modelParameters;
    /** Shape: [numEnvs, Observation.OBSERVATION_SPACE_SIZE] */
    private static Tensor nextObs;
    /** Shape: [numSteps, numEnvs, Observation.OBSERVATION_SPACE_SIZE] */
    private static Tensor observations;
    /** Shape: [numSteps, numEnvs, Action.ACTION_SPACE_SIZE] */
    private static Tensor actions;
    /** Shape: [numSteps, numEnvs] */
    private static Tensor logprobs;
    /** Shape: [numSteps, numEnvs] */
    private static Tensor rewards;
    /** Shape: [numSteps, numEnvs] */
    private static Tensor terminations;
    /** Shape: [numSteps, numEnvs] */
    private static Tensor truncations;
    /** Shape: [numSteps, numEnvs] */
    private static Tensor valuesPreReset;
    /** Shape: [numSteps, numEnvs] */
    private static Tensor values;
    /** Shape: [numEnvs] */
    private static Tensor nextTermination;
    /** Shape: [numEnvs] */
    private static Tensor nextTruncation;
    /** Shape: [numEnvs] */
    private static Tensor nextValuePreReset;
    /** Shape: [numSteps, numEnvs] */
    private static Tensor advantages;
    private static boolean runningInnerLoop = false;
    private static boolean needsPostTickStep = false;
    private static int numTerminations = 0;
    private static int numTruncations = 0;
    private static double lastValueLoss = 0.0;
    private static double lastPolicyLoss = 0.0;
    private static double lastApproxKl = 0.0;
    private static double lastClipFrac = 0.0;
    private static double lastIterationTime = 0.0;
    private static double lastSps = 0.0;
    private static double lastAverageRewards = 0.0;
    private static double lastTotalRewards = 0.0;
    private static int lastBowSelectedSteps = 0;
    private static int lastBowDrawingSteps = 0;
    private static int lastBowFullyDrawnSteps = 0;
    private static int lastShieldUsingSteps = 0;

    public static void setup() {

        // TODO: torch.cuda.amp.autocast for mixed precision training
        // TODO: gradient scaling

        // TODO: torch.compile()
        // TODO: Use pinned memory for a lot of this

        device = new Device("cuda:0");
        trainingLogger = new TrainingLogger(args);

        model = new MinecraftRL(args, device);
        model.loadCheckpoint(args.startingCheckpoint);
        model.to(device, false);
        modelParameters = model.parameters();

        environment = new SyncedVectorEnvironment(args);

        adamOptions = new AdamOptions(args.learningRate);
        adamOptions.eps().put(1e-5);
        optimizer = new Adam(modelParameters, adamOptions);

        TensorOptions deviceTensorOptions = new TensorOptions(device);
        observations = torch.zeros(new long[]{args.numSteps, args.numEnvs, Observation.OBSERVATION_SPACE_SIZE}, deviceTensorOptions);
        actions = torch.zeros(new long[]{args.numSteps, args.numEnvs, Action.ACTION_SPACE_SIZE}, deviceTensorOptions);
        logprobs = torch.zeros(new long[]{args.numSteps, args.numEnvs}, deviceTensorOptions);
        rewards = torch.zeros(new long[]{args.numSteps, args.numEnvs}, deviceTensorOptions);
        terminations = torch.zeros(new long[]{args.numSteps, args.numEnvs}, deviceTensorOptions);
        valuesPreReset = torch.zeros(new long[]{args.numSteps, args.numEnvs}, deviceTensorOptions);
        truncations = torch.zeros(new long[]{args.numSteps, args.numEnvs}, deviceTensorOptions);
        values = torch.zeros(new long[]{args.numSteps, args.numEnvs}, deviceTensorOptions);

        nextTermination = torch.zeros(new long[]{args.numEnvs}, deviceTensorOptions);
        nextTruncation = torch.zeros(new long[]{args.numEnvs}, deviceTensorOptions);
        nextLstmState = new MinecraftRL.LSTMState(
            torch.zeros(new long[]{model.getLSTMLayers(), args.numEnvs, model.getLSTMHiddenSize()}, deviceTensorOptions),
            torch.zeros(new long[]{model.getLSTMLayers(), args.numEnvs, model.getLSTMHiddenSize()}, deviceTensorOptions));
        nextValuePreReset = torch.zeros(new long[]{args.numEnvs}, deviceTensorOptions);


        // Allocate some stuff ahead of time
        advantages = torch.zeros(new long[]{args.numSteps, args.numEnvs}, deviceTensorOptions);
        zerosLikeNumEnvs = torch.zeros(new long[]{args.numEnvs}, deviceTensorOptions).cuda();
        onesLikeNumEnvs = torch.ones(new long[]{args.numEnvs}, deviceTensorOptions).cuda();


        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text(""), () -> logText));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Env. Ready"), () -> environment.isReady()));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Inner Loop"), () -> runningInnerLoop));
        Debugger.mainDebugWindow.addControl(new BooleanControl(Component.text("Run Training"), () -> runTraining, (value) -> runTraining = value));
        Debugger.mainDebugWindow.addControl(new ButtonControl(Component.text("Run Step"), () -> {
            runTraining = true;
            trainingStep();
            runTraining = false;
        }));
        Debugger.mainDebugWindow.addControl(new BooleanControl(Component.text("Sprint"), () -> sprint, (value) -> sprint = value));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Iteration"), () -> iteration));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Step"), () -> step));
        Debugger.mainDebugWindow.addText("");
        Debugger.mainDebugWindow.addText("Arguments:");
        // args
        Debugger.mainDebugWindow.addControl(new CounterControl(Component.text("Num Steps"), () -> args.numSteps, (value) -> args.numSteps = value));
        Debugger.mainDebugWindow.addControl(new CounterControl(Component.text("Num Iterations"), () -> args.numIterations, (value) -> args.numIterations = value));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Num Envs"), () -> args.numEnvs));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Batch Size"), () -> args.batchSize));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Num Mini Batch"), () -> args.numMinibatches));


        CUDAAllocator allocator = torch_cuda.getAllocator();
        DeviceStats deviceStats = torch_cuda.getAllocator().getDeviceStats(device.index());

        System.out.println("torch.cuda_is_available() = " + torch.cuda_is_available());
        System.out.println("torch.cuda_device_count() = " + torch.cuda_device_count());
        System.out.println("torch.hasCUDA() = " + torch.hasCUDA());
        System.out.println("allocator.isHistoryEnabled() = " + allocator.isHistoryEnabled());
        System.out.println("deviceStats.allocation().allocated() = " + deviceStats.allocation().allocated());
        System.out.println("deviceStats.allocation().current() = " + deviceStats.allocation().current());
        System.out.println("deviceStats.allocation().freed() = " + deviceStats.allocation().freed());
        System.out.println("deviceStats.allocation().peak() = " + deviceStats.allocation().peak());
        System.out.println("deviceStats.allocated_bytes().allocated() = " + deviceStats.allocated_bytes().allocated());
        System.out.println("deviceStats.allocated_bytes().current() = " + deviceStats.allocated_bytes().current());
        System.out.println("deviceStats.allocated_bytes().freed() = " + deviceStats.allocated_bytes().freed());
        System.out.println("deviceStats.allocated_bytes().peak() = " + deviceStats.allocated_bytes().peak());


    }

    public static void shutdown() {
        if (trainingLogger != null)
            trainingLogger.close();
        if (nextObs != null) {
            nextObs.close();
            nextObs = null;
        }
        if (nextTermination != null) {
            nextTermination.close();
            nextTermination = null;
        }
        if (nextTruncation != null) {
            nextTruncation.close();
            nextTruncation = null;
        }
        if (nextValuePreReset != null) {
            nextValuePreReset.close();
            nextValuePreReset = null;
        }
        if (nextLstmState != null) {
            nextLstmState.close();
            nextLstmState = null;
        }
    }

    // Placeholder for future implementation
    public static void trainingStep() {
        if (CURRENT_MODE != MachineLearningPlayer.Mode.TRAINING) return;
        if (!runTraining) return;


        if (!environment.isReady()) {
            Bukkit.broadcast(Component.text("Environment is not ready for training."));
            LOGGER.warn("Attempted to run training step, but environment is not ready.");
            runTraining = false;
            return;
        }

        if (resetResult == null) {
            LOGGER.info("Initial environment reset...");
            resetResult = environment.reset();
            Tensor tensor = resetResult.observationsTensor();
            nextObs = tensor.to(device, torch.ScalarType.Float);
            tensor.close();
            resetResult.close();
        }

        if (sprint) {
            Bukkit.getServerTickManager().requestGameToSprint(200);
        } else {
            Bukkit.getServerTickManager().stopSprinting();
        }

        // TODO: do observations every 2 ticks instead of every tick

        try {

            if (!runningInnerLoop) {
                epochSetup();
                if (!runTraining) return;
            }

            runPostTickStep();

            if (step >= args.numSteps) {
                step = 0;
                runningInnerLoop = false;
                finishEpoch();
                return;
            }

            runPreTickStep();
        } catch (Exception e) {
            LOGGER.error("Exception during training step at iteration {}, step {}: {}", iteration, step, e.getMessage());
            e.printStackTrace();
            runTraining = false;
        }
    }

    public static void epochSetup() {
        logText = "Epoch Setup...";

        iterationStartTime = System.currentTimeMillis();

        if (iteration >= args.numIterations + 1) {
            LOGGER.info("Maximum iterations reached. Stopping training.");
            runTraining = false;
            return;
        }

        // Reset iteration stats
        numTerminations = 0;
        numTruncations = 0;
        lastBowSelectedSteps = 0;
        lastBowDrawingSteps = 0;
        lastBowFullyDrawnSteps = 0;
        lastShieldUsingSteps = 0;


        /*
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
         */
        if (initialLSTMState != null) {
            initialLSTMState.close();
        }
        initialLSTMState = nextLstmState.clone();

        /*
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
         */
        if (args.annealLr) {
            double frac = 1.0 - (iteration - 1.0) / args.numIterations;
            double lrNow = frac * args.learningRate;
            OptimizerOptions options = optimizer.param_groups().get(0).options();
            options.set_lr(lrNow);
            options.close();
        }
    }

    public static void runPreTickStep() {
        PointerScope scope = new PointerScope();
        logText = "Run Steps...";

        /*
        for step in range(0, args.num_steps):
         */
        runningInnerLoop = true;

        /*
        global_step += args.num_envs
        obs[step] = next_obs
        dones[step] = next_done
         */

        // globalStep += args.numEnvs;
        try (Tensor slice = observations.get(step)) {
            Tensor detached = nextObs.detach();
            slice.copy_(detached);
            detached.close();
        }
        try (Tensor doneSlice = terminations.get(step)) {
            doneSlice.copy_(nextTermination);
        }
        try (Tensor doneSlice = truncations.get(step)) {
            doneSlice.copy_(nextTruncation);
        }
        try (Tensor valueSlice = valuesPreReset.get(step)) {
            valueSlice.copy_(nextValuePreReset);
        }

        /*
        with torch.no_grad():
            action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                next_obs, next_lstm_state, next_done
            )
            values[step] = value.flatten()
         */
        logText = "Getting action and value for step " + step;

        // with torch.no_grad():
        NoGradGuard noGrad = new NoGradGuard();
        MinecraftRL.ActionAndValue actionResult = model.getActionAndValue(nextObs, nextLstmState, nextTermination);
        nextLstmState.copy_(actionResult.lstmState());

        try (Tensor slice = values.get(step)) {
            Tensor value = actionResult.value();
            Tensor flattened = value.flatten(); // (numEnvs, 1) -> (numEnvs,)

            slice.copy_(flattened);

            value.close();
            flattened.close();
        }

        noGrad.close();

        /*
        actions[step] = action
        logprobs[step] = logprob
         */

        Tensor actionResultAction = actionResult.action();
        try (Tensor actionSlice = actions.get(step)) {
            Tensor detachedAction = actionResultAction.detach();
            actionSlice.copy_(detachedAction);
            detachedAction.close();
        }
        try (Tensor logProbSlice = logprobs.get(step)) {
            logProbSlice.copy_(actionResult.totalLogProbs());
        }

        logText = "Stepping environment for step " + step;
        /*
        next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
         */

        Tensor actionTensor = actionResultAction.cpu();
        actionResultAction.close();
        environment.preTickStep(actionTensor);

        actionResult.close();
        actionTensor.close();

        needsPostTickStep = true;
        scope.close();
    }

    ///  A server tick happens between these functions

    public static void runPostTickStep() {
        if (!needsPostTickStep) {
            LOGGER.warn("runPostTickStep called without a preTickStep. Skipping step.");
            return;
        }

        PointerScope scope = new PointerScope();

        /*
        next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        next_done = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
         */
        VectorStepResult stepResult = environment.postTickStep(model, nextLstmState, device);

        nextObs.close();
        Tensor rawObs = stepResult.observationsTensor();
        nextObs = rawObs.to(device, torch.ScalarType.Float);
        nextObs.retainReference();
        rawObs.close();

        nextTermination.close();
        Tensor cpuTerminated = Tensor.create(stepResult.terminated());
        nextTermination = cpuTerminated.to(device, torch.ScalarType.Float);
        cpuTerminated.close();
        nextTermination.retainReference();


        nextTruncation.close();
        Tensor cpuTruncated = Tensor.create(stepResult.truncated());
        nextTruncation = cpuTruncated.to(device, torch.ScalarType.Float);
        cpuTruncated.close();
        nextTruncation.retainReference();

        Tensor cpuStepRewards = Tensor.create(stepResult.rewards());
        Tensor gpuStepRewards = cpuStepRewards.to(device, torch.ScalarType.Float);
        Tensor newRewardsTensor = gpuStepRewards.view(-1); // (numEnvs,) -> (numEnvs,)
        try (Tensor rewardSlice = rewards.get(step)) {
            rewardSlice.copy_(newRewardsTensor);
        }
        newRewardsTensor.close();
        gpuStepRewards.close();
        cpuStepRewards.close();

        nextValuePreReset.close();
        nextValuePreReset = stepResult.nextValuePreReset();
        nextValuePreReset.retainReference();

        numTerminations += stepResult.numTerminations();
        numTruncations += stepResult.numTruncations();
        lastBowSelectedSteps += stepResult.bowSelectedSteps();
        lastBowDrawingSteps += stepResult.bowDrawingSteps();
        lastBowFullyDrawnSteps += stepResult.bowFullyDrawnSteps();
        lastShieldUsingSteps += stepResult.shieldUsingSteps();

        stepResult.close();

        // TODO: Handle logging of episodic returns and lengths

        scope.close();

        step++;
    }

    public static void finishEpoch() {
        logText = "Finish Epoch...";

        PointerScope scope = new PointerScope();

        /*
        with torch.no_grad():
         */
        AutogradState.get_tls_state().set_grad_mode(false); // with torch.no_grad():

        /*
        next_value = agent.get_value(
            next_obs,
            next_lstm_state,
            next_done,
        ).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
         */

        Tensor nextValueRaw = model.getValue(nextObs, nextLstmState, nextTermination);
        Tensor nextValue = nextValueRaw.reshape(-1); // (numEnvs, 1) -> (numEnvs,)
        nextValueRaw.close();
        advantages.zero_();

        /*
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
         */

        Tensor lastGAELam = torch.zeros(new long[]{args.numEnvs}, new TensorOptions(device));

        // TODO: Test whether pulling to the .cpu() and running java math is faster than doing it in torch

        for (int t = args.numSteps - 1; t >= 0; t--) {
            PointerScope loopScope = new PointerScope();

            Tensor terminated;
            Tensor truncated;
            Tensor valueTrunc;
            Tensor valueNormal;
            if (t == args.numSteps - 1) {
                terminated = nextTermination;
                truncated = nextTruncation;
                valueTrunc = nextValuePreReset;
                valueNormal = nextValue;
            } else {
                terminated = terminations.get(t + 1);
                truncated = truncations.get(t + 1);
                valueTrunc = valuesPreReset.get(t + 1);
                valueNormal = values.get(t + 1);
            }

            Tensor nextNonTerminal = onesLikeNumEnvs.sub(terminated);
            Tensor condition = truncated.eq(SCALAR_ONE);
            Tensor nextValues = torch.where(condition, valueTrunc, valueNormal).mul(nextNonTerminal);

            Tensor mul1 = nextValues.mul(SCALAR_GAMMA);
            Tensor add1 = rewards.get(t).add(mul1);
            Tensor delta = add1.sub(values.get(t));

            Tensor mul22 = nextNonTerminal.mul(SCALAR_GAMMA_GAE_LAMBDA);
            Tensor mul2 = mul22.mul(lastGAELam);
            Tensor advantage = delta.add(mul2);

            advantages.get(t).copy_(advantage);
            lastGAELam.copy_(advantage);

            mul1.close();
            add1.close();
            delta.close();
            mul22.close();
            mul2.close();
            advantage.close();
            nextNonTerminal.close();
            nextValues.close();
            if (t != args.numSteps - 1) {
                terminated.close();
                truncated.close();
                valueTrunc.close();
                valueNormal.close();
            }

            loopScope.close();
        }
        nextValue.close();

        /*
        returns = advantages + values
         */

        Tensor returns = advantages.add(values);

        AutogradState.get_tls_state().set_grad_mode(true);

        /*
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
         */

        /// [numSteps*numEnvs, obs_space]
        Tensor bObs = observations.reshape(-1, Observation.OBSERVATION_SPACE_SIZE); // (numSteps, numEnvs, OBSERVATION_SPACE_SIZE) -> (numSteps*numEnvs, OBSERVATION_SPACE_SIZE)
        /// [numSteps*numEnvs]
        Tensor bLogProbs = logprobs.reshape(-1); // (numSteps, numEnvs) -> (numSteps*numEnvs,)
        /// [numSteps*numEnvs, action_space]
        Tensor bActions = actions.reshape(-1, Action.ACTION_SPACE_SIZE); // (numSteps, numEnvs, ACTION_SPACE_SIZE) -> (numSteps*numEnvs, ACTION_SPACE_SIZE)
        /// [numSteps*numEnvs]
        Tensor bDones = terminations.reshape(-1); // (numSteps, numEnvs) -> (numSteps*numEnvs,)
        /// [numSteps*numEnvs]
        Tensor bAdvantages = advantages.reshape(-1); // (numSteps, numEnvs) -> (numSteps*numEnvs,)
        /// [numSteps*numEnvs]
        Tensor bReturns = returns.reshape(-1); // (numSteps, numEnvs) -> (numSteps*numEnvs,)
        /// [numSteps*numEnvs]
        Tensor bValues = values.reshape(-1); // (numSteps, numEnvs) -> (numSteps*numEnvs,)


        /*
        assert args.num_envs % args.num_minibatches == 0
        */

        assert args.numEnvs % args.numMinibatches == 0 : "Number of environments must be divisible by number of minibatches.";

        /*
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
         */

        int envsPerBatch = args.numEnvs / args.numMinibatches;
        Tensor envinds = torch.arange(SCALAR_NUM_ENVS, new TensorOptions(device)); // Shape: [numEnvs]
        Tensor flatinds = torch.arange(SCALAR_BATCH_SIZE, new TensorOptions(device)).reshape(args.numSteps, args.numEnvs); // (numSteps*numEnvs,) -> (numSteps, numEnvs)

        /*
        clipfracs = []
        for epoch in range(args.update_epochs):
            */

        double clipFracAccum = 0;
        int numClipFracs = 0;

        double vLoss = 0;
        double pgLoss = 0;
        double entropyLoss = 0;
        double approxKl = 0;
        double oldApproxKl = 0;


        for (int epoch = 0; epoch < args.updateEpochs; epoch++) {
            PointerScope epochScope = new PointerScope();

            /*
            np.random.shuffle(envinds)
             */
            Tensor randperm = torch.randperm(args.numEnvs, new TensorOptions(device));
            Tensor oldEnvIds = envinds;
            envinds = envinds.index_select(0, randperm); // (numEnvs,) -> (numEnvs,)
            envinds.retainReference();
            oldEnvIds.close();
            randperm.close();

            /*
            for start in range(0, args.num_envs, envsperbatch):
             */

            for (int start = 0; start < args.numEnvs; start += envsPerBatch) {
                PointerScope batchScope = new PointerScope();
                /*
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
                */

                Tensor mbenvinds = envinds.narrow(0, start, envsPerBatch); // (numEnvs,) -> (envsPerBatch,)
                Tensor flatindsindexselect = flatinds.index_select(1, mbenvinds);
                Tensor mb_inds = flatindsindexselect.ravel(); // (numSteps, numEnvs) -> (numSteps, envsPerBatch) -> (numSteps*envsPerBatch,)
                flatindsindexselect.close();

                /*
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions[mb_inds],
                )
                 */

                Tensor lstmStateHidden = initialLSTMState.hiddenState().index_select(1, mbenvinds); // (1, numEnvs, hidden_size) -> (1, envsPerBatch, hidden_size)
                Tensor lstmStateCell = initialLSTMState.cellState().index_select(1, mbenvinds); // (1, numEnvs, hidden_size) -> (1, envsPerBatch, hidden_size)

                MinecraftRL.ActionAndValue actionAndValueResult = model.getActionAndValue(
                    bObs.index_select(0, mb_inds), // (numSteps*numEnvs, OBSERVATION_SPACE_SIZE) -> (batchSize, OBSERVATION_SPACE_SIZE)
                    new MinecraftRL.LSTMState(lstmStateHidden, lstmStateCell),
                    bDones.index_select(0, mb_inds), // (numSteps*numEnvs,) -> (batchSize,)
                    bActions.index_select(0, mb_inds) // (numSteps*numEnvs, ACTION_SPACE_SIZE) -> (batchSize, ACTION_SPACE_SIZE)
                );
                lstmStateHidden.close();
                lstmStateCell.close();

                /*
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                */

                Tensor totalLogProbs = actionAndValueResult.totalLogProbs();
                Tensor selected = bLogProbs.index_select(0, mb_inds); // (numSteps*numEnvs,) -> (batchSize,)
                Tensor logRatio = totalLogProbs.sub(selected);
                Tensor ratio = logRatio.exp();

                totalLogProbs.close();
                selected.close();

                /*
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                 */

                AutogradState.get_tls_state().set_grad_mode(false); // with torch.no_grad():

                Tensor oldApproxKlTensor = logRatio.neg().mean();
                Tensor ratioSub = ratio.sub(SCALAR_ONE);
                Tensor subLogRa = ratioSub.sub(logRatio);
                Tensor approxKlTensor = subLogRa.mean();
                Tensor subAbs = ratioSub.abs();
                Tensor subAbsGT = subAbs.gt(SCALAR_CLIP_COEF);
                Tensor toFloat = subAbsGT.to(torch.ScalarType.Float);
                Tensor clipFracTensor = toFloat.mean();

                Scalar oldApproxKlScalar = oldApproxKlTensor.item();
                oldApproxKl = oldApproxKlScalar.toDouble();
                oldApproxKlScalar.close();
                Scalar approxKlScalar = approxKlTensor.item();
                approxKl = approxKlScalar.toDouble();
                approxKlScalar.close();
                Scalar clipFracScalar = clipFracTensor.item();
                clipFracAccum += clipFracScalar.toDouble();
                clipFracScalar.close();
                numClipFracs++;

                oldApproxKlTensor.close();
                ratioSub.close();
                subLogRa.close();
                approxKlTensor.close();
                subAbs.close();
                subAbsGT.close();
                toFloat.close();
                clipFracTensor.close();

                AutogradState.get_tls_state().set_grad_mode(true);

                /*
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                 */

                Tensor mbAdvantages = bAdvantages.index_select(0, mb_inds); // (numSteps*numEnvs,) -> (batchSize,)

                if (args.normAdv) {
                    Tensor mean = mbAdvantages.mean();
                    Tensor std = mbAdvantages.std();
                    Tensor shiftedAdvantages = mbAdvantages.sub(mean);
                    Tensor stdWithEps = std.add(SCALAR_1E_8);
                    Tensor normalizedAdvantages = shiftedAdvantages.div(stdWithEps);
                    mbAdvantages.close();
                    mean.close();
                    std.close();
                    shiftedAdvantages.close();
                    stdWithEps.close();
                    mbAdvantages = normalizedAdvantages;
                }

                /*
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                */


                Tensor pgLoss1 = mbAdvantages.neg().mul(ratio);
                Tensor pgLoss2 = mbAdvantages.neg().mul(
                    torch.clamp(ratio, SCALAR_1_SUB_CLIP_COEF, SCALAR_1_ADD_CLIP_COEF)
                );
                Tensor pgLossTensor = torch.max(pgLoss1, pgLoss2).mean();

                /*
                newvalue = newvalue.view(-1)
                */

                Tensor newvalue = actionAndValueResult.value().view(-1); // (batch, 1) -> (batch,)

                /*
                if args.clip_vloss:
                 */
                Tensor bReturnsMbInds = bReturns.index_select(0, mb_inds); // (numSteps*numEnvs,) -> (batchSize,)
                Tensor bValueMbInds = bValues.index_select(0, mb_inds); // (numSteps*numEnvs,) -> (batchSize,)


                Tensor vLossTensor;

                if (args.clipVloss) {
                    /*
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                     */

                    Tensor vLossUnclipped = newvalue.sub(bReturnsMbInds).square();

                    Tensor vClipped = bValueMbInds.add(
                        torch.clamp(
                            newvalue.sub(bValueMbInds),
                            SCALAR_NEG_CLIP_COEF,
                            SCALAR_OPT_CLIP_COEF
                        )
                    );

                    Tensor vLossClipped = vClipped.sub(bReturnsMbInds).square();
                    Tensor vLossMax = torch.max(vLossUnclipped, vLossClipped);
                    vLossTensor = vLossMax.mean().mul(SCALAR_0_5);
                    vLossUnclipped.close();
                    vClipped.close();
                    vLossClipped.close();
                    vLossMax.close();
                } else {
                    /*
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                     */
                    vLossTensor = newvalue.sub(bReturnsMbInds).square().mean().mul(SCALAR_0_5);
                }

                /*
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                */

                Tensor entropyLossTensor = actionAndValueResult.totalEntropy().mean();
                Tensor loss = pgLossTensor.sub(
                    entropyLossTensor.mul(SCALAR_ENT_COEF)
                ).add(
                    vLossTensor.mul(SCALAR_VF_COEF)
                );

                /*
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                */

                optimizer.zero_grad();
                loss.backward();
                torch.clip_grad_norm_(modelParameters, args.maxGradNorm);
                optimizer.step();

                actionAndValueResult.close();
                mbenvinds.close();
                mb_inds.close();
                mbAdvantages.close();
                logRatio.close();
                ratio.close();
                pgLoss1.close();
                pgLoss2.close();
                newvalue.close();
                bReturnsMbInds.close();
                bValueMbInds.close();
                loss.close();

                Scalar vLossScalar = vLossTensor.item();
                vLoss = vLossScalar.toDouble();
                vLossScalar.close();
                Scalar pgLossScalar = pgLossTensor.item();
                pgLoss = pgLossScalar.toDouble();
                pgLossScalar.close();
                Scalar entropyLossScalar = entropyLossTensor.item();
                entropyLoss = entropyLossScalar.toDouble();
                entropyLossScalar.close();
                vLossTensor.close();
                pgLossTensor.close();
                entropyLossTensor.close();

                batchScope.close();
            }

            epochScope.close();

            /*
            if args.target_kl is not None and approx_kl > args.target_kl:
                break
             */
            if (args.targetKl != null && approxKl > args.targetKl) {
                LOGGER.warn("Target KL ({}) exceeded ({}). Breaking from update epochs.", args.targetKl, approxKl);
                break;
            }

        }



        /*
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
         */
        // Tensor yPred = bValues.cpu();
        // Tensor yTrue = bReturns.cpu();
        // double varY = yTrue.var().item().toDouble();
        // double explainedVar;
        // if (varY == 0) {
        //     explainedVar = Double.NaN;
        // } else {
        //     Scalar scalar = new Scalar(varY);
        //     Tensor sub = yTrue.sub(yPred);
        //     Tensor var = sub.var();
        //     Tensor div = var.div(scalar);
        //     Scalar item = div.item();
        //     explainedVar = 1.0 - item.toDouble();
        //
        //     scalar.close();
        //     sub.close();
        //     var.close();
        //     div.close();
        //     item.close();
        // }

        LOGGER.info("==================== Finished Epoch for Iteration:      {} ====================", iteration - 1);

        // Write all sites to file for debugging memory leaks
        File logFile = new File("training/" + args.experimentId + "/sites/");
        logFile.mkdirs();
        String contents = "Native Allocation Tracer Sites for Iteration " + iteration + "\n";
        for (NativeAllocationTracer.Site site : NativeAllocationTracer.getSites()) {
            contents += site.toString() + "\n";
        }
        try {
            Files.writeString(logFile.toPath().resolve(iteration + "_sites.txt"), contents);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // SnapshotInfo allocatorSnapshot = torch_cuda.getAllocator().snapshot();
        // allocatorSnapshot.segments()

        if (iteration % 20 == 0) {
            SystemStats.HardwareMetrics hw = SystemStats.snapshot(device.index());

            LOGGER.info("GPU: {}% | Mem: {} / {} | Temp: {}C",
                hw.gpuUtil(),
                SystemStats.formatBytes(hw.gpuMemUsed()),
                SystemStats.formatBytes(hw.gpuMemTotal()),
                hw.gpuTemp()
            );
            LOGGER.info("CPU: {}% | Heap: {} | Native(JNI): {}",
                hw.cpuLoad(),
                SystemStats.formatBytes(hw.javaHeapUsed()),
                SystemStats.formatBytes(hw.javaNativeUsed(), 8)
            );

            try (PointerScope scopeLogger = new PointerScope()) {
                OptimizerOptions options = optimizer.param_groups().get(0).options();
                double learningRate = options.get_lr();
                options.close();

                double valueLoss = vLoss;
                double policyLoss = pgLoss;
                double entropyLossDouble = entropyLoss;

                double oldApproxKlVal = oldApproxKl;
                double approxKlVal = approxKl;
                double clipfrac = numClipFracs > 0 ? clipFracAccum / numClipFracs : 0;
                double iterationTime = ((System.currentTimeMillis() - iterationStartTime) / 1000.0);
                double sps = ((args.batchSize) / iterationTime);
                double averageRewards = rewards.mean().item().toDouble();
                double totalRewards = rewards.sum().item().toDouble();

                lastValueLoss = valueLoss;
                lastPolicyLoss = policyLoss;
                lastApproxKl = approxKlVal;
                lastClipFrac = clipfrac;
                lastIterationTime = iterationTime;
                lastSps = sps;
                lastAverageRewards = averageRewards;
                lastTotalRewards = totalRewards;

                trainingLogger.logStep(
                    iteration,
                    learningRate,
                    valueLoss,
                    policyLoss,
                    entropyLossDouble,
                    oldApproxKlVal,
                    approxKlVal,
                    clipfrac,
                    // explainedVar,
                    0,
                    iterationTime,
                    sps,
                    numTerminations,
                    numTruncations,
                    averageRewards,
                    totalRewards,
                    lastBowSelectedSteps,
                    lastBowDrawingSteps,
                    lastBowFullyDrawnSteps,
                    lastShieldUsingSteps,
                    hw.gpuMemUsed(),
                    hw.gpuMemTotal(),
                    hw.gpuUtil(),
                    hw.gpuTemp(),
                    hw.torchAllocatedBytesCurrent(),
                    hw.torchAllocatedBytesPeak(),
                    hw.torchReservedBytesCurrent(),
                    hw.torchReservedBytesPeak(),
                    hw.torchActiveBytesCurrent(),
                    hw.torchActiveBytesPeak(),
                    hw.torchInactiveSplitBytesCurrent(),
                    hw.torchInactiveSplitBytesPeak(),
                    hw.torchRequestedBytesCurrent(),
                    hw.torchRequestedBytesPeak(),
                    hw.torchNumAllocRetries(),
                    hw.torchNumOoms(),
                    hw.javaNativeUsed(),
                    hw.javaCppRegisteredBytes(),
                    hw.javaCppRegisteredCount(),
                    hw.javaHeapUsed(),
                    hw.osAvailablePhysicalBytes(),
                    hw.osTotalPhysicalBytes(),
                    hw.javaCppDeallocatorThreadAlive()
                );
                LOGGER.info(
                    "Iteration {}, LR: {}, VLoss: {}, PLoss: {}, Entropy: {}, OldKL: {}, KL: {}, ClipFrac: {}, ExplVar: {}, IterTime: {}s, SPS: {}, AvgRewards: {}, TotRewards: {}, BowSelected: {}, BowDrawing: {}, BowFull: {}, ShieldUsing: {}",
                    iteration,
                    learningRate,
                    valueLoss,
                    policyLoss,
                    entropyLossDouble,
                    oldApproxKlVal,
                    approxKlVal,
                    clipfrac,
                    // explainedVar,
                    0,
                    iterationTime,
                    sps,
                    averageRewards,
                    totalRewards,
                    lastBowSelectedSteps,
                    lastBowDrawingSteps,
                    lastBowFullyDrawnSteps,
                    lastShieldUsingSteps
                );
            } catch (Exception e) {
                LOGGER.error("Failed to log training metrics: {}", e.getMessage());
            }
            LOGGER.memory();
        }

        nextValue.close();
        lastGAELam.close();
        returns.close();
        bObs.close();
        bLogProbs.close();
        bActions.close();
        bDones.close();
        bAdvantages.close();
        bReturns.close();
        bValues.close();
        envinds.close();
        flatinds.close();
        if (initialLSTMState != null) {
            initialLSTMState.close();
            initialLSTMState = null;
        }
        // yPred.close();
        // yTrue.close();

        scope.close();

        iteration++;

        if (iteration % 100 == 0) {
            // allocator.snapshot();
            model.saveCheckpoint(iteration);
        }

        if (iteration % 500 == 0) {
            LOGGER.info("Cleaning up native memory...");
            System.gc();
            Pointer.deallocateReferences();
        }
    }

    public static void reset() {
        environment.reset();
    }

    public static int createCheckpoint() {
        model.saveCheckpoint(iteration);
        return iteration;
    }

    public static String getTrainingSummary() {
        return String.format(
            "Train: iter=%d step=%d running=%s ready=%s avgR=%.2f totalR=%.2f vLoss=%.4f pLoss=%.4f kl=%.4f clip=%.4f sps=%.1f, lastIterTime=%.2fs",
            iteration,
            step,
            runTraining,
            environment != null && environment.isReady(),
            lastAverageRewards,
            lastTotalRewards,
            lastValueLoss,
            lastPolicyLoss,
            lastApproxKl,
            lastClipFrac,
            lastSps,
            lastIterationTime
        );
    }
}
