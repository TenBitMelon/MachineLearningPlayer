package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.debugger.Debugger;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.BooleanControl;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.ButtonControl;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.CounterControl;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.VariableControl;
import com.tenbitmelon.machinelearningplayer.environment.Action;
import com.tenbitmelon.machinelearningplayer.environment.Observation;
import net.kyori.adventure.text.Component;
import org.bukkit.Bukkit;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.cuda.CUDAAllocator;
import org.bytedeco.pytorch.cuda.DeviceStats;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch_cuda;

import static com.tenbitmelon.machinelearningplayer.MachineLearningPlayer.LOGGER;

public class TrainingManager {

    private static final Scalar SCALAR_ONE = new Scalar(1.0);
    private static final Scalar SCALAR_1E_8 = new Scalar(1e-8);
    private static final Scalar SCALAR_0_5 = new Scalar(0.5);
    static public boolean runTraining = false;
    static public boolean sprint = false;
    public static Device device;
    static ExperimentConfig args = ExperimentConfig.getInstance();
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
    private static final Scalar SCALAR_ENT_COEF = new Scalar(args.entCoef);
    private static final Scalar SCALAR_VF_COEF = new Scalar(args.vfCoef);
    //
    static SyncedVectorEnvironment environment;
    static MinecraftRL model;
    static MinecraftRL.LSTMState initialLSTMState;
    static MinecraftRL.LSTMState nextLstmState;
    static VectorResetResult resetResult;
    static AdamWOptions adamOptions;
    static AdamW optimizer;
    static long iterationStartTime = System.currentTimeMillis();
    static int iteration = 1;
    static int step = 0;
    static String logText = "";
    static TrainingLogger trainingLogger;
    static CUDAAllocator allocator;
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
    private static Tensor dones;
    /** Shape: [numSteps, numEnvs] */
    private static Tensor values;
    /** Shape: [numEnvs] */
    private static Tensor nextDone;
    /** Shape: [numSteps, numEnvs] */
    private static Tensor advantages;
    /** Shape: [numEnvs] */
    private static Tensor zerosLikeNumEnvs;
    /** Shape: [numEnvs] */
    private static Tensor onesLikeNumEnvs;
    private static boolean runningInnerLoop = false;
    private static boolean needsPostTickStep = false;
    private static int numTerminations = 0;

    public static void setup() {
        // LOGGER.debug("--- Setting up Training Manager ---");
        // LOGGER.debug("ExperimentConfig loaded. numEnvs={}, numSteps={}", args.numEnvs, args.numSteps);
        environment = new SyncedVectorEnvironment(args);
        // LOGGER.debug("SyncedVectorEnvironment initialized.");
        model = new MinecraftRL(environment);
        model.loadCheckpoint(args.startingCheckpoint);
        device = new Device("cuda:0");
        model.to(device, false);
        modelParameters = model.parameters();
        LOGGER.memory();

        trainingLogger = new TrainingLogger(args, "logs/training");

        // LOGGER.debug("MinecraftRL model initialized.");

        // Device device = args.cuda ? torch.device(torch.kCUDA) : torch.device(torch.kCPU);
        System.out.println("torch.cuda_is_available() = " + torch.cuda_is_available());
        System.out.println("torch.cuda_device_count() = " + torch.cuda_device_count());
        System.out.println("torch.hasCUDA() = " + torch.hasCUDA());
        // Get GPU memory
        allocator = torch_cuda.getAllocator();
        System.out.println("allocator.isHistoryEnabled() = " + allocator.isHistoryEnabled());
        // allocator.snapshot();
        DeviceStats deviceStats = torch_cuda.getAllocator().getDeviceStats(device.index());
        System.out.println("deviceStats.allocation().allocated() = " + deviceStats.allocation().allocated());
        System.out.println("deviceStats.allocation().current() = " + deviceStats.allocation().current());
        System.out.println("deviceStats.allocation().freed() = " + deviceStats.allocation().freed());
        System.out.println("deviceStats.allocation().peak() = " + deviceStats.allocation().peak());
        System.out.println("deviceStats.allocated_bytes().allocated() = " + deviceStats.allocated_bytes().allocated());
        System.out.println("deviceStats.allocated_bytes().current() = " + deviceStats.allocated_bytes().current());
        System.out.println("deviceStats.allocated_bytes().freed() = " + deviceStats.allocated_bytes().freed());
        System.out.println("deviceStats.allocated_bytes().peak() = " + deviceStats.allocated_bytes().peak());


        nextLstmState = new MinecraftRL.LSTMState(
            torch.zeros(model.getLSTMLayers(), args.numEnvs, model.lstmHiddenSize).cuda(),
            torch.zeros(model.getLSTMLayers(), args.numEnvs, model.lstmHiddenSize).cuda());
        LOGGER.memory();
        // LOGGER.debug("Initial LSTM state created. Hidzzden shape: {}, Cell shape: {}", nextLstmState.hiddenState().shape(), nextLstmState.cellState().shape());

        // TODO: torch.cuda.amp.autocast for mixed precision training
        // TODO: gradient scaling

        // TODO: torch.compile()


        adamOptions = new AdamWOptions(args.learningRate);
        optimizer = new AdamW(modelParameters, adamOptions);
        adamOptions.eps().put(1e-7);
        LOGGER.memory();
        // LOGGER.debug("Adam optimizer initialized with learning rate: {}", args.learningRate);

        nextDone = torch.zeros(args.numEnvs).cuda();
        LOGGER.memory();
        // LOGGER.debug("Initial 'nextDone' tensor created. Shape: {}", nextDone.shape());

        // TODO: Use pinned memory for a lot of this
        observations = torch.zeros(args.numSteps, args.numEnvs, Observation.OBSERVATION_SPACE_SIZE).cuda();
        actions = torch.zeros(args.numSteps, args.numEnvs, Action.ACTION_SPACE_SIZE).cuda();
        logprobs = torch.zeros(args.numSteps, args.numEnvs).cuda();
        rewards = torch.zeros(args.numSteps, args.numEnvs).cuda();
        dones = torch.zeros(args.numSteps, args.numEnvs).cuda();
        values = torch.zeros(args.numSteps, args.numEnvs).cuda();
        LOGGER.memory();
        // LOGGER.debug("Storage tensors initialized. Shape (e.g., observations): {}", observations.shape());

        advantages = torch.zeros(args.numSteps, args.numEnvs).cuda();
        zerosLikeNumEnvs = torch.zeros(args.numEnvs).cuda();
        onesLikeNumEnvs = torch.ones(args.numEnvs).cuda();
        LOGGER.memory();

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


        // LOGGER.debug("--- Training Manager setup complete ---");
    }

    public static void shutdown() {
        if (trainingLogger != null)
            trainingLogger.close();
    }

    // Placeholder for future implementation
    public static void trainingStep() {
        if (!runTraining) {
            return;
        }
        // LOGGER.debug("--- Tick ---");

        if (!environment.isReady()) {
            Bukkit.broadcast(Component.text("Environment is not ready for training."));
            LOGGER.warn("Attempted to run training step, but environment is not ready.");
            runTraining = false;
            return;
        }

        if (resetResult == null) {
            LOGGER.info("Initial environment reset...");
            resetResult = environment.reset();

            // Log
            // LOGGER.debug("Environment reset complete. Observations shape: {}, Infos length: {}", resetObservations.shape(), resetResult.infos().length);
            // LOGGER.debug("Initial observations: {}", tensorString(resetObservations));
            nextObs = resetResult.observationsTensor().cuda();
            System.out.println("nextObs.device().type() = " + nextObs.device().type());
            LOGGER.memory();
            // LOGGER.debug("Environment reset complete. Initial observation shape: {}", nextObs.shape());


        }

        LOGGER.memory();
        if (sprint) {
            Bukkit.getServerTickManager().requestGameToSprint(200);
        } else {
            Bukkit.getServerTickManager().stopSprinting();
        }

        // TODO: do observations every 2 ticks instead of every tick


        LOGGER.memory();
        if (!runningInnerLoop) {
            epochSetup();
            LOGGER.memory();
            if (!runTraining) return;
        }
        LOGGER.memory();
        runPostTickStep();
        LOGGER.memory();
        runPreTickStep();
        LOGGER.memory();
    }

    public static void epochSetup() {
        LOGGER.memory();
        // LOGGER.info("==================== Starting Epoch Setup for Iteration: {} ====================", iteration);
        logText = "Epoch Setup...";

        iterationStartTime = System.currentTimeMillis();


        // for iteration in range(1, args.num_iterations + 1):

        // if (iteration >= 2) {
        if (iteration >= args.numIterations + 1) {
            // LOGGER.debug("Maximum iterations reached. Stopping training.");
            runTraining = false;
            return;
        }

        numTerminations = 0;


            /*
            initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
             */
        initialLSTMState = nextLstmState.clone();
        LOGGER.memory();
        // LOGGER.debug("Cloned LSTM state for the new epoch.");

            /*
            if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
             */
        if (args.annealLr) {
            double frac = 1.0 - (iteration - 1.0) / args.numIterations;
            double lrNow = frac * args.learningRate;
            optimizer.param_groups().get(0).options().set_lr(lrNow);
            // LOGGER.debug("Annealed learning rate for iteration {}: {}", iteration, lrNow);
        }
        LOGGER.memory();
    }

    public static void runPreTickStep() {
        LOGGER.memory();
        // LOGGER.debug("[===] Running steps for iteration {}, step {}", iteration, step);
        logText = "Run Steps...";

        // for step in range(0, args.num_steps):
        runningInnerLoop = true;

        if (step >= args.numSteps) {
            // LOGGER.debug("Completed all {} steps for iteration {}. Finishing epoch.", args.numSteps, iteration);
            step = 0;
            runningInnerLoop = false;
            finishEpoch();
            return;
        }

                /*
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                 */

        // globalStep += args.numEnvs;
        // LOGGER.debug("Next Obs: {}", tensorString(nextObs));
        // LOGGER.debug("Observations before storing: {}", tensorString(observations));
        // LOGGER.debug("Observations after storing: {}", tensorString(observations));
        observations.get(step).copy_(nextObs.detach());
        dones.get(step).copy_(nextDone);
        LOGGER.memory();
        // LOGGER.debug("[Step {}] Stored next_obs (shape: {}) and next_done (shape: {})", step, nextObs.shape(), nextDone.shape());

                /*
                with torch.no_grad():
                    action, action_dict, logprob, _, value, next_lstm_state = (
                        agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                    )
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                 */
        logText = "Getting action and value for step " + step;
        // LOGGER.debug("[Step {}] Calling model.getActionAndValue...", step);

        AutogradState.get_tls_state().set_grad_mode(false);
        MinecraftRL.ActionAndValue actionResult = model.getActionAndValue(nextObs, nextLstmState, nextDone);
        values.get(step).copy_(actionResult.value().flatten());
        AutogradState.get_tls_state().set_grad_mode(true);
        LOGGER.memory();

        actions.get(step).copy_(actionResult.action().detach());
        logprobs.get(step).copy_(actionResult.totalLogProbs());
        LOGGER.memory();
        // LOGGER.debug("[Step {}] Stored values (shape: {}), actions (shape: {}), logprobs (shape: {})", step, values.get(step).shape(), actions.get(step).shape(), logprobs.get(step).shape());

                /*
                next_obs, reward, terminations, truncations, infos = envs.step(action_dict)
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                    next_done
                ).to(device)
                 */

        logText = "Stepping environment for step " + step;
        // LOGGER.debug("[Step {}] Stepping environment with action (shape: {})", step, actionResult.action().shape());

        // TODO: convert items to booleans and bitpack them into ints for faster transfer
        Tensor actionTensor = actionResult.action().cpu();
        environment.preTickStep(actionTensor);
        needsPostTickStep = true;
        LOGGER.memory();
    }

    ///  A server tick happens between these functions

    public static void runPostTickStep() {
        LOGGER.memory();
        if (!needsPostTickStep) {
            LOGGER.warn("runPostTickStep called without a preTickStep. Skipping step.");
            return;
        }
        VectorStepResult stepResult = environment.postTickStep();
        // TODO: Better typing for CUDA transfer and tensor creation
        nextDone = Tensor.create(stepResult.logicalOrTerminationsAndTruncations()).cuda();
        rewards.get(step).copy_(Tensor.create(stepResult.rewards()).cuda());
        nextObs = stepResult.observationsTensor().cuda();
        numTerminations += stepResult.numTerminations();
        LOGGER.memory();
        // LOGGER.debug("Next obs: {}", tensorString(nextObs));
        // LOGGER.debug("[Step {}] Environment step complete. New obs shape: {}, new rewards shape: {}, new done shape: {}", step, nextObs.shape(), rewards.get(step).shape(), nextDone.shape());

                /*
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(
                                f"global_step={global_step}, episodic_return={info['episode']['r']}"
                            )
                            writer.add_scalar(
                                "charts/episodic_return", info["episode"]["r"], global_step
                            )
                            writer.add_scalar(
                                "charts/episodic_length", info["episode"]["l"], global_step
                            )
                */

        //     TODO: Handle logging of episodic returns and lengths

        // Sleep for a short duration to allow the environment to process the step
        // try {
        //     Thread.sleep(5000);
        // } catch (InterruptedException e) {
        //     throw new RuntimeException(e);
        // }

        step++;
    }

    public static void finishEpoch() {
        LOGGER.memory();
        logText = "Finish Epoch...";



            /*
            next_value = agent.get_value(
                next_obs,
                next_lstm_state,
                next_done,
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values
             */

        AutogradState.get_tls_state().set_grad_mode(false);

        Tensor nextValue = model.getValue(nextObs, nextLstmState, nextDone);
        nextValue = nextValue.reshape(-1);
        LOGGER.memory();

        advantages.zero_();
        Tensor lastGAELam = null;
        LOGGER.memory();

        for (int t = args.numSteps - 1; t >= 0; t--) {
            Tensor nextNonTerminal;
            Tensor nextValues;
            if (t == args.numSteps - 1) {
                nextNonTerminal = onesLikeNumEnvs.sub(nextDone);
                nextValues = nextValue;
            } else {
                nextNonTerminal = onesLikeNumEnvs.sub(dones.get(t + 1));
                nextValues = values.get(t + 1);
            }
            // delta = (
            //     rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            // )
            Tensor delta = rewards.get(t)
                .add(
                    nextValues.mul(nextNonTerminal).mul(SCALAR_GAMMA)
                )
                .sub(values.get(t));

            /// Everthing above here can be vectorized if neeeded

            // advantages[t] = lastgaelam = (
            //     delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            // )
            Tensor advantage = delta.add(
                nextNonTerminal.mul(SCALAR_GAMMA_GAE_LAMBDA).mul(lastGAELam == null ? zerosLikeNumEnvs : lastGAELam)
            );
            advantages.get(t).copy_(advantage);
            lastGAELam = advantage;
        }
        LOGGER.memory();

        Tensor returns = advantages.add(values);

        AutogradState.get_tls_state().set_grad_mode(true);
        LOGGER.memory();

            /*
            # flatten the batch
            b_obs = obs.reshape((-1, MinecraftDummyEnv.OBSERVATION_SPACE_SIZE))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, MinecraftDummyEnv.ACTION_SPACE_SIZE))
            b_dones = dones.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
             */

        Tensor bObs = observations.reshape(-1, Observation.OBSERVATION_SPACE_SIZE);
        Tensor bLogProbs = logprobs.reshape(-1);
        Tensor bActions = actions.reshape(-1, Action.ACTION_SPACE_SIZE);
        Tensor bDones = dones.reshape(-1);
        Tensor bAdvantages = advantages.reshape(-1);
        Tensor bReturns = returns.reshape(-1);
        Tensor bValues = values.reshape(-1);
        LOGGER.memory();


            /*
            envsperbatch = args.num_envs // args.num_minibatches
            envinds = np.arange(args.num_envs)
            flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
             */

        assert args.numEnvs % args.numMinibatches == 0 : "Number of environments must be divisible by number of minibatches.";

        int envsPerBatch = args.numEnvs / args.numMinibatches;
        Tensor envinds = torch.arange(SCALAR_NUM_ENVS).cuda();
        Tensor flatinds = torch.arange(SCALAR_BATCH_SIZE).reshape(args.numSteps, args.numEnvs).cuda();
        LOGGER.memory();

            /*
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(envinds)
                for start in range(0, args.num_envs, envsperbatch):
                */

        // ArrayList<Float> clipFracs = new ArrayList<>();
        Tensor clipFracs = torch.zeros(1).cuda();
        int numClipFracs = 0;
        Tensor vLoss = null;
        Tensor pgLoss = null;
        Tensor entropyLoss = null;
        Tensor approxKl = null;
        Tensor oldApproxKl = null;
        LOGGER.memory();

        for (int epoch = 0; epoch < args.updateEpochs; epoch++) {

            // Shuffle environment indices
            // Random rnd = new Random();
            // for (int i = envinds.length - 1; i > 0; i--) {
            //     int j = rnd.nextInt(i + 1);
            //     int temp = envinds[i];
            //     envinds[i] = envinds[j];
            //     envinds[j] = temp;
            // }


            for (int start = 0; start < args.numEnvs; start += envsPerBatch) {
                    /*
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[
                        :, mbenvinds
                    ].ravel()  # be really careful about the index

                    */
                // int end = start + envsPerBatch;
                Tensor mbenvinds = envinds.narrow(0, start, envsPerBatch);
                // int[] mbenvinds = Arrays.copyOfRange(envinds, start, end);

                Tensor mb_inds = flatinds.index_select(1, mbenvinds).ravel();
                LOGGER.memory();


                // int[] mb_inds = new int[args.numSteps * mbenvinds.length];
                // for (int i = 0; i < args.numSteps; i++) {
                //     for (int j = 0; j < mbenvinds.length; j++) {
                //         mb_inds[i * mbenvinds.length + j] = flatinds[i][mbenvinds[j]];
                //     }
                // }

            /*

                    _, _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                        b_obs[mb_inds],
                        (
                            initial_lstm_state[0][:, mbenvinds],
                            initial_lstm_state[1][:, mbenvinds],
                        ),
                        b_dones[mb_inds],
                        b_actions.long()[mb_inds],
                    )
                    */

                Tensor bObs_mb_inds = bObs.index_select(0, mb_inds);
                // Tensor bObsMB = torch.zeros(mb_inds.length, Observation.OBSERVATION_SPACE_SIZE).cuda();
                // for (int i = 0; i < mb_inds.length; i++) {
                //     bObsMB.get(i).copy_(bObs.get(mb_inds[i]));
                // }

                Tensor lstmStateHidden = initialLSTMState.hiddenState().index_select(1, mbenvinds);
                Tensor lstmStateCell = initialLSTMState.cellState().index_select(1, mbenvinds);
                LOGGER.memory();

                Tensor bDones_mb_inds = bDones.index_select(0, mb_inds);
                // Tensor bDonesMB = torch.zeros(mb_inds.length).cuda();
                // for (int i = 0; i < mb_inds.length; i++) {
                //     bDonesMB.get(i).copy_(bDones.get(mb_inds[i]));
                // }

                Tensor bActions_mb_inds = bActions.index_select(0, mb_inds);
                // Tensor bActionsMB = torch.zeros(mb_inds.length, Action.ACTION_SPACE_SIZE).cuda();
                // for (int i = 0; i < mb_inds.length; i++) {
                //     bActionsMB.get(i).copy_(bActions.get(mb_inds[i]));
                // }

                // _, _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value...
                MinecraftRL.ActionAndValue actionAndValueResult = model.getActionAndValue(
                    bObs_mb_inds,
                    new MinecraftRL.LSTMState(lstmStateHidden, lstmStateCell),
                    bDones_mb_inds,
                    bActions_mb_inds
                );
                LOGGER.memory();


                    /*
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    */

                Tensor logRatio = actionAndValueResult.totalLogProbs().sub(
                    bLogProbs.index_select(0, mb_inds)
                );
                // Tensor bLogProbsMB = torch.zeros(mb_inds.length).cuda();
                // for (int i = 0; i < mb_inds.length; i++) {
                //     bLogProbsMB.get(i).copy_(bLogProbs.get(mb_inds[i]));
                // }
                // Tensor logratio = actionAndValueResult.totalLogProbs().sub(bLogProbsMB);
                Tensor ratio = logRatio.exp();
                LOGGER.memory();

                    /*

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    */

                AutogradState.get_tls_state().set_grad_mode(false);

                oldApproxKl = logRatio.neg().mean();
                approxKl = ratio.sub(SCALAR_ONE).sub_(logRatio).mean();
                // Scalar item = ratio.sub(SCALAR_ONE).abs().gt(SCALAR_CLIP_COEF).toType(torch.ScalarType.Float).mean().item();
                // clipFracs.add(item.toFloat());
                Tensor clipFracTensor = ratio.sub(SCALAR_ONE).abs_().gt_(SCALAR_CLIP_COEF).mean();
                clipFracs.add_(clipFracTensor);
                numClipFracs++;
                LOGGER.memory();

                AutogradState.get_tls_state().set_grad_mode(true);

                    /*

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    */

                Tensor mbAdvantages = bAdvantages.index_select(0, mb_inds);
                // Tensor mbAdvantages = torch.zeros(mb_inds.length).cuda();
                // for (int i = 0; i < mb_inds.length; i++) {
                //     mbAdvantages.get(i).copy_(bAdvantages.get(mb_inds[i]));
                // }

                if (args.normAdv) {
                    Tensor mean = mbAdvantages.mean();
                    Tensor std = mbAdvantages.std();
                    mbAdvantages = mbAdvantages.sub(mean).div(std.add(SCALAR_1E_8));
                }
                LOGGER.memory();

                    /*

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    */

                Tensor pgLoss1 = mbAdvantages.neg().mul(ratio);
                Tensor pgLoss2 = mbAdvantages.neg().mul(
                    torch.clamp(ratio, SCALAR_1_SUB_CLIP_COEF, SCALAR_1_ADD_CLIP_COEF)
                );
                pgLoss = torch.max(pgLoss1, pgLoss2).mean();
                LOGGER.memory();

                    /*

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                    */

                Tensor newValue = actionAndValueResult.value().view(-1);
                // Tensor bReturnsMB = torch.zeros(mb_inds.length).cuda();
                // for (int i = 0; i < mb_inds.length; i++) {
                //     bReturnsMB.get(i).copy_(bReturns.get(mb_inds[i]));
                // }
                Tensor mbReturns = bReturns.index_select(0, mb_inds);
                LOGGER.memory();
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

                    Tensor vLossUnclipped = newValue.sub(mbReturns).square();

                    Tensor bValuesMB = bValues.index_select(0, mb_inds);
                    // Tensor bValuesMB = torch.zeros(mb_inds.length).cuda();
                    // for (int i = 0; i < mb_inds.length; i++) {
                    //     bValuesMB.get(i).copy_(bValues.get(mb_inds[i]));
                    // }

                    Tensor vClipped = bValuesMB.add(
                        torch.clamp(newValue.sub(bValuesMB),
                            SCALAR_NEG_CLIP_COEF,
                            SCALAR_OPT_CLIP_COEF)
                    );

                    Tensor vLossClipped = vClipped.sub(mbReturns).square();
                    Tensor vLossMax = torch.max(vLossUnclipped, vLossClipped);
                    vLoss = vLossMax.mean().mul(SCALAR_0_5);
                } else {
                        /*

                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                         */
                    vLoss = newValue.sub(mbReturns).square().mean().mul(SCALAR_0_5);
                }
                LOGGER.memory();

                    /*

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                        */

                entropyLoss = actionAndValueResult.totalEntropy().mean();
                Tensor loss = pgLoss.sub(
                    entropyLoss.mul(SCALAR_ENT_COEF)
                ).add(
                    vLoss.mul(SCALAR_VF_COEF)
                );
                LOGGER.memory();

                    /*

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    */

                optimizer.zero_grad();
                LOGGER.memory();
                loss.backward();
                LOGGER.memory();
                // Clip gradients
                torch.clip_grad_norm_(modelParameters, args.maxGradNorm);
                LOGGER.memory();
                optimizer.step();
                LOGGER.memory();
            }
                    /*

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
             */
            if (args.targetKl != null && approxKl != null && approxKl.item().toFloat() > args.targetKl) {
                LOGGER.warn("Target KL ({}) exceeded ({}). Breaking from update epochs.", args.targetKl, approxKl.item().toFloat());
                break;
            }
        }

            /*
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
             */
        Tensor yPred = bValues;
        Tensor yTrue = bReturns;
        double varY = yTrue.var().item().toDouble();
        double explainedVar = varY == 0 ? Double.NaN : 1.0 - yTrue.sub(yPred).var().div(new Scalar(varY)).item().toDouble();
        LOGGER.memory();

            /*
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )
             */

        // LOGGER.info("--- Epoch {} Summary ---", iteration);
        // LOGGER.info("Learning Rate: {}", optimizer.param_groups().get(0).options().get_lr());
        // LOGGER.info("Value Loss: {}", vLoss.item().toFloat());
        // LOGGER.info("Policy Loss: {}", pgLoss.item().toFloat());
        // LOGGER.info("Entropy: {}", entropyLoss.item().toFloat());
        // if (oldApproxKl != null) LOGGER.info("Old Approx KL: {}", oldApproxKl.item().toFloat());
        // if (approxKl != null) LOGGER.info("Approx KL: {}", approxKl.item().toFloat());
        // LOGGER.info("Clip Fraction: {}", clipFracs.stream().mapToDouble(Float::doubleValue).average().orElse(0.0));
        // LOGGER.info("Explained Variance: {}", explainedVar);
        // LOGGER.info("Steps Per Second (SPS): {}", (int) (globalStep / ((System.currentTimeMillis() - startTime) / 1000.0)));
        // LOGGER.info("Steps Per Second Per Env (SPSPE): {}", (int) ((double) globalStep / args.numEnvs / ((System.currentTimeMillis() - startTime) / 1000.0)));
        // LOGGER.info("Global Step Count: {}", globalStep);
        LOGGER.info("==================== Finished Epoch for Iteration:      {} ====================", iteration - 1);

        // After metrics calculation, log to CSV
        try {
            // TODO: Log the GPU time vs the CPU time & ratio
            double learningRate = optimizer.param_groups().get(0).options().get_lr();
            double valueLoss = vLoss.item().toDouble();
            double policyLoss = pgLoss.item().toDouble();
            double entropy = entropyLoss.item().toDouble();
            Double oldApproxKlVal = oldApproxKl != null ? oldApproxKl.item().toDouble() : null;
            Double approxKlVal = approxKl != null ? approxKl.item().toDouble() : null;
            double clipfrac = clipFracs.div(new Scalar(numClipFracs)).item().toFloat();
            double iterationTime = ((System.currentTimeMillis() - iterationStartTime) / 1000.0);
            double sps = ((args.numEnvs * args.numSteps) / iterationTime);
            double averageRewards = rewards.mean().item().toDouble();
            trainingLogger.logStep(iteration, learningRate, valueLoss, policyLoss, entropy, oldApproxKlVal, approxKlVal, clipfrac, explainedVar, iterationTime, sps, numTerminations, averageRewards);

            LOGGER.info("Approx KL this epoch: {}", approxKlVal);
        } catch (Exception e) {
            LOGGER.error("Failed to log training metrics: {}", e.getMessage());
        }
        LOGGER.memory();

        iteration++;

        if (iteration % 200 == 0) {
            // allocator.snapshot();
            model.saveCheckpoint(iteration);
        }
        // 17:49:52 20
        // 17:59:26 185

        // LOGGER.info("=========================================================================");
    }
}
