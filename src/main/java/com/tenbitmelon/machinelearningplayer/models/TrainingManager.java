package com.tenbitmelon.machinelearningplayer.models;

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
import org.bytedeco.javacpp.PointerScope;
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
    static AdamOptions adamOptions;
    static Adam optimizer;
    static long iterationStartTime = System.currentTimeMillis();
    static int iteration = 1;
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
    private static int numTruncations = 0;

    public static void setup() {

        // TODO: torch.cuda.amp.autocast for mixed precision training
        // TODO: gradient scaling

        // TODO: torch.compile()
        // TODO: Use pinned memory for a lot of this

        device = new Device("cuda:0");
        trainingLogger = new TrainingLogger(args, "logs/training");

        environment = new SyncedVectorEnvironment(args);

        model = new MinecraftRL(environment);
        model.loadCheckpoint(args.startingCheckpoint);
        model.to(device, false);
        modelParameters = model.parameters();

        adamOptions = new AdamOptions(args.learningRate);
        optimizer = new Adam(modelParameters, adamOptions);
        adamOptions.eps().put(1e-5);

        TensorOptions deviceTensorOptions = new TensorOptions(device);
        observations = torch.zeros(new long[]{args.numSteps, args.numEnvs, Observation.OBSERVATION_SPACE_SIZE}, deviceTensorOptions);
        actions = torch.zeros(new long[]{args.numSteps, args.numEnvs, Action.ACTION_SPACE_SIZE}, deviceTensorOptions);
        logprobs = torch.zeros(new long[]{args.numSteps, args.numEnvs}, deviceTensorOptions);
        rewards = torch.zeros(new long[]{args.numSteps, args.numEnvs}, deviceTensorOptions);
        dones = torch.zeros(new long[]{args.numSteps, args.numEnvs}, deviceTensorOptions);
        values = torch.zeros(new long[]{args.numSteps, args.numEnvs}, deviceTensorOptions);

        nextDone = torch.zeros(new long[]{args.numEnvs}, deviceTensorOptions);
        nextLstmState = new MinecraftRL.LSTMState(
            torch.zeros(new long[]{model.getLSTMLayers(), args.numEnvs, model.getLSTMHiddenSize()}, deviceTensorOptions),
            torch.zeros(new long[]{model.getLSTMLayers(), args.numEnvs, model.getLSTMHiddenSize()}, deviceTensorOptions));


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
    }

    // Placeholder for future implementation
    public static void trainingStep() {
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
            nextObs = resetResult.observationsTensor().to(device, torch.ScalarType.Float);
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


        /*
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
         */
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
        observations.get(step).copy_(nextObs.detach());
        dones.get(step).copy_(nextDone);

        /*
        with torch.no_grad():
            action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                next_obs, next_lstm_state, next_done
            )
            values[step] = value.flatten()
         */
        logText = "Getting action and value for step " + step;

        AutogradState.get_tls_state().set_grad_mode(false); // with torch.no_grad():

        MinecraftRL.ActionAndValue actionResult = model.getActionAndValue(nextObs, nextLstmState, nextDone);
        nextLstmState.copy_(actionResult.lstmState());

        Tensor actionResultValue = actionResult.value();
        values.get(step).copy_(actionResultValue.flatten());
        actionResultValue.close();

        AutogradState.get_tls_state().set_grad_mode(true);

        /*
        actions[step] = action
        logprobs[step] = logprob
         */

        Tensor actionResultAction = actionResult.action();
        actions.get(step).copy_(actionResultAction.detach());
        logprobs.get(step).copy_(actionResult.totalLogProbs());

        logText = "Stepping environment for step " + step;
        /*
        next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
         */

        Tensor actionTensor = actionResultAction.cpu();
        actionResultAction.close();
        environment.preTickStep(actionTensor);

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
        VectorStepResult stepResult = environment.postTickStep();

        nextObs.close();
        nextObs = stepResult.observationsTensor().to(device, torch.ScalarType.Float);
        nextObs.retainReference();

        nextDone.close();
        nextDone = Tensor.create(stepResult.logicalOrTerminationsAndTruncations()).to(device, torch.ScalarType.Float);
        nextDone.retainReference();

        Tensor newRewardsTensor = Tensor.create(stepResult.rewards()).to(device, torch.ScalarType.Float).view(-1);
        rewards.get(step).copy_(newRewardsTensor);
        newRewardsTensor.close();

        // Moved theses parts onto the lines above
        // nextObs = torch.tensor(nextObs).to(device, torch.ScalarType.Float);
        // nextDone = torch.tensor(nextDone).to(device, torch.ScalarType.Float);

        numTerminations += stepResult.numTerminations();
        numTruncations += stepResult.numTruncations();

        /*
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        */

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

        Tensor nextValue = model.getValue(nextObs, nextLstmState, nextDone);
        nextValue = nextValue.reshape(-1);
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

        for (int t = args.numSteps - 1; t >= 0; t--) {
            PointerScope loopScope = new PointerScope();

            Tensor nextNonTerminal;
            Tensor nextValues;
            if (t == args.numSteps - 1) {
                nextNonTerminal = onesLikeNumEnvs.sub(nextDone);
                nextValues = nextValue;
            } else {
                nextNonTerminal = onesLikeNumEnvs.sub(dones.get(t + 1));
                nextValues = values.get(t + 1);
            }

            Tensor mul1 = nextValues.mul(nextNonTerminal).mul(SCALAR_GAMMA);
            Tensor delta = rewards.get(t)
                .add(mul1)
                .sub(values.get(t));


            Tensor mul2 = nextNonTerminal.mul(SCALAR_GAMMA_GAE_LAMBDA).mul(lastGAELam);
            Tensor advantage = delta.add(mul2);

            advantages.get(t).copy_(advantage);
            lastGAELam.copy_(advantage);

            loopScope.close();
        }

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
        Tensor bObs = observations.reshape(-1, Observation.OBSERVATION_SPACE_SIZE);
        /// [numSteps*numEnvs]
        Tensor bLogProbs = logprobs.reshape(-1);
        /// [numSteps*numEnvs, action_space]
        Tensor bActions = actions.reshape(-1, Action.ACTION_SPACE_SIZE);
        /// [numSteps*numEnvs]
        Tensor bDones = dones.reshape(-1);
        /// [numSteps*numEnvs]
        Tensor bAdvantages = advantages.reshape(-1);
        /// [numSteps*numEnvs]
        Tensor bReturns = returns.reshape(-1);
        /// [numSteps*numEnvs]
        Tensor bValues = values.reshape(-1);


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
        Tensor flatinds = torch.arange(SCALAR_BATCH_SIZE, new TensorOptions(device)).reshape(args.numSteps, args.numEnvs); // Shape: [numSteps, numEnvs]

        /*
        clipfracs = []
        for epoch in range(args.update_epochs):
            */

        Tensor clipFracs = torch.zeros(new long[]{1}, new TensorOptions(device)); // To accumulate clip fractions
        int numClipFracs = 0;

        Tensor vLoss = null;
        Tensor pgLoss = null;
        Tensor entropyLoss = null;
        Tensor approxKl = null;
        Tensor oldApproxKl = null;


        for (int epoch = 0; epoch < args.updateEpochs; epoch++) {

            /*
            np.random.shuffle(envinds)
             */
            Tensor randperm = torch.randperm(args.numEnvs, new TensorOptions(device));
            envinds = envinds.index_select(0, randperm);

            /*
            for start in range(0, args.num_envs, envsperbatch):
             */

            for (int start = 0; start < args.numEnvs; start += envsPerBatch) {
                /*
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
                */

                Tensor mbenvinds = envinds.narrow(0, start, envsPerBatch);
                Tensor mb_inds = flatinds.index_select(1, mbenvinds).ravel();

                /*
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions[mb_inds],
                )
                 */

                Tensor lstmStateHidden = initialLSTMState.hiddenState().index_select(1, mbenvinds);
                Tensor lstmStateCell = initialLSTMState.cellState().index_select(1, mbenvinds);

                MinecraftRL.ActionAndValue actionAndValueResult = model.getActionAndValue(
                    bObs.index_select(0, mb_inds),
                    new MinecraftRL.LSTMState(lstmStateHidden, lstmStateCell),
                    bDones.index_select(0, mb_inds),
                    bActions.index_select(0, mb_inds)
                );

                /*
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                */

                Tensor logRatio = actionAndValueResult.totalLogProbs().sub(bLogProbs.index_select(0, mb_inds));
                Tensor ratio = logRatio.exp();

                /*
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                 */

                AutogradState.get_tls_state().set_grad_mode(false); // with torch.no_grad():

                oldApproxKl = logRatio.neg().mean();
                approxKl = ratio.sub(SCALAR_ONE).sub_(logRatio).mean();
                Tensor clipFracTensor = ratio.sub(SCALAR_ONE).abs_().gt_(SCALAR_CLIP_COEF).to(torch.ScalarType.Float).mean();
                clipFracs.add_(clipFracTensor);
                numClipFracs++;

                AutogradState.get_tls_state().set_grad_mode(true);

                /*
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                 */

                Tensor mbAdvantages = bAdvantages.index_select(0, mb_inds);

                if (args.normAdv) {
                    Tensor mean = mbAdvantages.mean();
                    Tensor std = mbAdvantages.std();
                    mbAdvantages = mbAdvantages.sub(mean).div(std.add(SCALAR_1E_8));
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
                pgLoss = torch.max(pgLoss1, pgLoss2).mean();

                /*
                newvalue = newvalue.view(-1)
                */

                Tensor newvalue = actionAndValueResult.value().view(-1);

                /*
                if args.clip_vloss:
                 */
                Tensor bReturnsMbInds = bReturns.index_select(0, mb_inds);
                Tensor bValueMbInds = bValues.index_select(0, mb_inds);
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
                    vLoss = vLossMax.mean().mul(SCALAR_0_5);
                } else {
                    /*
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                     */
                    vLoss = newvalue.sub(bReturnsMbInds).square().mean().mul(SCALAR_0_5);
                }

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
        Tensor yPred = bValues.cpu();
        Tensor yTrue = bReturns.cpu();
        double varY = yTrue.var().item().toDouble();
        double explainedVar = varY == 0 ? Double.NaN : 1.0 - yTrue.sub(yPred).var().div(new Scalar(varY)).item().toDouble();

        LOGGER.info("==================== Finished Epoch for Iteration:      {} ====================", iteration - 1);


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

        try {
            OptimizerOptions options = optimizer.param_groups().get(0).options();
            double learningRate = options.get_lr();
            options.close();

            double valueLoss = vLoss.item().toDouble();
            double policyLoss = pgLoss.item().toDouble();
            double entropyLossDouble = entropyLoss.item().toDouble();

            Double oldApproxKlVal = oldApproxKl.item().toDouble();
            Double approxKlVal = approxKl.item().toDouble();
            double clipfrac = clipFracs.div(new Scalar(numClipFracs)).item().toFloat();
            double iterationTime = ((System.currentTimeMillis() - iterationStartTime) / 1000.0);
            double sps = ((args.numEnvs * args.numSteps) / iterationTime);
            double averageRewards = rewards.mean().item().toDouble();
            double totalRewards = rewards.sum().item().toDouble();

            trainingLogger.logStep(
                iteration,
                learningRate,
                valueLoss,
                policyLoss,
                entropyLossDouble,
                oldApproxKlVal,
                approxKlVal,
                clipfrac,
                explainedVar,
                iterationTime,
                sps,
                numTerminations,
                numTruncations,
                averageRewards,
                totalRewards,
                hw.gpuMemUsed(),
                hw.javaNativeUsed()
            );
            LOGGER.info(
                "Iteration {}, LR: {}, VLoss: {}, PLoss: {}, Entropy: {}, OldKL: {}, KL: {}, ClipFrac: {}, ExplVar: {}, IterTime: {}s, SPS: {}, AvgRewards: {}, TotRewards: {}",
                iteration,
                learningRate,
                valueLoss,
                policyLoss,
                entropyLossDouble,
                oldApproxKlVal,
                approxKlVal,
                clipfrac,
                explainedVar,
                iterationTime,
                sps,
                averageRewards,
                totalRewards
            );
        } catch (Exception e) {
            LOGGER.error("Failed to log training metrics: {}", e.getMessage());
        }
        LOGGER.memory();
        clipFracs.close();
        scope.close();

        iteration++;

        if (iteration % 100 == 0) {
            // allocator.snapshot();
            model.saveCheckpoint(iteration);
        }
    }

    public static int createCheckpoint() {
        model.saveCheckpoint(iteration);
        return iteration;
    }
}
