package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.debugger.Debugger;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.BooleanControl;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.ButtonControl;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.CounterControl;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.VariableControl;
import com.tenbitmelon.machinelearningplayer.environment.Action;
import com.tenbitmelon.machinelearningplayer.environment.MinecraftEnvironment;
import com.tenbitmelon.machinelearningplayer.environment.Observation;
import net.kyori.adventure.text.Component;
import org.bukkit.Bukkit;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import java.util.*;

import static com.tenbitmelon.machinelearningplayer.MachineLearningPlayer.LOGGER;

public class TrainingManager {

    static public boolean runTraining = false;

    static ExperimentConfig args;
    static SyncedVectorEnvironment environment;
    static MinecraftRL model;
    static MinecraftRL.LSTMState initialLSTMState;
    static MinecraftRL.LSTMState nextLstmState;
    static VectorResetResult resetResult;
    static Tensor nextObs;
    static AdamOptions adamOptions;
    static Adam optimizer;
    static Tensor observations;
    static Tensor actions;
    static Tensor logprobs;
    static Tensor rewards;
    static Tensor dones;
    static Tensor values;
    static Tensor nextDone;

    static int globalStep = 0;
    static long startTime = System.currentTimeMillis();
    static int iteration = 1;
    static int step = 0;

    static String logText = "";

    private static boolean runningInnerLoop = false;

    public static void setup() {
        LOGGER.info("--- Setting up Training Manager ---");
        args = ExperimentConfig.getInstance();
        LOGGER.info("ExperimentConfig loaded. numEnvs={}, numSteps={}", args.numEnvs, args.numSteps);
        environment = new SyncedVectorEnvironment(args);
        LOGGER.info("SyncedVectorEnvironment initialized.");
        model = new MinecraftRL(environment);
        LOGGER.info("MinecraftRL model initialized.");

        nextLstmState = new MinecraftRL.LSTMState(
            torch.zeros(model.getLSTMLayers(), args.numEnvs, model.lstmHiddenSize),
            torch.zeros(model.getLSTMLayers(), args.numEnvs, model.lstmHiddenSize));
        LOGGER.info("Initial LSTM state created. Hidden shape: {}, Cell shape: {}", nextLstmState.hiddenState().shape(), nextLstmState.cellState().shape());

        adamOptions = new AdamOptions(args.learningRate);
        optimizer = new Adam(model.parameters(), adamOptions);
        adamOptions.eps().put(1e-5);
        LOGGER.info("Adam optimizer initialized with learning rate: {}", args.learningRate);

        nextDone = torch.zeros(args.numEnvs);
        LOGGER.info("Initial 'nextDone' tensor created. Shape: {}", nextDone.shape());

        observations = torch.zeros(args.numSteps, args.numEnvs, Observation.OBSERVATION_SPACE_SIZE);
        actions = torch.zeros(args.numSteps, args.numEnvs, Action.ACTION_SPACE_SIZE);
        logprobs = torch.zeros(args.numSteps, args.numEnvs);
        rewards = torch.zeros(args.numSteps, args.numEnvs);
        dones = torch.zeros(args.numSteps, args.numEnvs);
        values = torch.zeros(args.numSteps, args.numEnvs);
        LOGGER.info("Storage tensors initialized. Shape (e.g., observations): {}", observations.shape());

        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text(""), () -> logText));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Env. Ready"), () -> environment.isReady()));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Inner Loop"), () -> runningInnerLoop));
        Debugger.mainDebugWindow.addControl(new BooleanControl(Component.text("Run Training"), () -> runTraining, (value) -> runTraining = value));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Iteration"), () -> iteration));
        Debugger.mainDebugWindow.addControl(new VariableControl(Component.text("Step"), () -> step));
        Debugger.mainDebugWindow.addText("");
        Debugger.mainDebugWindow.addText("Arguments:");
        // args
        Debugger.mainDebugWindow.addControl(new CounterControl(Component.text("Num Iterations"), () -> args.numIterations, (value) -> args.numIterations = value));
        Debugger.mainDebugWindow.addControl(new CounterControl(Component.text("Num Steps"), () -> args.numSteps, (value) -> args.numSteps = value));

        LOGGER.info("--- Training Manager setup complete ---");
    }

    // Placeholder for future implementation
    public static void trainingStep() {
        if (!runTraining) {
            return;
        }
        LOGGER.info("--- Tick ---");

        if (!environment.isReady()) {
            Bukkit.broadcast(Component.text("Environment is not ready for training."));
            LOGGER.warn("Attempted to run training step, but environment is not ready.");
            runTraining = false;
            return;
        }

        if (resetResult == null) {
            LOGGER.info("Initial environment reset...");
            resetResult = environment.reset(args.seed);
            nextObs = resetResult.observationsTensor();
            LOGGER.info("Environment reset complete. Initial observation shape: {}", nextObs.shape());
        }


        Debugger.mainDebugWindow.refresh();


        if (runningInnerLoop) {
            runSteps();
        } else {
            epochSetup();
            if (!runTraining) return;
            runSteps();
        }
    }

    public static void epochSetup() {
        LOGGER.info("==================== Starting Epoch Setup for Iteration: {} ====================", iteration);
        logText = "Epoch Setup...";
        Debugger.mainDebugWindow.refresh();
        // for iteration in range(1, args.num_iterations + 1):

        // if (iteration >= 2) {
        if (iteration >= args.numIterations + 1) {
            LOGGER.info("Maximum iterations reached. Stopping training.");
            runTraining = false;
            return;
        }


            /*
            initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
             */
        initialLSTMState = nextLstmState.clone();
        LOGGER.info("Cloned LSTM state for the new epoch.");

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
            LOGGER.info("Annealed learning rate for iteration {}: {}", iteration, lrNow);
        }

    }

    public static void runSteps() {
        LOGGER.info("[===] Running steps for iteration {}, step {}", iteration, step);
        logText = "Run Steps...";
        Debugger.mainDebugWindow.refresh();

        // for step in range(0, args.num_steps):
        runningInnerLoop = true;

        if (step >= args.numSteps) {
            LOGGER.info("Completed all {} steps for iteration {}. Finishing epoch.", args.numSteps, iteration);
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

        globalStep += args.numEnvs;
        observations.get(step).put(nextObs);
        dones.get(step).put(nextDone);
        LOGGER.info("[Step {}] Stored next_obs (shape: {}) and next_done (shape: {})", step, nextObs.shape(), nextDone.shape());

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
        Debugger.mainDebugWindow.refresh();
        LOGGER.info("[Step {}] Calling model.getActionAndValue...", step);
        MinecraftRL.ActionAndValue actionResult = model.getActionAndValue(nextObs, nextLstmState, nextDone);
        values.get(step).put(actionResult.value().flatten());
        actions.get(step).put(actionResult.action());
        logprobs.get(step).put(actionResult.totalLogProbs());
        LOGGER.info("[Step {}] Stored values (shape: {}), actions (shape: {}), logprobs (shape: {})",
            step, values.get(step).shape(), actions.get(step).shape(), logprobs.get(step).shape());

                /*
                next_obs, reward, terminations, truncations, infos = envs.step(action_dict)
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                    next_done
                ).to(device)
                 */

        logText = "Stepping environment for step " + step;
        Debugger.mainDebugWindow.refresh();
        LOGGER.info("[Step {}] Stepping environment with action (shape: {})", step, actionResult.action().shape());
        VectorStepResult stepResult = environment.step(actionResult.action());
        nextDone = Tensor.create(stepResult.logicalOrTerminationsAndTruncations());
        rewards.get(step).put(Tensor.create(stepResult.rewards()));
        nextObs = stepResult.observationsTensor();
        LOGGER.info("[Step {}] Environment step complete. New obs shape: {}, new rewards shape: {}, new done shape: {}",
            step, nextObs.shape(), rewards.get(step).shape(), nextDone.shape());

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
        LOGGER.info("==================== Finishing Epoch for Iteration: {} ====================", iteration);
        logText = "Finish Epoch...";
        Debugger.mainDebugWindow.refresh();

        iteration++;


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
        Tensor returns;
        LOGGER.info("Calculating value for the last observation to bootstrap rewards...");
        Tensor value = model.getValue(nextObs, nextLstmState, nextDone);
        value = value.reshape(1, -1);
        LOGGER.info("Bootstrap value shape: {}", value.shape());

        LOGGER.info("Calculating GAE advantages and returns...");
        Tensor advantages = torch.zeros_like(rewards);
        Tensor lastGAELam = torch.zeros_like(rewards.get(0));
        for (int t = args.numSteps - 1; t >= 0; t--) {
            Tensor nextNonTerminal;
            Tensor nextValues;
            if (t == args.numSteps - 1) {
                nextNonTerminal = torch.ones_like(nextDone).sub(nextDone);
                nextValues = value;
            } else {
                nextNonTerminal = torch.ones_like(dones.get(t + 1)).sub(dones.get(t + 1));
                nextValues = values.get(t + 1);
            }
            Tensor delta = rewards.get(t).add(nextValues.mul(nextNonTerminal)).sub(values.get(t));
            // advantages[t] = lastgaelam = (
            //     delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            // )

            double multpart = args.gamma * args.gaeLambda;
            Tensor advantage = delta.add(
                nextNonTerminal.mul(new Scalar(multpart)).mul(lastGAELam)
            );
            advantages.get(t).put(advantage);
            lastGAELam = advantage;
        }
        returns = advantages.add(values);
        LOGGER.info("GAE calculation complete. Advantages shape: {}, Returns shape: {}", advantages.shape(), returns.shape());

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

        LOGGER.info("Flattening batch data for updates...");
        Tensor bObs = observations.reshape(-1, Observation.OBSERVATION_SPACE_SIZE);
        Tensor bLogProbs = logprobs.reshape(-1);
        Tensor bActions = actions.reshape(-1, Action.ACTION_SPACE_SIZE);
        Tensor bDones = dones.reshape(-1);
        Tensor bAdvantages = advantages.reshape(-1);
        Tensor bReturns = returns.reshape(-1);
        Tensor bValues = values.reshape(-1);
        LOGGER.info("Flattened batch shapes: b_obs: {}, b_actions: {}, b_logprobs: {}", bObs.shape(), bActions.shape(), bLogProbs.shape());


            /*
            envsperbatch = args.num_envs // args.num_minibatches
            envinds = np.arange(args.num_envs)
            flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
             */

        int envsPerBatch = args.numEnvs / args.numMinibatches;
        int[] envinds = new int[args.numEnvs];
        for (int i = 0; i < args.numEnvs; i++) {
            envinds[i] = i;
        }
        int[][] flatinds = new int[args.numSteps][args.numEnvs];
        int idx = 0;
        for (int i = 0; i < args.numSteps; i++) {
            for (int j = 0; j < args.numEnvs; j++) {
                flatinds[i][j] = idx++;
            }
        }
        LOGGER.info("Minibatch setup: {} minibatches, {} envs per batch.", args.numMinibatches, envsPerBatch);


        LOGGER.info("--- Starting PPO Update Loop ({} epochs) ---", args.updateEpochs);
            /*
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(envinds)
                for start in range(0, args.num_envs, envsperbatch):
                */

        ArrayList<Float> clipFracs = new ArrayList<>();
        Tensor vLoss = null;
        Tensor pgLoss = null;
        Tensor entropyLoss = null;
        Tensor approxKl = null;
        Tensor oldApproxKl = null;
        for (int epoch = 0; epoch < args.updateEpochs; epoch++) {
            LOGGER.info("Update epoch {}/{}", epoch + 1, args.updateEpochs);
            // Shuffle environment indices
            Random rnd = new Random();
            for (int i = envinds.length - 1; i > 0; i--) {
                int j = rnd.nextInt(i + 1);
                int temp = envinds[i];
                envinds[i] = envinds[j];
                envinds[j] = temp;
            }

            for (int start = 0; start < args.numEnvs; start += envsPerBatch) {
                    /*
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[
                        :, mbenvinds
                    ].ravel()  # be really careful about the index

                    */
                int end = start + envsPerBatch;
                LOGGER.info("  Minibatch: envs from {} to {}", start, end - 1);
                int[] mbenvinds = Arrays.copyOfRange(envinds, start, end);
                int[] mb_inds = new int[args.numSteps * mbenvinds.length];
                for (int i = 0; i < args.numSteps; i++) {
                    for (int j = 0; j < mbenvinds.length; j++) {
                        mb_inds[i * mbenvinds.length + j] = flatinds[i][mbenvinds[j]];
                    }
                }

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

                Tensor bObsMB = torch.zeros(mb_inds.length, Observation.OBSERVATION_SPACE_SIZE);
                for (int i = 0; i < mb_inds.length; i++) {
                    bObsMB.get(i).put(bObs.get(mb_inds[i]));
                }

                Tensor mbenvindsIndicesTensor = torch.tensor(mbenvinds);

                Tensor lstmStateHidden = initialLSTMState.hiddenState().index_select(1, mbenvindsIndicesTensor);
                Tensor lstmStateCell = initialLSTMState.cellState().index_select(1, mbenvindsIndicesTensor);


                Tensor bDonesMB = torch.zeros(mb_inds.length);
                for (int i = 0; i < mb_inds.length; i++) {
                    bDonesMB.get(i).put(bDones.get(mb_inds[i]));
                }

                Tensor bActionsMB = torch.zeros(mb_inds.length, Action.ACTION_SPACE_SIZE);
                for (int i = 0; i < mb_inds.length; i++) {
                    bActionsMB.get(i).put(bActions.get(mb_inds[i]));
                }

                // _, _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value
                LOGGER.info("  Minibatch shapes: obs={}, actions={}, dones={}, lstm_hidden={}",
                    bObsMB.shape(), bActionsMB.shape(), bDonesMB.shape(), lstmStateHidden.shape());
                MinecraftRL.ActionAndValue actionAndValueResult = model.getActionAndValue(
                    bObsMB,
                    new MinecraftRL.LSTMState(lstmStateHidden, lstmStateCell),
                    bDonesMB,
                    bActionsMB
                );
                LOGGER.info("  Forward pass for minibatch complete.");


                    /*
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    */

                Tensor bLogProbsMB = torch.zeros(mb_inds.length);
                for (int i = 0; i < mb_inds.length; i++) {
                    bLogProbsMB.get(i).put(bLogProbs.get(mb_inds[i]));
                }
                Tensor logratio = actionAndValueResult.totalLogProbs().sub(bLogProbsMB);
                Tensor ratio = logratio.exp();
                LOGGER.info("  Calculated logratio shape: {}, ratio shape: {}", logratio.shape(), ratio.shape());

                    /*

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    */

                oldApproxKl = logratio.neg().mean();
                approxKl = ratio.sub(new Scalar(1.0)).sub(logratio).mean();
                Scalar item = ratio.sub(new Scalar(1.0)).abs().gt(new Scalar(args.clipCoef)).toType(torch.ScalarType.Float).mean().item();
                clipFracs.add(item.toFloat());

                    /*

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    */

                Tensor mbAdvantages = torch.zeros(mb_inds.length);
                for (int i = 0; i < mb_inds.length; i++) {
                    mbAdvantages.get(i).put(bAdvantages.get(mb_inds[i]));
                }

                if (args.normAdv) {
                    Tensor mean = mbAdvantages.mean();
                    Tensor std = mbAdvantages.std();
                    mbAdvantages = mbAdvantages.sub(mean).div(std.add(new Scalar(1e-8)));
                    LOGGER.info("  Normalized minibatch advantages.");
                }

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
                    torch.clamp(ratio, new ScalarOptional(new Scalar(1.0 - args.clipCoef)), new ScalarOptional(new Scalar(1.0 + args.clipCoef)))
                );
                pgLoss = torch.max(pgLoss1, pgLoss2).mean();

                    /*

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                    */

                Tensor newValue = actionAndValueResult.value().view(-1);
                Tensor bReturnsMB = torch.zeros(mb_inds.length);
                for (int i = 0; i < mb_inds.length; i++) {
                    bReturnsMB.get(i).put(bReturns.get(mb_inds[i]));
                }
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

                    Tensor vLossUnclipped = newValue.sub(bReturnsMB).square();

                    Tensor bValuesMB = torch.zeros(mb_inds.length);
                    for (int i = 0; i < mb_inds.length; i++) {
                        bValuesMB.get(i).put(bValues.get(mb_inds[i]));
                    }
                    Tensor vClipped = bValuesMB.add(
                        torch.clamp(newValue.sub(bValuesMB),
                            new ScalarOptional(new Scalar(-args.clipCoef)),
                            new ScalarOptional(new Scalar(args.clipCoef)))
                    );

                    Tensor vLossClipped = vClipped.sub(bReturnsMB).square();
                    Tensor vLossMax = torch.max(vLossUnclipped, vLossClipped);
                    vLoss = vLossMax.mean().mul(new Scalar(0.5));
                        /*

                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                         */
                } else {
                    vLoss = newValue.sub(bReturnsMB).square().mean().mul(new Scalar(0.5));
                }

                    /*

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                        */

                entropyLoss = actionAndValueResult.totalEntropy().mean();
                Tensor loss = pgLoss.sub(
                    entropyLoss.mul(new Scalar(args.entCoef))
                ).add(
                    vLoss.mul(new Scalar(args.vfCoef))
                );
                LOGGER.info("  Calculated losses: Total={}, PG={}, VLoss={}, Entropy={}",
                    loss.item().toFloat(), pgLoss.item().toFloat(), vLoss.item().toFloat(), entropyLoss.item().toFloat());

                    /*

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    */

                optimizer.zero_grad();
                loss.backward();
                // Clip gradients
                torch.clip_grad_norm_(model.parameters(), args.maxGradNorm);
                optimizer.step();
                LOGGER.info("  Optimizer step completed (zero_grad, backward, clip_grad_norm, step).");
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
        LOGGER.info("--- PPO Update Loop Finished ---");

            /*
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
             */
        Tensor yPred = bValues;
        Tensor yTrue = bReturns;
        double varY = yTrue.var().item().toDouble();
        double explainedVar = varY == 0 ? Double.NaN : 1.0 - yTrue.sub(yPred).var().div(new Scalar(varY)).item().toDouble();

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

        LOGGER.info("--- Epoch {} Summary ---", iteration);
        LOGGER.info("Learning Rate: {}", optimizer.param_groups().get(0).options().get_lr());
        if (vLoss != null) LOGGER.info("Value Loss: {}", vLoss.item().toFloat());
        if (pgLoss != null) LOGGER.info("Policy Loss: {}", pgLoss.item().toFloat());
        if (entropyLoss != null) LOGGER.info("Entropy: {}", entropyLoss.item().toFloat());
        if (oldApproxKl != null) LOGGER.info("Old Approx KL: {}", oldApproxKl.item().toFloat());
        if (approxKl != null) LOGGER.info("Approx KL: {}", approxKl.item().toFloat());
        LOGGER.info("Clip Fraction: {}", clipFracs.stream().mapToDouble(Float::doubleValue).average().orElse(0.0));
        LOGGER.info("Explained Variance: {}", explainedVar);
        LOGGER.info("Steps Per Second (SPS): {}", (int) (globalStep / ((System.currentTimeMillis() - startTime) / 1000.0)));
        LOGGER.info("Global Step Count: {}", globalStep);
        LOGGER.info("=========================================================================");
    }
}
