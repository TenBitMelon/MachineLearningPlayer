package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.environment.Observation;
import com.tenbitmelon.machinelearningplayer.util.distributions.Categorical;
import com.tenbitmelon.machinelearningplayer.util.distributions.Normal;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;
import org.bytedeco.pytorch.global.torch;

import javax.annotation.Nullable;

public class MinecraftRL extends Module {

    final SequentialImpl network;
    final LSTMImpl lstm;
    final LinearImpl actorForwardMoveKeys;
    final LinearImpl actorStrafingMoveKeys;
    final LinearImpl yawMean;
    final Tensor yawLogSTD;
    final LinearImpl pitchMean;
    final Tensor pitchLogSTD;
    final LinearImpl actorJumpKey;
    final LinearImpl actorSprintSneakKeys;
    final LinearImpl critic;
    final LinearImpl actorAttackUseItem;

    public MinecraftRL(SyncedVectorEnvironment environment) {
        /*
        other_features_dim = 128
        self.network = nn.Sequential(
           layer_init(nn.Linear(obs_shape, 64)),
           nn.Tanh(),
           layer_init(nn.Linear(64, 64)),
           nn.Tanh(),
        )
        */

        long observationSize = Observation.OBSERVATION_SPACE_SIZE;

        LinearImpl networkLinear1 = createLinearLayer(observationSize, 64);
        TanhImpl networkTanh1 = new TanhImpl();
        LinearImpl networkLinear2 = createLinearLayer(64, 64);
        TanhImpl networkTanh2 = new TanhImpl();

        SequentialImpl network = new SequentialImpl();
        network.push_back("network_linear1", networkLinear1);
        network.push_back("network_tanh1", networkTanh1);
        network.push_back("network_linear2", networkLinear2);
        network.push_back("network_tanh2", networkTanh2);

        register_module("network", network);
        this.network = network;


        /*
        self.lstm = nn.LSTM(64, 64)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        */

        LSTMImpl lstm = new LSTMImpl(64, 64);
        StringVector keys = lstm.named_parameters().keys();
        for (int i = 0; i < keys.size(); i++) {
            String name = keys.get(i).getString();
            Tensor param = lstm.named_parameters().get(name);
            if (name.contains("bias")) {
                torch.constant_(param, new Scalar(0.0f));
            } else if (name.contains("weight")) {
                orthogonal_(param, Math.sqrt(1.0));
            }
        }

        register_module("lstm", lstm);
        this.lstm = lstm;


        /*
        self.x_actor = layer_init(nn.Linear(64, 3), std=0.01)
        self.y_actor = layer_init(nn.Linear(64, 3), std=0.01)
        */

        LinearImpl actorForwardMoveKeys = createLinearLayer(64, 3, 0.01);
        register_module("actor_forward_move_keys", actorForwardMoveKeys);
        this.actorForwardMoveKeys = actorForwardMoveKeys;

        LinearImpl actorStrafingMoveKeys = createLinearLayer(64, 3, 0.01);
        register_module("actor_strafing_move_keys", actorStrafingMoveKeys);
        this.actorStrafingMoveKeys = actorStrafingMoveKeys;

        /*
        self.rot_mean = layer_init(nn.Linear(64, 1), std=0.01)
        self.rot_logstd = nn.Parameter(torch.ones(1) * -1.0)
        */

        LinearImpl yawMean = createLinearLayer(64, 1, 0.01);
        register_module("yaw_mean", yawMean);
        this.yawMean = yawMean;
        Tensor yawLogSTD = torch.ones(new long[]{1}, new TensorOptions(torch.ScalarType.Float)).mul(new Scalar(-1.0f));
        register_parameter("yaw_logstd", yawLogSTD);
        this.yawLogSTD = yawLogSTD;

        LinearImpl pitchMean = createLinearLayer(64, 1, 0.01);
        register_module("pitch_mean", pitchMean);
        this.pitchMean = pitchMean;
        Tensor pitchLogSTD = torch.ones(new long[]{1}, new TensorOptions(torch.ScalarType.Float)).mul(new Scalar(-1.0f));
        register_parameter("pitch_logstd", pitchLogSTD);
        this.pitchLogSTD = pitchLogSTD;

        // Jump
        LinearImpl actorJumpKey = createLinearLayer(64, 2, 0.01); // 2 outputs: jump or not jump
        register_module("actor_jump_key", actorJumpKey);
        this.actorJumpKey = actorJumpKey;

        // Sprint & Sneak

        LinearImpl actorSprintSneakKeys = createLinearLayer(64, 3, 0.01); // 3 outputs: no sprint/sneak, sprint, sneak
        register_module("actor_sprint_sneak_keys", actorSprintSneakKeys);
        this.actorSprintSneakKeys = actorSprintSneakKeys;

        // Attack & Use
        LinearImpl actorAttackUseItem = createLinearLayer(64, 3, 0.01); // 3 outputs: no attack/use, attack, use
        register_module("actor_attack_use_item", actorAttackUseItem);
        this.actorAttackUseItem = actorAttackUseItem;

        /*
        self.critic = layer_init(nn.Linear(64, 1), std=1)
        */

        LinearImpl criticLinear = createLinearLayer(64, 1, 1.0);
        register_module("critic", criticLinear);
        this.critic = criticLinear;
    }

    static void orthogonal_(Tensor tensor, double std) {
        // if tensor.ndimension() < 2:
        //  raise ValueError("Only tensors with 2 or more dimensions are supported")

        if (tensor.ndimension() < 2) {
            throw new IllegalArgumentException("Only tensors with 2 or more dimensions are supported");
        }

        // if tensor.numel() == 0:
        //     # no-op
        //     return tensor

        if (tensor.numel() == 0) {
            return;
        }

        Device originalDevice = tensor.device();

        // rows = tensor.size(0)
        // cols = tensor.numel() // rows
        // flattened = tensor.new_empty((rows, cols)).normal_(0, 1, generator=generator)

        long rows = tensor.size(0);
        long cols = tensor.numel() / rows;
        Tensor flattened = torch.empty(rows, cols).normal_(0, 1, null);

        // if rows < cols:
        //     flattened.t_()

        if (rows < cols) {
            flattened.t_(); // Transpose the tensor if rows < cols
            // long temp = rows;
            // rows = cols;
            // cols = temp;
        }

        // # Compute the qr factorization
        // q, r = torch.linalg.qr(flattened)
        // # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        // d = torch.diag(r, 0)
        // ph = d.sign()
        // q *= ph

        // Move tensor to GPU because LAPACK is not available on CPU in libtorch Java
        flattened = flattened.to(TrainingManager.device, torch.ScalarType.Float);
        T_TensorTensor_T qr = torch.linalg_qr(flattened);

        Tensor qGpu = qr.get0();
        Tensor rGpu = qr.get1();
        Tensor dGpu = torch.diag(rGpu, 0);
        Tensor phGpu = dGpu.sign();
        qGpu = qGpu.mul(phGpu); // Make Q uniform according to the paper

        // if rows < cols:
        //     q.t_()

        if (rows < cols) {
            qGpu.t_(); // Transpose
        }

        // with torch.no_grad():
        //     tensor.view_as(q).copy_(q)
        //     tensor.mul_(gain)

        Tensor q = qGpu.to(originalDevice, torch.ScalarType.Float);

        AutogradState.get_tls_state().set_grad_mode(false);

        tensor.view_as(q).copy_(q); // Copy the orthogonal matrix to the tensor
        tensor.mul_(new Scalar(std)); // Scale the tensor by the standard deviation

        AutogradState.get_tls_state().set_grad_mode(true);
    }

    static LinearImpl createLinearLayer(long inputsDim, long outputDims, double std) {
        LinearImpl layer = new LinearImpl(inputsDim, outputDims);

        orthogonal_(layer.weight(), std);
        torch.constant_(layer.bias(), new Scalar(0.0f));
        return layer;
    }

    static LinearImpl createLinearLayer(long inputsDim, long outputDims) {
        double root2 = Math.sqrt(2.0);
        return createLinearLayer(inputsDim, outputDims, root2);
    }

    // Tensor observation is [B, OBSERVATION_SPACE_SIZE]
    public States getStates(Tensor observation, LSTMState lstmState, Tensor done) {
        /*
        hidden = self.network(x)
         */

        Tensor hidden = this.network.forward(observation);

        /*
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
         */

        Tensor startingHiddenState = lstmState.hiddenState();
        long batchSize = startingHiddenState.size(1); // batchSize
        hidden = hidden.reshape(-1, batchSize, this.lstm.options().input_size().get()); // size (B, batchSize, input_size)
        done = done.reshape(-1, batchSize); // size (B, batchSize)

        long seqLen = hidden.size(0); // seqLen = B

        TensorVector newHidden = new TensorVector();

        Tensor hiddenState = startingHiddenState.clone();
        Tensor cellState = lstmState.cellState().clone();

        startingHiddenState.close();

        Tensor ones = torch.ones_like(done, new TensorOptions(TrainingManager.device), null); // size (B, batchSize)
        Tensor oneSubDone = ones.sub_(done); // size (B, batchSize)

        TensorVector hiddenList = torch.unbind(hidden, 0);
        TensorVector oneSubDoneList = torch.unbind(oneSubDone, 0);

        /*
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
         */
        for (int i = 0; i < seqLen; i++) {
            Tensor h = hiddenList.get(i).unsqueeze(0); // size (1, batchSize, input_size)
            Tensor d = oneSubDoneList.get(i) // size (batchSize,)
                .view(1, -1, 1); // Reshape to (1, batchSize, 1)

            Tensor newHiddenState = hiddenState.mul(d); // Hidden state size (1, batchSize, hidden_size)
            Tensor newCellState = cellState.mul(d); // Cell state size (1, batchSize, hidden_size)

            hiddenState.close();
            cellState.close();

            T_TensorTensor_T inputState = new T_TensorTensor_T(newHiddenState, newCellState);
            T_TensorT_TensorTensor_T_T hNew_LSTMState = this.lstm.forward(h, inputState);

            newHiddenState.close();
            newCellState.close();

            newHidden.push_back(hNew_LSTMState.get0());

            T_TensorTensor_T outState = hNew_LSTMState.get1();
            hiddenState = outState.get0();
            cellState = outState.get1();
        }

        /*
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
         */
        Tensor newHiddenTensor = torch.flatten(torch.cat(newHidden), 0, 1);

        /*
        return new_hidden, lstm_state
         */
        return new States(newHiddenTensor, new LSTMState(hiddenState, cellState));
    }

    /**
     * Get the value from the critic head.
     *
     * @param observation Shape: (numEnvs, OBSERVATION_SPACE_SIZE)
     * @param lstmState   LSTM state containing hidden and cell states.
     * @param done        Shape: (numEnvs, 1) - 1 if done, 0 otherwise.
     * @return Shape (numEnvs, 1) - the value for each environment.
     */
    public Tensor getValue(Tensor observation, LSTMState lstmState, Tensor done) {
        /*
        hidden, _ = self.get_states(x, lstm_state, done)
         */

        States states = this.getStates(observation, lstmState, done);
        Tensor hidden = states.newHiddenTensor;

        /*
        return self.critic(hidden)
         */
        return this.critic.forward(hidden);
    }

    public ActionAndValue getActionAndValue(Tensor nextObs, LSTMState nextLstmState, Tensor nextDone) {
        return getActionAndValue(nextObs, nextLstmState, nextDone, null);
    }

    public ActionAndValue getActionAndValue(Tensor observation, LSTMState lstmState, Tensor done, @Nullable Tensor action) {
        /*
        hidden, lstm_state = self.get_states(x, lstm_state, done)
         */
        States states = this.getStates(observation, lstmState, done);
        Tensor hidden = states.newHiddenTensor;

        /*
        x_logits = self.x_actor(hidden)
        y_logits = self.y_actor(hidden)
         */

        Tensor forwardMoveKeysLogits = this.actorForwardMoveKeys.forward(hidden);
        Tensor strafingMoveKeysLogits = this.actorStrafingMoveKeys.forward(hidden);

        /*
        x_probs = Categorical(logits=x_logits)
        y_probs = Categorical(logits=y_logits)
         */

        Categorical forwardMoveKeysProbs = new Categorical(forwardMoveKeysLogits);
        Categorical strafingMoveKeysProbs = new Categorical(strafingMoveKeysLogits);

        /*
        rot_mean = self.rot_mean(hidden).squeeze(-1)  # (batch,)
        rot_std = torch.exp(self.rot_logstd)
        rot_dist = Normal(rot_mean, rot_std)
         */

        Tensor yawMean = this.yawMean.forward(hidden).squeeze(-1); // Shape (batch,)
        Tensor yawStd = torch.exp(this.yawLogSTD);
        Normal yawDist = new Normal(yawMean, yawStd);

        Tensor pitchMean = this.pitchMean.forward(hidden).squeeze(-1); // Shape (batch,)
        Tensor pitchStd = torch.exp(this.pitchLogSTD);
        Normal pitchDist = new Normal(pitchMean, pitchStd);

        // Jump

        Tensor jumpKeyLogits = this.actorJumpKey.forward(hidden);
        Categorical jumpKeyProbs = new Categorical(jumpKeyLogits);

        // Sprint & Sneak

        Tensor sprintSneakKeysLogits = this.actorSprintSneakKeys.forward(hidden);
        Categorical sprintSneakKeysProbs = new Categorical(sprintSneakKeysLogits);

        // Attack & Use Item

        Tensor attackUseItemLogits = this.actorAttackUseItem.forward(hidden);
        Categorical attackUseItemProbs = new Categorical(attackUseItemLogits);

        /*
        if action is None:
            x_action = x_probs.sample()
            y_action = y_probs.sample()
            rot_action = rot_dist.sample()

            action = torch.stack([x_action.float(), y_action.float(), rot_action], dim=1)
        else:
            # when actions are provided from the buffer, first two may be floats that encode indices
            x_action = action[:, 0].long()
            y_action = action[:, 1].long()
            rot_action = action[:, 2].to(rot_mean.dtype)
         */

        Tensor forwardMoveKeysAction;
        Tensor strafingMoveKeysAction;
        Tensor yawAction;
        Tensor pitchAction;
        Tensor jumpKeyAction;
        Tensor sprintSneakKeysAction;
        Tensor attackUseItemAction;

        if (action == null) {
            forwardMoveKeysAction = forwardMoveKeysProbs.sample(); // LongTensor
            strafingMoveKeysAction = strafingMoveKeysProbs.sample(); // LongTensor
            yawAction = yawDist.sample(); // FloatTensor
            pitchAction = pitchDist.sample(); // FloatTensor
            jumpKeyAction = jumpKeyProbs.sample(); // LongTensor
            sprintSneakKeysAction = sprintSneakKeysProbs.sample(); // LongTensor
            attackUseItemAction = attackUseItemProbs.sample(); // LongTensor

            // ! THIS MUST MATCH THE ORDER IN Action CLASS
            action = torch.stack(new TensorVector(
                jumpKeyAction.to(torch.ScalarType.Float),
                sprintSneakKeysAction.to(torch.ScalarType.Float),
                yawAction,
                pitchAction,
                forwardMoveKeysAction.to(torch.ScalarType.Float),
                strafingMoveKeysAction.to(torch.ScalarType.Float),
                attackUseItemAction.to(torch.ScalarType.Float)
            ), 1);
        } else {
            // ! THIS MUST MATCH THE ORDER IN Action CLASS
            jumpKeyAction = action.narrow(1, 0, 1).squeeze(1).to(torch.ScalarType.Long);
            sprintSneakKeysAction = action.narrow(1, 1, 1).squeeze(1).to(torch.ScalarType.Long);
            yawAction = action.narrow(1, 2, 1).squeeze(1).to(yawMean.dtype());
            pitchAction = action.narrow(1, 3, 1).squeeze(1).to(pitchMean.dtype());
            forwardMoveKeysAction = action.narrow(1, 4, 1).squeeze(1).to(torch.ScalarType.Long);
            strafingMoveKeysAction = action.narrow(1, 5, 1).squeeze(1).to(torch.ScalarType.Long);
            attackUseItemAction = action.narrow(1, 6, 1).squeeze(1).to(torch.ScalarType.Long);
        }

        /*
        x_logprob = x_probs.log_prob(x_action)
        y_logprob = y_probs.log_prob(y_action)
        rot_logprob = rot_dist.log_prob(rot_action)

        logprobs = x_logprob + y_logprob + rot_logprob
         */

        Tensor forwardMoveKeysLogProbs = forwardMoveKeysProbs.logProb(forwardMoveKeysAction);
        Tensor strafingMoveKeysLogProbs = strafingMoveKeysProbs.logProb(strafingMoveKeysAction);
        Tensor yawLogProbs = yawDist.logProb(yawAction);
        Tensor pitchLogProbs = pitchDist.logProb(pitchAction);
        Tensor jumpKeyLogProbs = jumpKeyProbs.logProb(jumpKeyAction);
        Tensor sprintSneakKeysLogProbs = sprintSneakKeysProbs.logProb(sprintSneakKeysAction);
        Tensor attackUseItemLogProbs = attackUseItemProbs.logProb(attackUseItemAction);

        Tensor totalLogProbs = forwardMoveKeysLogProbs
            .add(strafingMoveKeysLogProbs)
            .add(yawLogProbs)
            .add(pitchLogProbs)
            .add(jumpKeyLogProbs)
            .add(sprintSneakKeysLogProbs)
            .add(attackUseItemLogProbs);

        /*
        entropy = x_probs.entropy() + y_probs.entropy() + rot_dist.entropy()
         */

        Tensor forwardMoveKeysEntropy = forwardMoveKeysProbs.entropy();
        Tensor strafingMoveKeysEntropy = strafingMoveKeysProbs.entropy();
        Tensor yawEntropy = yawDist.entropy();
        Tensor pitchEntropy = pitchDist.entropy();
        Tensor jumpKeyEntropy = jumpKeyProbs.entropy();
        Tensor sprintSneakKeysEntropy = sprintSneakKeysProbs.entropy();
        Tensor attackUseItemEntropy = attackUseItemProbs.entropy();

        Tensor totalEntropy = forwardMoveKeysEntropy
            .add(strafingMoveKeysEntropy)
            .add(yawEntropy)
            .add(pitchEntropy)
            .add(jumpKeyEntropy)
            .add(sprintSneakKeysEntropy)
            .add(attackUseItemEntropy);

        /*
        return (
            action,
            logprobs,
            entropy,
            self.critic(hidden),
            lstm_state,
        )
         */

        Tensor value = this.critic.forward(hidden);

        return new ActionAndValue(
            action,
            totalLogProbs,
            totalEntropy,
            value,
            states.lstmState
        );
    }

    public long getLSTMLayers() {
        return this.lstm.options().num_layers().get();
    }

    public long getLSTMHiddenSize() {
        return this.lstm.options().hidden_size().get();
    }

    public void saveCheckpoint(int iteration) {
        OutputArchive outputArchive = new OutputArchive();
        this.save(outputArchive);
        outputArchive.save_to("model_files/minecraft_rl_checkpoint_" + iteration + ".pt");
    }

    public void loadCheckpoint(Integer iteration) {
        if (iteration == null) {
            return;
        }
        InputArchive inputArchive = new InputArchive();
        inputArchive.load_from("model_files/minecraft_rl_checkpoint_" + iteration + ".pt");
        System.out.println("Loading MinecraftRL checkpoint from iteration " + iteration);
        this.load(inputArchive);

    }

    /**
     * Holds the state of an LSTM layer.
     * <p>
     * hiddenState Shape (numEnvs, batchSize, hiddenSize)
     * cellState   Shape (numEnvs, batchSize, hiddenSize)
     */
    public static final class LSTMState implements AutoCloseable {
        private T_TensorTensor_T lstmState;

        public LSTMState(T_TensorTensor_T lstmState) {this.lstmState = lstmState;}

        public LSTMState(Tensor hiddenState, Tensor cellState) {
            this(new T_TensorTensor_T(hiddenState, cellState));
        }

        public void set(T_TensorTensor_T hNewLstmState1) {
            this.lstmState = hNewLstmState1;
        }

        public Tensor hiddenState() {return lstmState.get0();}

        public Tensor cellState() {return lstmState.get1();}

        public T_TensorTensor_T lstmState() {return lstmState;}

        @Override
        public LSTMState clone() {
            Tensor hiddenState = this.hiddenState();
            Tensor cellState = this.cellState();
            Tensor clonedHidden = hiddenState.clone();
            Tensor clonedCell = cellState.clone();
            hiddenState.close();
            cellState.close();
            return new LSTMState(clonedHidden, clonedCell);
        }

        @Override
        public void close() {
            this.lstmState.get0().close();
            this.lstmState.get1().close();
            this.lstmState.close();
        }

        public void retainReference() {
            this.lstmState.get0().retainReference();
            this.lstmState.get1().retainReference();
            this.lstmState.retainReference();
        }

        public void copy_(LSTMState lstmState) {
            this.hiddenState().copy_(lstmState.hiddenState());
            this.cellState().copy_(lstmState.cellState());
        }
    }

    /**
     * Holds the new hidden tensor and the LSTM state after processing an observation.
     *
     * @param newHiddenTensor Shape (numEnvs, batchSize, hiddenSize)
     * @param lstmState       The LSTM state after processing the observation.
     */
    public record States(Tensor newHiddenTensor, MinecraftRL.LSTMState lstmState) {}

    /**
     * Holds the action, total log probabilities, total entropy, value, and LSTM state.
     *
     * @param action        The action tensor.
     * @param totalLogProbs The total log probabilities of the action.
     * @param totalEntropy  The total entropy of the action distribution.
     * @param value         The value tensor from the critic head.
     * @param lstmState     The LSTM state after processing the observation.
     */
    public record ActionAndValue(Tensor action, Tensor totalLogProbs, Tensor totalEntropy, Tensor value,
                                 MinecraftRL.LSTMState lstmState) {}
}
