package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.environment.MinecraftEnvironment;
import com.tenbitmelon.machinelearningplayer.environment.Observation;
import com.tenbitmelon.machinelearningplayer.util.distrobutions.Bernoulli;
import com.tenbitmelon.machinelearningplayer.util.distrobutions.Normal;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.Module;

import javax.annotation.Nullable;

public class MinecraftRL extends Module {

    private final SequentialImpl voxelCNN;
    private final SequentialImpl otherInputsProcessor;
    private final SequentialImpl sharedNetwork;
    private final LSTMImpl lstm;
    private final LinearImpl critic;
    private final LinearImpl actorLookChangeMean;
    private final Tensor actorLookChangeLogSTD;
    private final LinearImpl actorSprintKey;
    private final LinearImpl actorSneakKey;
    private final LinearImpl actorJumpKey;
    private final LinearImpl actorMoveKeys;

    public long otherFeaturesDim = 128;
    public long sharedOutDim = 256;
    public long lstmHiddenSize = 128;

    public MinecraftRL(SyncedVectorEnvironment environment) {

        /*
        self.voxel_cnn = nn.Sequential(
            # Input: (B, 1, X, Z, Y)
            layer_init(nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))),
            nn.ReLU(),
            layer_init(nn.Conv3d( 16, 32, kernel_size=(3, 3, MinecraftDummyEnv.GRID_SIZE_Y), stride=(1, 1, 1), padding=(1, 1, 0))),
            nn.ReLU(),
            layer_init(nn.Conv3d(32, 64, kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))),
            nn.ReLU(),
            nn.Flatten(),
        )
         */


        final Conv3dImpl conv1 = createConv3d(1, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1);
        final ReLUImpl relu = new ReLUImpl();
        final Conv3dImpl conv2 = createConv3d(16, 32, 3, 3, MinecraftEnvironment.GRID_SIZE_Y, 1, 1, 1, 1, 1, 0);
        final ReLUImpl relu2 = new ReLUImpl();
        final Conv3dImpl conv3 = createConv3d(32, 64, 3, 3, 1, 2, 2, 1, 1, 1, 0);
        final ReLUImpl relu3 = new ReLUImpl();
        final FlattenImpl flatten = new FlattenImpl();

        final SequentialImpl sequential = new SequentialImpl();
        sequential.push_back("conv1", conv1);
        sequential.push_back("relu", relu);
        sequential.push_back("conv2", conv2);
        sequential.push_back("relu2", relu2);
        sequential.push_back("conv3", conv3);
        sequential.push_back("relu3", relu3);
        sequential.push_back("flatten", flatten);

        register_module("voxel_cnn", sequential);
        this.voxelCNN = sequential;

        /*
        # Sneakily calculate the ending size of the CNN output
        with torch.no_grad():
            # Voxel grid has shape (X, Z, Y) -> add channel dim -> (B,1,X,Z,Y)
            dummy_voxel_input = torch.zeros(
                1,
                1,
                MinecraftDummyEnv.GRID_SIZE_XZ,
                MinecraftDummyEnv.GRID_SIZE_XZ,
                MinecraftDummyEnv.GRID_SIZE_Y,
            )
            cnn_out_dim = self.voxel_cnn(dummy_voxel_input).shape[1]
            print("cnn_out_dim", cnn_out_dim)
         */

        // Sneakily calculate the ending size of the CNN output
        long cnnOutDim = 0;
        try (Tensor dummyVoxelInput = torch.zeros(
            1,
            1,
            MinecraftEnvironment.GRID_SIZE_XZ,
            MinecraftEnvironment.GRID_SIZE_XZ,
            MinecraftEnvironment.GRID_SIZE_Y)) {
            Tensor cnnOut = sequential.forward(dummyVoxelInput);
            cnnOutDim = cnnOut.shape()[1];
            System.out.println("cnn_out_dim: " + cnnOutDim);
        }


        /*
        # 2. Other Inputs Processor (Dense Layers)
        other_inputs_dim = (
            MinecraftDummyEnv.OBSERVATION_SPACE_SIZE - MinecraftDummyEnv.GRID_VOLUME
        )

        other_features_dim = 128
        self.other_inputs_processor = nn.Sequential(
            layer_init(nn.Linear(other_inputs_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, other_features_dim)),
            nn.ReLU(),
        )
        */

        long otherInputsDim = Observation.OBSERVATION_SPACE_SIZE - MinecraftEnvironment.GRID_VOLUME;


        LinearImpl otherLinear1 = createLinearLayer(otherInputsDim, 64);
        ReLUImpl otherReLU1 = new ReLUImpl();
        LinearImpl otherLinear2 = createLinearLayer(64, otherFeaturesDim);
        ReLUImpl otherReLU2 = new ReLUImpl();

        SequentialImpl otherInputsProcessor = new SequentialImpl();
        otherInputsProcessor.push_back("other_linear1", otherLinear1);
        otherInputsProcessor.push_back("other_relu1", otherReLU1);
        otherInputsProcessor.push_back("other_linear2", otherLinear2);
        otherInputsProcessor.push_back("other_relu2", otherReLU2);

        register_module("other_inputs_processor", otherInputsProcessor);
        this.otherInputsProcessor = otherInputsProcessor;


        /*
        # 3. Shared Network (after concatenation)
        combined_features_dim = cnn_out_dim + other_features_dim
        shared_out_dim = 256
        self.shared_network = nn.Sequential(
            layer_init(nn.Linear(combined_features_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, shared_out_dim)),
            nn.ReLU(),
        )
        */

        long combinedFeaturesDim = cnnOutDim + otherFeaturesDim;

        LinearImpl sharedLinear1 = createLinearLayer(combinedFeaturesDim, 256);
        ReLUImpl sharedReLU1 = new ReLUImpl();
        LinearImpl sharedLinear2 = createLinearLayer(256, sharedOutDim);
        ReLUImpl sharedReLU2 = new ReLUImpl();

        SequentialImpl sharedNetwork = new SequentialImpl();
        sharedNetwork.push_back("shared_linear1", sharedLinear1);
        sharedNetwork.push_back("shared_relu1", sharedReLU1);
        sharedNetwork.push_back("shared_linear2", sharedLinear2);
        sharedNetwork.push_back("shared_relu2", sharedReLU2);

        register_module("shared_network", sharedNetwork);
        this.sharedNetwork = sharedNetwork;

        /*
        # 4. LSTM
        lstm_hidden_size = 128
        self.lstm = nn.LSTM(shared_out_dim, lstm_hidden_size)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))
         */


        LSTMImpl lstm = new LSTMImpl(sharedOutDim, lstmHiddenSize);
        StringVector keys = lstm.named_parameters().keys();
        for (int i = 0; i < keys.size(); i++) {
            String name = keys.get(i).getString();
            Tensor param = lstm.named_parameters().get(name);
            if (name.contains("bias")) {
                torch.constant_(param, new Scalar(0.0f));
            } else if (name.contains("weight")) {
                // torch.orthogonal_(param, Math.sqrt(2.0));
            }
        }

        register_module("lstm", lstm);
        this.lstm = lstm;


        /*
        # 5. Actor Heads (outputting parameters for distributions)
        actor_input_dim = lstm_hidden_size

        self.actor_look_change_mean = layer_init(
            nn.Linear(actor_input_dim, 2), std=0.01
        )  # d_yaw, d_pitch means
        self.actor_look_change_logstd = nn.Parameter(
            torch.zeros(1, 2)
        )  # Learnable log_std for look_change

        self.actor_sprint_key = layer_init(
            nn.Linear(actor_input_dim, 1), std=0.01
        )  # Logit for sprint
        self.actor_sneak_key = layer_init(
            nn.Linear(actor_input_dim, 1), std=0.01
        )  # Logit for sneak
        self.actor_jump_key = layer_init(
            nn.Linear(actor_input_dim, 1), std=0.01
        )  # Logit for jump
        self.actor_move_keys = layer_init(
            nn.Linear(actor_input_dim, 4), std=0.01
        )  # Logits for W,S,A,D

        # Critic Head
        self.critic = layer_init(nn.Linear(actor_input_dim, 1), std=1)
         */

        long actorInputDim = lstmHiddenSize;
        LinearImpl actorLookChangeMean = createLinearLayer(actorInputDim, 2, 0.01);
        Tensor actorLookChangeLogSTD = torch.zeros(1, 2);

        LinearImpl actorSprintKey = createLinearLayer(actorInputDim, 1, 0.01);
        LinearImpl actorSneakKey = createLinearLayer(actorInputDim, 1, 0.01);
        LinearImpl actorJumpKey = createLinearLayer(actorInputDim, 1, 0.01);
        LinearImpl actorMoveKeys = createLinearLayer(actorInputDim, 4, 0.01);

        LinearImpl critic = createLinearLayer(actorInputDim, 1, 1.0);

        register_module("actor_look_change_mean", actorLookChangeMean);
        register_parameter("actor_look_change_logstd", actorLookChangeLogSTD);
        register_module("actor_sprint_key", actorSprintKey);
        register_module("actor_sneak_key", actorSneakKey);
        register_module("actor_jump_key", actorJumpKey);
        register_module("actor_move_keys", actorMoveKeys);
        register_module("critic", critic);

        this.actorLookChangeMean = actorLookChangeMean;
        this.actorLookChangeLogSTD = actorLookChangeLogSTD;
        this.actorSprintKey = actorSprintKey;
        this.actorSneakKey = actorSneakKey;
        this.actorJumpKey = actorJumpKey;
        this.actorMoveKeys = actorMoveKeys;
        this.critic = critic;
    }

    static LinearImpl createLinearLayer(long inputsDim, long outputDims, double std) {
        LinearImpl layer = new LinearImpl(inputsDim, outputDims);
        // torch.orthogonal_(layer.weight(), std);
        torch.constant_(layer.bias(), new Scalar(0.0f));
        return layer;
    }

    static LinearImpl createLinearLayer(long inputsDim, long outputDims) {
        double root2 = Math.sqrt(2.0);
        return createLinearLayer(inputsDim, outputDims, root2);
    }

    static Conv3dImpl createConv3d(int inChannels, int outChannels, int kernelSize1, int kernelSize2, int kernelSize3, int stride1, int stride2, int stride3, int padding1, int padding2, int padding3) {
        LongPointer kernelSizePointer = new LongPointer(3);
        kernelSizePointer.put(kernelSize1, kernelSize2, kernelSize3);
        Conv3dOptions conv3dOptions = new Conv3dOptions(inChannels, outChannels, kernelSizePointer);
        LongPointer stridePointer = conv3dOptions.stride();
        stridePointer.put(stride1, stride2, stride3);
        Conv3dPadding paddingPointer = conv3dOptions.padding();
        LongPointer padding0 = paddingPointer.get0();
        padding0.put(padding1, padding2, padding3);

        Conv3dImpl layer = new Conv3dImpl(conv3dOptions);

        // double root2 = Math.sqrt(2.0);
        // torch.orthogonal_(layer.weight(), root2);
        torch.constant_(layer.bias(), new Scalar(0.0f));
        return layer;
    }

    public States getStates(Tensor observation, LSTMState lstmState, Tensor done) {
        /*
        voxel_data = x[:, : MinecraftDummyEnv.GRID_VOLUME]
        other_inputs = x[:, MinecraftDummyEnv.GRID_VOLUME :]
        */
        Tensor voxelData = observation.narrow(1, 0, MinecraftEnvironment.GRID_VOLUME);
        Tensor otherInputs = observation.narrow(1, MinecraftEnvironment.GRID_VOLUME, Observation.OBSERVATION_SPACE_SIZE - MinecraftEnvironment.GRID_VOLUME);

        /*
        # Reshape voxel data from (75) to (5, 5, 3)
        voxel_data = voxel_data.view(
            x.shape[0],
            MinecraftDummyEnv.GRID_SIZE_XZ,
            MinecraftDummyEnv.GRID_SIZE_XZ,
            MinecraftDummyEnv.GRID_SIZE_Y,
        )
        voxel_data = voxel_data.unsqueeze(1)  # (B, 1, X, Z, Y)

        voxel_features = self.voxel_cnn(voxel_data / 2.0)

        # Concatenate other inputs
        other_features = self.other_inputs_processor(other_inputs)
        combined_features = torch.cat([voxel_features, other_features], dim=1)
        shared_features = self.shared_network(combined_features)
         */

        voxelData = voxelData.view(
            observation.size(0),
            MinecraftEnvironment.GRID_SIZE_XZ,
            MinecraftEnvironment.GRID_SIZE_XZ,
            MinecraftEnvironment.GRID_SIZE_Y
        ).unsqueeze(1); // (B, 1, X, Z, Y)

        Tensor voxelFeatures = this.voxelCNN.forward(voxelData.div(new Scalar(2.0f)));

        Tensor otherFeatures = this.otherInputsProcessor.forward(otherInputs);
        TensorVector tensorsToCombine = new TensorVector(voxelFeatures, otherFeatures);
        Tensor combinedFeatures = torch.cat(tensorsToCombine, 1);
        Tensor sharedFeatures = this.sharedNetwork.forward(combinedFeatures);

        /*
        # # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = shared_features.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
         */

        long batchSize = lstmState.hiddenState.size(1);
        Tensor hidden = sharedFeatures.reshape(-1, batchSize, this.lstm.options().input_size().get());
        done = done.reshape(-1, batchSize);
        TensorVector newHidden = new TensorVector();
        for (int i = 0; i < hidden.size(0); i++) {
            Tensor h = hidden.get(i);
            Tensor d = done.get(i);

            // Hidden state
            Tensor oneMinusD = Tensor.create(1.0).sub(d).view(1, -1, 1);
            Tensor hState = oneMinusD.mul(lstmState.hiddenState).toType(torch.ScalarType.Float);
            // Cell state
            Tensor cState = oneMinusD.mul(lstmState.cellState).toType(torch.ScalarType.Float);

            T_TensorTensor_T tTensorTensorT = new T_TensorTensor_T(hState, cState);

            T_TensorT_TensorTensor_T_T hNew_LSTMState = this.lstm.forward(h.unsqueeze(0), tTensorTensorT);
            newHidden.push_back(hNew_LSTMState.get0());
            lstmState = new LSTMState(hNew_LSTMState.get1().get0(), hNew_LSTMState.get1().get1());
        }
        Tensor newHiddenTensor = torch.flatten(torch.cat(newHidden), 0, 1);

        /*
        return new_hidden, lstm_state
         */

        return new States(newHiddenTensor, lstmState);
    }

    public Tensor getValue(Tensor observation, LSTMState lstmState, Tensor done) {
        /*
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)
         */
        States states = this.getStates(observation, lstmState, done);
        Tensor hidden = states.newHiddenTensor;
        Tensor value = this.critic.forward(hidden);
        return value;
    }

    public ActionAndValue getActionAndValue(Tensor nextObs, LSTMState nextLstmState, Tensor nextDone) {
        return getActionAndValue(nextObs, nextLstmState, nextDone, null);
    }

    public ActionAndValue getActionAndValue(Tensor observation, LSTMState lstmState, Tensor done, @Nullable Tensor action) {
        States states = this.getStates(observation, lstmState, done);
        Tensor sharedFeatures = states.newHiddenTensor;

        /*
        # --- Actor Heads ---
        # Look Change (Continuous - Gaussian)
        look_means = torch.tanh(
            self.actor_look_change_mean(actor_critic_features)
        )  # Squash means to [-1, 1]
        look_std = torch.exp(self.actor_look_change_logstd.expand_as(look_means))
        look_dist = torch.distributions.Normal(look_means, look_std)

        # Sprint Key (Discrete - Bernoulli)
        sprint_logits = self.actor_sprint_key(actor_critic_features)
        sprint_dist = torch.distributions.Bernoulli(logits=sprint_logits)

        # Sneak Key (Discrete - Bernoulli)
        sneak_logits = self.actor_sneak_key(actor_critic_features)
        sneak_dist = torch.distributions.Bernoulli(logits=sneak_logits)

        # Jump Key (Discrete - Bernoulli)
        jump_logits = self.actor_jump_key(actor_critic_features)
        jump_dist = torch.distributions.Bernoulli(logits=jump_logits)

        # Move Keys (MultiBinary - Bernoulli for each)
        move_logits = self.actor_move_keys(actor_critic_features)  # Shape: (B, 4)
        move_dist = torch.distributions.Bernoulli(
            logits=move_logits
        )  # Will sample (B,4) of 0s and 1s
         */

        Tensor lookMeans = torch.tanh(this.actorLookChangeMean.forward(sharedFeatures));
        Tensor lookStd = torch.exp(this.actorLookChangeLogSTD.expand(lookMeans.shape()));
        Normal lookDist = new Normal(lookMeans, lookStd);

        Tensor sprintLogits = this.actorSprintKey.forward(sharedFeatures);
        Bernoulli sprintDist = new Bernoulli(sprintLogits);

        Tensor sneakLogits = this.actorSneakKey.forward(sharedFeatures);
        Bernoulli sneakDist = new Bernoulli(sneakLogits);

        Tensor jumpLogits = this.actorJumpKey.forward(sharedFeatures);
        Bernoulli jumpDist = new Bernoulli(jumpLogits);

        Tensor moveLogits = this.actorMoveKeys.forward(sharedFeatures); // Shape: (B, 4)
        Bernoulli moveDist = new Bernoulli(moveLogits); // Will sample (B,4) of 0s and 1s

        /*
        if action is None:
            look_sample = look_dist.sample()  # (B, 2) e.g., pitch, yaw
            sprint_sample = sprint_dist.sample()  # (B, 1)
            sneak_sample = sneak_dist.sample()  # (B, 1)
            jump_sample = jump_dist.sample()  # (B, 1)
            move_sample = move_dist.sample()  # (B, 4)

            # Concatenate along the last dimension -> flat per-sample vector
            # Resulting shape: (B, 2+1+1+1+4) = (B, 9)

            action = torch.cat(
                [
                    look_sample,
                    sprint_sample,
                    sneak_sample,
                    jump_sample,
                    move_sample,
                ],
                dim=-1,
            )
         */

        if (action == null) {
            Tensor lookSample = lookDist.sample();
            Tensor sprintSample = sprintDist.sample();
            Tensor sneakSample = sneakDist.sample();
            Tensor jumpSample = jumpDist.sample();
            Tensor moveSample = moveDist.sample();

            // Concatenate along the last dimension -> flat per-sample vector
            // Resulting shape: (B, 2+1+1+1+4) = (B, 9)
            action = torch.cat(new TensorVector(lookSample, sprintSample, sneakSample, jumpSample, moveSample), -1);
        }

        /*
        # Calculate log_probs and entropy
        look_action = action[..., 0:2].float()
        log_probs_look = look_dist.log_prob(look_action).sum(dim=-1)
        sprint_action = action[..., 2:3].float()
        log_probs_sprint = sprint_dist.log_prob(sprint_action).squeeze(-1)
        sneak_action = action[..., 3:4].float()
        log_probs_sneak = sneak_dist.log_prob(sneak_action).squeeze(-1)
        jump_action = action[..., 4:5].float()
        log_probs_jump = jump_dist.log_prob(jump_action).squeeze(-1)
        move_action = action[..., 5:].float()
        log_probs_move = move_dist.log_prob(move_action).sum(dim=-1)

        total_log_probs = (
            log_probs_look
            + log_probs_sprint
            + log_probs_sneak
            + log_probs_jump
            + log_probs_move
        )
         */

        Tensor lookAction = action.narrow(-1, 0, 2);
        Tensor logProbsLook = lookDist.logProb(lookAction).sum(-1);
        Tensor sprintAction = action.narrow(-1, 2, 1);
        Tensor logProbsSprint = sprintDist.logProb(sprintAction).squeeze(-1);
        Tensor sneakAction = action.narrow(-1, 3, 1);
        Tensor logProbsSneak = sneakDist.logProb(sneakAction).squeeze(-1);
        Tensor jumpAction = action.narrow(-1, 4, 1);
        Tensor logProbsJump = jumpDist.logProb(jumpAction).squeeze(-1);
        Tensor moveAction = action.narrow(-1, 5, 4);
        Tensor logProbsMove = moveDist.logProb(moveAction).sum(-1);

        Tensor totalLogProbs = logProbsLook.add(logProbsSprint).add(logProbsSneak)
            .add(logProbsJump).add(logProbsMove);

        /*
        entropy_look = look_dist.entropy().sum(dim=-1)
        entropy_sprint = sprint_dist.entropy().squeeze(-1)
        entropy_sneak = sneak_dist.entropy().squeeze(-1)
        entropy_jump = jump_dist.entropy().squeeze(-1)
        entropy_move = move_dist.entropy().sum(dim=-1)

        total_entropy = (
            entropy_look + entropy_sprint + entropy_sneak + entropy_jump + entropy_move
        )
         */

        Tensor entropyLook = lookDist.entropy().sum(-1);
        Tensor entropySprint = sprintDist.entropy().squeeze(-1);
        Tensor entropySneak = sneakDist.entropy().squeeze(-1);
        Tensor entropyJump = jumpDist.entropy().squeeze(-1);
        Tensor entropyMove = moveDist.entropy().sum(-1);

        Tensor totalEntropy = entropyLook.add(entropySprint).add(entropySneak)
            .add(entropyJump).add(entropyMove);

        /*
        # --- Critic Value ---
        value = self.critic(actor_critic_features)
         */

        Tensor value = this.critic.forward(sharedFeatures);

        /*
        # if action is None, sample new actions
        return (
            action,
            action_dict,
            total_log_probs,
            total_entropy,
            value,
            lstm_state,
        )
         */

        return new ActionAndValue(
            action,
            totalLogProbs,
            totalEntropy,
            value,
            lstmState
        );
    }

    public long getLSTMLayers() {
        return this.lstm.options().num_layers().get();
    }

    public record LSTMState(Tensor hiddenState, Tensor cellState) {
        @Override
        protected LSTMState clone() {
            return new LSTMState(hiddenState.clone(), cellState.clone());
        }
    }

    public record States(Tensor newHiddenTensor, MinecraftRL.LSTMState lstmState) {}

    public record ActionAndValue(Tensor action, Tensor totalLogProbs, Tensor totalEntropy, Tensor value,
                                 MinecraftRL.LSTMState lstmState) {
    }
}
