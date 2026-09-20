package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.ExperimentConfig;
import com.tenbitmelon.machinelearningplayer.environment.Observation;
import com.tenbitmelon.machinelearningplayer.util.distributions.Categorical;
import com.tenbitmelon.machinelearningplayer.util.distributions.Normal;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;
import org.bytedeco.pytorch.global.torch;

import javax.annotation.Nullable;
import java.io.File;

import static com.tenbitmelon.machinelearningplayer.MachineLearningPlayer.LOGGER;

public class MinecraftRL extends Module {

    private static final ScalarOptional SCALAR_n5 = new ScalarOptional(new Scalar(-5.0f));
    private static final ScalarOptional SCALAR_2 = new ScalarOptional(new Scalar(2.0f));
    final ExperimentConfig args;
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
    final Device device;
    final SequentialImpl localHeightmapConv;
    final LinearImpl actorSlot;

    public MinecraftRL(ExperimentConfig args, Device device) {
        this.device = device;
        this.args = args;
        /*
         // Conv2d for local heightmaps
        self.local_heightmap_conv = nn.Sequential(
            layer_init(nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0)),
            nn.Tanh(),
            nn.Flatten(),
            layer_init(nn.Linear(5 * 5 * 4, 16)),
            nn.Tanh(),
         */

        int observationHeightSizeIn = 7 * 7;
        int observationHeightSizeOut = 16;
        SequentialImpl localHeightmapConv = new SequentialImpl();

        if (args.featureFlags.contains(ExperimentConfig.FeatureFlag.HEIGHT_MAP_CONV)) {
            // Conv2d Input size = [B, 1, 7, 7]
            Conv2dOptions conv2dOptions = new Conv2dOptions(1, 4, new LongPointer(3, 3));
            Conv2dImpl conv2d = new Conv2dImpl(conv2dOptions);
            localHeightmapConv.push_back("height_conv_conv2d", conv2d);

            TanhImpl convTanh1 = new TanhImpl();
            localHeightmapConv.push_back("height_conv_tanh1", convTanh1);

            observationHeightSizeIn = 4 * 5 * 5; // 4 channels, 5x5 output from conv2d
        }

        FlattenImpl flatten = new FlattenImpl();
        localHeightmapConv.push_back("height_conv_flatten", flatten);

        LinearImpl convLinear = createLinearLayer(observationHeightSizeIn, observationHeightSizeOut, device);
        localHeightmapConv.push_back("height_conv_linear", convLinear);

        TanhImpl convTanh2 = new TanhImpl();
        localHeightmapConv.push_back("height_conv_tanh2", convTanh2);

        register_module("height_conv", localHeightmapConv);
        this.localHeightmapConv = localHeightmapConv;

        /*
        other_features_dim = 128
        self.network = nn.Sequential(
           layer_init(nn.Linear(obs_shape, 64)),
           nn.Tanh(),
           layer_init(nn.Linear(64, 64)),
           nn.Tanh(),
        )
        */

        long observationSize = Observation.OBSERVATION_SPACE_SIZE - Observation.SIZE_LOCAL_HEIGHT_MAP + observationHeightSizeOut; // 16 is the output size of the local heightmap conv layers

        LinearImpl networkLinear1 = createLinearLayer(observationSize, 64, device);
        TanhImpl networkTanh1 = new TanhImpl();
        LinearImpl networkLinear2 = createLinearLayer(64, 64, device);
        TanhImpl networkTanh2 = new TanhImpl();

        SequentialImpl network = new SequentialImpl();
        network.push_back("network_linear1", networkLinear1);

        if (args.featureFlags.contains(ExperimentConfig.FeatureFlag.LAYER_NORM)) {
            LayerNormOptions layerNormOptions = new LayerNormOptions(new LongPointer(64));
            LayerNormImpl layerNorm = new LayerNormImpl(layerNormOptions);
            network.push_back("network_layer_norm", layerNorm);
        }


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

        int lstmSize = args.featureFlags.contains(ExperimentConfig.FeatureFlag.LSTM_SIZE_128) ? 128 : 64;
        LSTMImpl lstm = new LSTMImpl(64, lstmSize);
        StringVector keys = lstm.named_parameters().keys();
        for (int i = 0; i < keys.size(); i++) {
            String name = keys.get(i).getString();
            Tensor param = lstm.named_parameters().get(name);
            if (name.contains("bias")) {
                torch.constant_(param, new Scalar(0.0f));
            } else if (name.contains("weight")) {
                orthogonal_(param, Math.sqrt(1.0), device);
            }
        }

        register_module("lstm", lstm);
        this.lstm = lstm;


        /*
        self.x_actor = layer_init(nn.Linear(64, 3), std=0.01)
        self.y_actor = layer_init(nn.Linear(64, 3), std=0.01)
        */

        LinearImpl actorForwardMoveKeys = createLinearLayer(lstmSize, 3, 0.01, device);
        register_module("actor_forward_move_keys", actorForwardMoveKeys);
        this.actorForwardMoveKeys = actorForwardMoveKeys;

        LinearImpl actorStrafingMoveKeys = createLinearLayer(lstmSize, 3, 0.01, device);
        register_module("actor_strafing_move_keys", actorStrafingMoveKeys);
        this.actorStrafingMoveKeys = actorStrafingMoveKeys;

        /*
        self.rot_mean = layer_init(nn.Linear(64, 1), std=0.01)
        self.rot_logstd = nn.Parameter(torch.ones(1) * -1.0)
        */

        LinearImpl yawMean = createLinearLayer(lstmSize, 1, 0.01, device);
        register_module("yaw_mean", yawMean);
        this.yawMean = yawMean;
        Tensor yawLogSTD = torch.ones(new long[]{1}, new TensorOptions(torch.ScalarType.Float)).mul(new Scalar(-1.0f));
        register_parameter("yaw_logstd", yawLogSTD);
        this.yawLogSTD = yawLogSTD;

        LinearImpl pitchMean = createLinearLayer(lstmSize, 1, 0.01, device);
        register_module("pitch_mean", pitchMean);
        this.pitchMean = pitchMean;
        Tensor pitchLogSTD = torch.ones(new long[]{1}, new TensorOptions(torch.ScalarType.Float)).mul(new Scalar(-1.0f));
        register_parameter("pitch_logstd", pitchLogSTD);
        this.pitchLogSTD = pitchLogSTD;

        // Jump
        LinearImpl actorJumpKey = createLinearLayer(lstmSize, 2, 0.01, device); // 2 outputs: jump or not jump
        register_module("actor_jump_key", actorJumpKey);
        this.actorJumpKey = actorJumpKey;

        // Sprint & Sneak

        LinearImpl actorSprintSneakKeys = createLinearLayer(lstmSize, 3, 0.01, device); // 3 outputs: no sprint/sneak, sprint, sneak
        register_module("actor_sprint_sneak_keys", actorSprintSneakKeys);
        this.actorSprintSneakKeys = actorSprintSneakKeys;

        // Attack & Use
        LinearImpl actorAttackUseItem = createLinearLayer(lstmSize, 3, 0.01, device); // 3 outputs: no attack/use, attack, use
        register_module("actor_attack_use_item", actorAttackUseItem);
        this.actorAttackUseItem = actorAttackUseItem;

        // Slot 0 or 1
        LinearImpl actorSlot = createLinearLayer(lstmSize, 2, 0.01, device); // 2 outputs: slot 0 or slot 1
        register_module("actor_slot", actorSlot);
        this.actorSlot = actorSlot;

        /*
        self.critic = layer_init(nn.Linear(64, 1), std=1)
        */

        LinearImpl criticLinear = createLinearLayer(lstmSize, 1, 1.0, device);
        register_module("critic", criticLinear);
        this.critic = criticLinear;
    }

    static void orthogonal_(Tensor tensor, double std, Device device) {
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
            flattened.t_(); // (rows, cols) -> (cols, rows)
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
        flattened = flattened.to(device, torch.ScalarType.Float);
        T_TensorTensor_T qr = torch.linalg_qr(flattened);

        Tensor qGpu = qr.get0();
        Tensor rGpu = qr.get1();
        Tensor dGpu = torch.diag(rGpu, 0);
        Tensor phGpu = dGpu.sign();
        qGpu = qGpu.mul(phGpu); // Make Q uniform according to the paper

        // if rows < cols:
        //     q.t_()

        if (rows < cols) {
            qGpu.t_(); // (rows, cols) -> (cols, rows)
        }

        // with torch.no_grad():
        //     tensor.view_as(q).copy_(q)
        //     tensor.mul_(gain)

        Tensor q = qGpu.to(originalDevice, torch.ScalarType.Float);

        try (NoGradGuard noGradGuard = new NoGradGuard()) {
            tensor.view_as(q).copy_(q); // Copy the orthogonal matrix to the tensor
            tensor.mul_(new Scalar(std)); // Scale the tensor by the standard deviation
        }
    }

    static LinearImpl createLinearLayer(long inputsDim, long outputDims, double std, Device device) {
        LinearImpl layer = new LinearImpl(inputsDim, outputDims);

        orthogonal_(layer.weight(), std, device);
        torch.constant_(layer.bias(), new Scalar(0.0f));
        return layer;
    }

    static LinearImpl createLinearLayer(long inputsDim, long outputDims, Device device) {
        double root2 = Math.sqrt(2.0);
        return createLinearLayer(inputsDim, outputDims, root2, device);
    }

    // Tensor observation is [B, OBSERVATION_SPACE_SIZE]
    public States getStates(Tensor observationTensor, LSTMState lstmState, Tensor done) {
        PointerScope scope = new PointerScope();
        /*
        height = x[:, -self.local_heightmap_conv[0][0].in_channels * self.local_heightmap_conv[0][0].kernel_size[0] ** 2 :].reshape(-1, 1, 7, 7)
        height_features = self.local_heightmap_conv(height).reshape(-1, 16)

        remaining_obs = x[:, :-self.local_heightmap_conv[0][0].in_channels * self.local_heightmap_conv[0][0].kernel_size[0] ** 2]
        observation = torch.cat([remaining_obs, height_features], dim=1)
        hidden = self.network(observation)
         */
        if (observationTensor.dim() == 1) {
            observationTensor = observationTensor.unsqueeze(0); // (OBSERVATION_SPACE_SIZE,) -> (1, OBSERVATION_SPACE_SIZE)
        }

        Tensor localHeightMap = observationTensor.narrow(1, Observation.OFFSET_LOCAL_HEIGHT_MAP, Observation.SIZE_LOCAL_HEIGHT_MAP); // (B, OBSERVATION_SPACE_SIZE) -> (B, 49)
        Tensor localHeightMapReshaped = localHeightMap.reshape(-1, 1, 7, 7); // (B, 49) -> (B, 1, 7, 7)
        Tensor heightFeatures = this.localHeightmapConv.forward(localHeightMapReshaped); // size (B, 16)

        Tensor remainingObs = observationTensor.narrow(1, 0, Observation.OFFSET_LOCAL_HEIGHT_MAP); // (B, OBSERVATION_SPACE_SIZE) -> (B, OBSERVATION_SPACE_SIZE - 49)
        TensorVector combinedObsTensors = new TensorVector(remainingObs, heightFeatures);
        Tensor combinedObs = torch.cat(combinedObsTensors, 1); // (B, OBSERVATION_SPACE_SIZE - 49) + (B, 16) -> (B, OBSERVATION_SPACE_SIZE - 49 + 16)
        combinedObsTensors.close();

        Tensor hidden = this.network.forward(combinedObs); // size (B, 64)
        localHeightMap.close();
        localHeightMapReshaped.close();
        heightFeatures.close();
        remainingObs.close();
        combinedObs.close();

        /*
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
         */

        Tensor startingHiddenState = lstmState.hiddenState();
        long batchSize = startingHiddenState.size(1); // batchSize
        hidden = hidden.reshape(-1, batchSize, this.lstm.options().input_size().get()); // (batchSize, 64) -> (1, batchSize, input_size)
        done = done.reshape(-1, batchSize); // (batchSize,) -> (1, batchSize)

        long seqLen = hidden.size(0); // seqLen = B

        TensorVector newHidden = new TensorVector();

        Tensor hiddenState = startingHiddenState.clone();
        Tensor cellState = lstmState.cellState().clone();

        startingHiddenState.close();

        Tensor ones = torch.ones_like(done, new TensorOptions(device), null); // size (B, batchSize)
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
            Tensor h1 = hiddenList.get(i);
            Tensor h = h1.unsqueeze(0); // (batchSize, input_size) -> (1, batchSize, input_size)
            Tensor d1 = oneSubDoneList.get(i);
            Tensor d = d1
                .view(1, -1, 1); // (batchSize,) -> (1, batchSize, 1)

            Tensor newHiddenState = hiddenState.mul(d); // Hidden state size (1, batchSize, hidden_size)
            Tensor newCellState = cellState.mul(d); // Cell state size (1, batchSize, hidden_size)

            hiddenState.close();
            cellState.close();

            T_TensorTensor_T inputState = new T_TensorTensor_T(newHiddenState, newCellState);
            T_TensorT_TensorTensor_T_T hNew_LSTMState = this.lstm.forward(h, inputState);

            newHiddenState.close();
            newCellState.close();
            h1.close();
            h.close();
            d1.close();
            d.close();
            inputState.close();

            newHidden.push_back(hNew_LSTMState.get0());

            T_TensorTensor_T outState = hNew_LSTMState.get1();
            hiddenState = outState.get0();
            cellState = outState.get1();
        }

        /*
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
         */
        Tensor cat = torch.cat(newHidden);
        Tensor newHiddenTensor = torch.flatten(cat, 0, 1); // (seqLen, batchSize, input_size) -> (seqLen * batchSize, input_size)
        cat.close();
        for (int i = 0; i < newHidden.size(); i++) {
            newHidden.get(i).close();
        }
        newHidden.close();

        hiddenState = hiddenState.clone();
        hiddenState.retainReference();
        cellState = cellState.clone();
        cellState.retainReference();
        newHiddenTensor.retainReference();

        scope.close();

        /*
        return new_hidden, lstm_state
         */
        LSTMState lstmState1 = new LSTMState(hiddenState, cellState);
        return new States(newHiddenTensor, lstmState1);
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
        Tensor forward = this.critic.forward(hidden);
        states.close();
        return forward;
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

        PointerScope scope = new PointerScope();

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

        Tensor yMeanForward = this.yawMean.forward(hidden);
        Tensor yawMean = yMeanForward.squeeze(-1); // (batch, 1) -> (batch,)
        Tensor yawStd = torch.exp(torch.clamp(this.yawLogSTD, SCALAR_n5, SCALAR_2));
        Normal yawDist = new Normal(yawMean, yawStd);
        yMeanForward.close();

        Tensor pitchMeanForward = this.pitchMean.forward(hidden);
        Tensor pitchMean = pitchMeanForward.squeeze(-1); // (batch, 1) -> (batch,)
        Tensor pitchStd = torch.exp(torch.clamp(this.pitchLogSTD, SCALAR_n5, SCALAR_2));
        Normal pitchDist = new Normal(pitchMean, pitchStd);
        pitchMeanForward.close();

        // Jump

        Tensor jumpKeyLogits = this.actorJumpKey.forward(hidden);
        Categorical jumpKeyProbs = new Categorical(jumpKeyLogits);

        // Sprint & Sneak

        Tensor sprintSneakKeysLogits = this.actorSprintSneakKeys.forward(hidden);
        Categorical sprintSneakKeysProbs = new Categorical(sprintSneakKeysLogits);

        // Attack & Use Item

        Tensor attackUseItemLogits = this.actorAttackUseItem.forward(hidden);
        Categorical attackUseItemProbs = new Categorical(attackUseItemLogits);

        // Slot 0 or 1
        Tensor slotLogits = this.actorSlot.forward(hidden);
        Categorical slotProbs = new Categorical(slotLogits);

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
        Tensor slotAction;

        if (action == null) {
            forwardMoveKeysAction = forwardMoveKeysProbs.sample(); // LongTensor
            strafingMoveKeysAction = strafingMoveKeysProbs.sample(); // LongTensor
            yawAction = yawDist.sample(); // FloatTensor
            pitchAction = pitchDist.sample(); // FloatTensor
            jumpKeyAction = jumpKeyProbs.sample(); // LongTensor
            sprintSneakKeysAction = sprintSneakKeysProbs.sample(); // LongTensor
            attackUseItemAction = attackUseItemProbs.sample(); // LongTensor
            slotAction = slotProbs.sample(); // LongTensor

            // ! THIS MUST MATCH THE ORDER IN Action CLASS
            Tensor jumpFloat = jumpKeyAction.to(torch.ScalarType.Float);
            Tensor sprintSneakFlaot = sprintSneakKeysAction.to(torch.ScalarType.Float);
            Tensor forwardFloat = forwardMoveKeysAction.to(torch.ScalarType.Float);
            Tensor strafingFloat = strafingMoveKeysAction.to(torch.ScalarType.Float);
            Tensor useFloat = attackUseItemAction.to(torch.ScalarType.Float);
            Tensor slotFloat = slotAction.to(torch.ScalarType.Float);
            TensorVector tensorVector = new TensorVector(
                jumpFloat,
                sprintSneakFlaot,
                yawAction,
                pitchAction,
                forwardFloat,
                strafingFloat,
                useFloat,
                slotFloat
            );
            action = torch.stack(tensorVector, 1); // (numEnvs,) x8 -> (numEnvs, 8)
            jumpFloat.close();
            sprintSneakFlaot.close();
            forwardFloat.close();
            strafingFloat.close();
            useFloat.close();
            slotFloat.close();
            tensorVector.close();
        } else {
            // ! THIS MUST MATCH THE ORDER IN Action CLASS

            Tensor jumpKeyNarrow = action.narrow(1, 0, 1); // (numEnvs, 8) -> (numEnvs, 1)
            Tensor jumpKeySqueeze = jumpKeyNarrow.squeeze(1); // (numEnvs, 1) -> (numEnvs,)
            jumpKeyAction = jumpKeySqueeze.to(torch.ScalarType.Long);
            jumpKeyNarrow.close();
            jumpKeySqueeze.close();

            Tensor sprintSneakKeysNarrow = action.narrow(1, 1, 1); // (numEnvs, 8) -> (numEnvs, 1)
            Tensor sprintSneakKeysSqueeze = sprintSneakKeysNarrow.squeeze(1); // (numEnvs, 1) -> (numEnvs,)
            sprintSneakKeysAction = sprintSneakKeysSqueeze.to(torch.ScalarType.Long);
            sprintSneakKeysNarrow.close();
            sprintSneakKeysSqueeze.close();

            Tensor yawNarrow = action.narrow(1, 2, 1); // (numEnvs, 8) -> (numEnvs, 1)
            Tensor yawSqueeze = yawNarrow.squeeze(1); // (numEnvs, 1) -> (numEnvs,)
            yawAction = yawSqueeze.to(yawMean.dtype());
            yawNarrow.close();
            yawSqueeze.close();

            Tensor pitchNarrow = action.narrow(1, 3, 1); // (numEnvs, 8) -> (numEnvs, 1)
            Tensor pitchSqueeze = pitchNarrow.squeeze(1); // (numEnvs, 1) -> (numEnvs,)
            pitchAction = pitchSqueeze.to(pitchMean.dtype());
            pitchNarrow.close();
            pitchSqueeze.close();

            Tensor forwardMoveKeysNarrow = action.narrow(1, 4, 1); // (numEnvs, 8) -> (numEnvs, 1)
            Tensor forwardMoveKeysSqueeze = forwardMoveKeysNarrow.squeeze(1); // (numEnvs, 1) -> (numEnvs,)
            forwardMoveKeysAction = forwardMoveKeysSqueeze.to(torch.ScalarType.Long);
            forwardMoveKeysNarrow.close();
            forwardMoveKeysSqueeze.close();

            Tensor strafingMoveKeysNarrow = action.narrow(1, 5, 1); // (numEnvs, 8) -> (numEnvs, 1)
            Tensor strafingMoveKeysSqueeze = strafingMoveKeysNarrow.squeeze(1); // (numEnvs, 1) -> (numEnvs,)
            strafingMoveKeysAction = strafingMoveKeysSqueeze.to(torch.ScalarType.Long);
            strafingMoveKeysNarrow.close();
            strafingMoveKeysSqueeze.close();

            Tensor attackUseItemNarrow = action.narrow(1, 6, 1); // (numEnvs, 8) -> (numEnvs, 1)
            Tensor attackUseItemSqueeze = attackUseItemNarrow.squeeze(1); // (numEnvs, 1) -> (numEnvs,)
            attackUseItemAction = attackUseItemSqueeze.to(torch.ScalarType.Long);
            attackUseItemNarrow.close();
            attackUseItemSqueeze.close();

            Tensor slotNarrow = action.narrow(1, 7, 1); // (numEnvs, 8) -> (numEnvs, 1)
            Tensor slotSqueeze = slotNarrow.squeeze(1); // (numEnvs, 1) -> (numEnvs,)
            slotAction = slotSqueeze.to(torch.ScalarType.Long);
            slotNarrow.close();
            slotSqueeze.close();
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
        Tensor slotLogProbs = slotProbs.logProb(slotAction);

        Tensor totalLogProbs = forwardMoveKeysLogProbs
            .add(strafingMoveKeysLogProbs)
            .add_(yawLogProbs)
            .add_(pitchLogProbs)
            .add_(jumpKeyLogProbs)
            .add_(sprintSneakKeysLogProbs)
            .add_(attackUseItemLogProbs)
            .add_(slotLogProbs);
        forwardMoveKeysLogProbs.close();
        strafingMoveKeysLogProbs.close();
        yawLogProbs.close();
        pitchLogProbs.close();
        jumpKeyLogProbs.close();
        sprintSneakKeysLogProbs.close();
        attackUseItemLogProbs.close();
        slotLogProbs.close();

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
        Tensor slotEntropy = slotProbs.entropy();

        Tensor totalEntropy = forwardMoveKeysEntropy
            .add(strafingMoveKeysEntropy)
            .add_(yawEntropy)
            .add_(pitchEntropy)
            .add_(jumpKeyEntropy)
            .add_(sprintSneakKeysEntropy)
            .add_(attackUseItemEntropy)
            .add_(slotEntropy);
        forwardMoveKeysEntropy.close();
        strafingMoveKeysEntropy.close();
        yawEntropy.close();
        pitchEntropy.close();
        jumpKeyEntropy.close();
        sprintSneakKeysEntropy.close();
        attackUseItemEntropy.close();
        slotEntropy.close();

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

        forwardMoveKeysProbs.close();
        strafingMoveKeysProbs.close();
        yawDist.close();
        pitchDist.close();
        jumpKeyProbs.close();
        sprintSneakKeysProbs.close();
        attackUseItemProbs.close();
        slotProbs.close();
        forwardMoveKeysAction.close();
        strafingMoveKeysAction.close();
        yawAction.close();
        pitchAction.close();
        jumpKeyAction.close();
        sprintSneakKeysAction.close();
        attackUseItemAction.close();
        slotAction.close();

        action.retainReference();
        totalLogProbs.retainReference();
        totalEntropy.retainReference();
        value.retainReference();

        scope.close();

        // states.close();
        states.newHiddenTensor.close();
        hidden.close();

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
        File folder = new File("training/" + args.experimentId + "/model_files/");
        folder.mkdirs();
        this.save(outputArchive);
        outputArchive.save_to("training/" + args.experimentId + "/model_files/" + iteration + ".pt");
        outputArchive.close();
    }

    public void loadCheckpoint(Integer iteration) {
        if (iteration == null) {
            return;
        }
        InputArchive inputArchive = new InputArchive();
        inputArchive.load_from("training/" + args.experimentId + "/model_files/" + iteration + ".pt");
        LOGGER.info("Loading MinecraftRL checkpoint from iteration " + iteration);
        this.load(inputArchive);
        inputArchive.close();

    }


    public void copyParametersFrom(MinecraftRL other) {
        AutogradState.get_tls_state().set_grad_mode(false);

        try (TensorVector parameters = this.parameters(); TensorVector otherPrams = other.parameters()) {
            if (parameters.size() != otherPrams.size()) {
                throw new IllegalArgumentException("Models have different number of parameters");
            }

            for (long i = 0; i < parameters.size(); i++) {
                parameters.get(i).copy_(otherPrams.get(i));
            }
        } finally {
            AutogradState.get_tls_state().set_grad_mode(true);
        }
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
    public record States(Tensor newHiddenTensor, MinecraftRL.LSTMState lstmState) implements AutoCloseable {
        @Override
        public void close() {
            newHiddenTensor.close();
            lstmState.close();
        }
    }

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
                                 MinecraftRL.LSTMState lstmState) implements AutoCloseable {
        @Override
        public void close() {
            action.close();
            totalLogProbs.close();
            totalEntropy.close();
            value.close();
            lstmState.close();
        }
    }
}
