package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.environment.MinecraftEnvironment;
import com.tenbitmelon.machinelearningplayer.environment.Observation;
import com.tenbitmelon.machinelearningplayer.util.distributions.Categorical;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.openblas.global.openblas;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;
import org.bytedeco.pytorch.global.torch;

import javax.annotation.Nullable;

public class MinecraftRL extends Module {

    public final long otherFeaturesDim = 128;
    public final long sharedOutDim = 256;
    public final long lstmHiddenSize = 128;
    private final SequentialImpl voxelCNN;
    private final SequentialImpl otherInputsProcessor;
    private final SequentialImpl sharedNetwork;
    private final LSTMImpl lstm;
    private final SequentialImpl critic;
    private final LinearImpl actorLookChangeMean;
    private final Tensor actorLookChangeLogSTD;
    private final LinearImpl actorSprintKey;
    private final LinearImpl actorSneakKey;
    private final LinearImpl actorJumpKey;
    private final LinearImpl actorForwardMoveKeys;
    private final LinearImpl actorStrafingMoveKeys;

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

        /*
            Trying to simplify this
            # Input: (B, 1, X, Z, Y)
            layer_init(nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))), # Output Size: (B, 16, X, Z, Y)
            nn.ReLU(),
            layer_init(nn.Conv3d(16, 32, kernel_size=(3, 3, MinecraftEnvironment.GRID_SIZE_Y), stride=(1, 1, 1), padding=(1, 1, 0))), # Output Size: (B, 32, X, Z, Y)
            nn.ReLU(),
            nn.Flatten(), # Flatten the output to (B, 32 * X * Z * Y)
         */


        final Conv3dImpl conv1 = createConv3d(1, 16, 3, 3, 3, 1, 1, 1, 1, 1, 1);
        final ReLUImpl relu = new ReLUImpl();
        final Conv3dImpl conv2 = createConv3d(16, 32, 3, 3, MinecraftEnvironment.GRID_SIZE_Y, 1, 1, 1, 1, 1, 0);
        final ReLUImpl relu2 = new ReLUImpl();
        // final Conv3dImpl conv3 = createConv3d(32, 64, 3, 3, 1, 2, 2, 1, 1, 1, 0);
        // final ReLUImpl relu3 = new ReLUImpl();
        final FlattenImpl flatten = new FlattenImpl();

        final SequentialImpl sequential = new SequentialImpl();
        sequential.push_back("conv1", conv1);
        sequential.push_back("relu", relu);
        sequential.push_back("conv2", conv2);
        sequential.push_back("relu2", relu2);
        // sequential.push_back("conv3", conv3);
        // sequential.push_back("relu3", relu3);
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
        // LinearImpl otherLinear2 = createLinearLayer(64, 64);
        // ReLUImpl otherReLU2 = new ReLUImpl();
        // LinearImpl otherLinear3 = createLinearLayer(64, 64);
        // ReLUImpl otherReLU3 = new ReLUImpl();
        LinearImpl otherLinear4 = createLinearLayer(64, otherFeaturesDim);
        ReLUImpl otherReLU4 = new ReLUImpl();

        SequentialImpl otherInputsProcessor = new SequentialImpl();
        otherInputsProcessor.push_back("other_linear1", otherLinear1);
        otherInputsProcessor.push_back("other_relu1", otherReLU1);
        // otherInputsProcessor.push_back("other_linear2", otherLinear2);
        // otherInputsProcessor.push_back("other_relu2", otherReLU2);
        // otherInputsProcessor.push_back("other_linear3", otherLinear3);
        // otherInputsProcessor.push_back("other_relu3", otherReLU3);
        otherInputsProcessor.push_back("other_linear4", otherLinear4);
        otherInputsProcessor.push_back("other_relu4", otherReLU4);

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

        long combinedFeaturesDim = /** cnnOutDim + */otherFeaturesDim;

        LinearImpl sharedLinear1 = createLinearLayer(combinedFeaturesDim, 256);
        ReLUImpl sharedReLU1 = new ReLUImpl();
        // LinearImpl sharedLinear2 = createLinearLayer(256, 256);
        // ReLUImpl sharedReLU2 = new ReLUImpl();
        // LinearImpl sharedLinear3 = createLinearLayer(256, 256);
        // ReLUImpl sharedReLU3 = new ReLUImpl();
        LinearImpl sharedLinear4 = createLinearLayer(256, sharedOutDim);
        ReLUImpl sharedReLU4 = new ReLUImpl();

        SequentialImpl sharedNetwork = new SequentialImpl();
        sharedNetwork.push_back("shared_linear1", sharedLinear1);
        sharedNetwork.push_back("shared_relu1", sharedReLU1);
        // sharedNetwork.push_back("shared_linear2", sharedLinear2);
        // sharedNetwork.push_back("shared_relu2", sharedReLU2);
        // sharedNetwork.push_back("shared_linear3", sharedLinear3);
        // sharedNetwork.push_back("shared_relu3", sharedReLU3);
        sharedNetwork.push_back("shared_linear4", sharedLinear4);
        sharedNetwork.push_back("shared_relu4", sharedReLU4);

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
                orthogonal_(param, Math.sqrt(2.0));
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
        self.actor_froward_move_keys = layer_init(
            nn.Linear(actor_input_dim, 3), std=0.01
        )  # Logits for W,S, & none
        self.actor_strafing_move_keys = layer_init(
            nn.Linear(actor_input_dim, 3), std=0.01
        )  # Logits for A,D, & none

        # Critic Head
        self.critic = layer_init(nn.Linear(actor_input_dim, 1), std=1)
         */

        long actorInputDim = lstmHiddenSize;
        LinearImpl actorLookChangeMean = createLinearLayer(actorInputDim, 2, 0.01);
        // Tensor actorLookChangeLogSTD = torch.zeros(1, 2); // This would lead to max motion 68% of time
        Tensor actorLookChangeLogSTD = torch.ones(1, 2).mul(new Scalar(-1.0f)); // Start with small stddev

        LinearImpl actorSprintKey = createLinearLayer(actorInputDim, 1, 0.01);
        LinearImpl actorSneakKey = createLinearLayer(actorInputDim, 1, 0.01);
        LinearImpl actorJumpKey = createLinearLayer(actorInputDim, 1, 0.01);
        LinearImpl actorForwardMoveKeys = createLinearLayer(actorInputDim, 3, 0.01);
        LinearImpl actorStrafingMoveKeys = createLinearLayer(actorInputDim, 3, 0.01);


        // LinearImpl critic = createLinearLayer(actorInputDim, 1, 1.0);
        LinearImpl criticLinear1 = createLinearLayer(actorInputDim, 256, 1.0);
        ReLUImpl criticReLU1 = new ReLUImpl();
        LinearImpl criticLinear2 = createLinearLayer(256, 1, 1.0);

        SequentialImpl critic = new SequentialImpl();
        critic.push_back("critic_linear1", criticLinear1);
        critic.push_back("critic_relu1", criticReLU1);
        critic.push_back("critic_linear2", criticLinear2);

        register_module("actor_look_change_mean", actorLookChangeMean);
        register_parameter("actor_look_change_logstd", actorLookChangeLogSTD);
        register_module("actor_sprint_key", actorSprintKey);
        register_module("actor_sneak_key", actorSneakKey);
        register_module("actor_jump_key", actorJumpKey);
        register_module("actor_forward_move_keys", actorForwardMoveKeys);
        register_module("actor_strafing_move_keys", actorStrafingMoveKeys);
        register_module("critic", critic);

        this.actorLookChangeMean = actorLookChangeMean;
        this.actorLookChangeLogSTD = actorLookChangeLogSTD;
        this.actorSprintKey = actorSprintKey;
        this.actorSneakKey = actorSneakKey;
        this.actorJumpKey = actorJumpKey;
        this.actorForwardMoveKeys = actorForwardMoveKeys;
        this.actorStrafingMoveKeys = actorStrafingMoveKeys;
        this.critic = critic;

        // LOGGER.debug("MinecraftRL model initialized successfully.");
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
            flattened = flattened.t(); // Transpose the tensor if rows < cols
            long temp = rows;
            rows = cols;
            cols = temp;
        }

        // # Compute the qr factorization
        // q, r = torch.linalg.qr(flattened)
        // # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        // d = torch.diag(r, 0)
        // ph = d.sign()
        // q *= ph


        T_TensorTensor_T qr = linalg_qr(flattened);

        Tensor q = qr.get0();
        Tensor r = qr.get1();
        Tensor d = torch.diag(r, 0);
        Tensor ph = d.sign();
        q = q.mul(ph); // Make Q uniform according to the paper

        // if rows < cols:
        //     q.t_()

        if (rows < cols) {
            q = q.t(); // Transpose
        }

        // with torch.no_grad():
        //     tensor.view_as(q).copy_(q)
        //     tensor.mul_(gain)

        AutogradState.get_tls_state().set_grad_mode(false);

        tensor.view_as(q).copy_(q); // Copy the orthogonal matrix to the tensor
        tensor.mul_(new Scalar(std)); // Scale the tensor by the standard deviation

        AutogradState.get_tls_state().set_grad_mode(true);

        // return tensor
    }

    static T_TensorTensor_T linalg_qr(Tensor tensor) {
        // Ensure tensor is contiguous and float precision
        Tensor a = tensor.contiguous();

        int m = (int) a.size(0);
        int n = (int) a.size(1);
        int k = Math.min(m, n);

        // Get data pointer from PyTorch tensor
        FloatPointer aPtr = new FloatPointer(a.data_ptr());
        FloatPointer tau = new FloatPointer(k);


        // Perform QR decomposition
        int info = openblas.LAPACKE_sgeqrf(openblas.LAPACK_ROW_MAJOR, m, n, aPtr, n, tau);
        if (info != 0) {
            throw new RuntimeException("LAPACKE_sgeqrf failed with info: " + info);
        }

        // Extract R matrix (upper triangular part)
        Tensor r = torch.zeros(new long[]{k, n}, new TensorOptions(torch.ScalarType.Float));
        FloatPointer rPtr = new FloatPointer(r.data_ptr());

        // Copy upper triangular part to R
        for (int i = 0; i < k; i++) {
            for (int j = i; j < n; j++) {
                rPtr.put(i * n + j, aPtr.get(i * n + j));
            }
        }
        // Generate orthogonal matrix Q
        Tensor q = a.clone(); // Start with the modified input
        FloatPointer qPtr = new FloatPointer(q.data_ptr());

        info = openblas.LAPACKE_sorgqr(openblas.LAPACK_ROW_MAJOR, m, k, k, qPtr, k, tau);
        if (info != 0) {
            throw new RuntimeException("Q generation failed with info: " + info);
        }

        // Resize Q to proper dimensions if needed
        if (k < n) {
            q = q.narrow(1, 0, k); // Take only first k columns
        }

        return new T_TensorTensor_T(q, r);

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

        orthogonal_(layer.weight(), Math.sqrt(2.0));
        torch.constant_(layer.bias(), new Scalar(0.0f));
        return layer;
    }

    // Tensor observation is [B, OBSERVATION_SPACE_SIZE]
    public States getStates(Tensor observation, LSTMState lstmState, Tensor done) {
        // PointerScope scope = new PointerScope();
        // LOGGER.debug("--- Entering getStates ---");
        // LOGGER.debug("Initial observation shape: {}", observation.shape());
        // LOGGER.debug("Initial observation: {}", tensorString(observation));
        // LOGGER.debug("Initial LSTM hidden state shape: {}", lstmState.hiddenState().shape());
        // LOGGER.debug("Initial LSTM hidden state: {}", tensorString(lstmState.hiddenState()));
        // LOGGER.debug("Initial LSTM cell state shape: {}", lstmState.cellState().shape());
        // LOGGER.debug("Initial LSTM cell state: {}", tensorString(lstmState.cellState()));
        // LOGGER.debug("Initial done shape: {}", done.shape());
        // LOGGER.debug("Initial done values: {}", tensorString(done));

        /*
        voxel_data = x[:, : MinecraftDummyEnv.GRID_VOLUME]
        other_inputs = x[:, MinecraftDummyEnv.GRID_VOLUME :]
        */
        Tensor voxelData = observation.narrow(1, 0, MinecraftEnvironment.GRID_VOLUME);
        Tensor otherInputs = observation.narrow(1, MinecraftEnvironment.GRID_VOLUME, Observation.OBSERVATION_SPACE_SIZE - MinecraftEnvironment.GRID_VOLUME);


        // LOGGER.debug("Split voxel data shape: {}", voxelData.shape());
        // LOGGER.debug("Split other inputs shape: {}", otherInputs.shape());

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

        // LOGGER.debug("Reshaped voxel data shape (for CNN): {}", voxelData.shape());

        /// Tensor voxelFeatures = this.voxelCNN.forward(voxelData);
        Tensor otherFeatures = this.otherInputsProcessor.forward(otherInputs);
        /// TensorVector tensorsToCombine = new TensorVector(voxelFeatures, otherFeatures);
        /// Tensor combinedFeatures = torch.cat(tensorsToCombine, 1);
        /// Tensor sharedFeatures = this.sharedNetwork.forward(combinedFeatures);
        Tensor sharedFeatures = this.sharedNetwork.forward(otherFeatures);

        // LOGGER.debug("Voxel features shape (after CNN): {}", voxelFeatures.shape());
        // LOGGER.debug("Other features shape (after processing): {}", otherFeatures.shape());
        // LOGGER.debug("Combined features shape (after cat): {}", combinedFeatures.shape());
        // LOGGER.debug("Shared features shape (after shared network): {}", sharedFeatures.shape());

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

        // // LOGGER.debug("Shared features shape: {}", sharedFeatures.shape());

        Tensor startingHiddenState = lstmState.hiddenState();
        long batchSize = startingHiddenState.size(1); // batchSize
        Tensor hidden = sharedFeatures.reshape(-1, batchSize, this.lstm.options().input_size().get()); // size (B, batchSize, input_size)
        done = done.reshape(-1, batchSize); // size (B, batchSize)

        long seqLen = hidden.size(0); // seqLen = B

        TensorVector newHidden = new TensorVector();
        // Tensor newHidden = torch.zeros(new long[]{seqLen, batchSize, this.lstmHiddenSize}, new TensorOptions(TrainingManager.device));

        Tensor hiddenState = startingHiddenState.clone();
        Tensor cellState = lstmState.cellState().clone();

        startingHiddenState.close();

        Tensor ones = torch.ones_like(done, new TensorOptions(TrainingManager.device), null); // size (B, batchSize)
        Tensor oneSubDone = ones.sub_(done); // size (B, batchSize)

        TensorVector hiddenList = torch.unbind(hidden, 0);
        TensorVector oneSubDoneList = torch.unbind(oneSubDone, 0);

        for (int i = 0; i < seqLen; i++) {
            Tensor h = hiddenList.get(i).unsqueeze(0); // size (1, batchSize, input_size)
            Tensor d = oneSubDoneList.get(i) // size (batchSize,)
                .view(1, -1, 1); // Reshape to (1, batchSize, 1)

            hiddenState.mul_(d); // Hidden state size (1, batchSize, hidden_size)
            cellState.mul_(d); // Cell state size (1, batchSize, hidden_size)

            T_TensorTensor_T inputState = new T_TensorTensor_T(hiddenState, cellState);
            T_TensorT_TensorTensor_T_T hNew_LSTMState = this.lstm.forward(h, inputState);

            hiddenState.close();
            cellState.close();

            newHidden.push_back(hNew_LSTMState.get0());
            // newHidden.index_copy_(0, torch.tensor(i), hNew_LSTMState.get0());
            // newHidden.narrow(0, i, 1).copy_(hNew_LSTMState.get0());
            // lstmState = new LSTMState(hNew_LSTMState.get1().get0(), hNew_LSTMState.get1().get1());
            // lstmState.set(hNew_LSTMState.get1());
            T_TensorTensor_T outState = hNew_LSTMState.get1();
            hiddenState = outState.get0();
            cellState = outState.get1();
            // hNew_LSTMState.close();
        }
        Tensor newHiddenTensor = torch.flatten(torch.cat(newHidden), 0, 1);
        // Tensor newHiddenTensor = torch.flatten(newHidden, 0, 1);

        /*
        return new_hidden, lstm_state
         */

        // scope.close();

        return new States(newHiddenTensor, lstmState);
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
        return self.critic(hidden)
         */
        // LOGGER.debug("--- Entering getValue ---");
        States states = this.getStates(observation, lstmState, done);
        Tensor hidden = states.newHiddenTensor;
        // LOGGER.debug("Hidden tensor shape for critic: {}", hidden.shape());
        // LOGGER.debug("Output value shape from critic: {}", value.shape());
        // LOGGER.debug("--- Exiting getValue ---");
        return this.critic.forward(hidden);
    }

    public ActionAndValue getActionAndValue(Tensor nextObs, LSTMState nextLstmState, Tensor nextDone) {
        return getActionAndValue(nextObs, nextLstmState, nextDone, null);
    }

    public ActionAndValue getActionAndValue(Tensor observation, LSTMState lstmState, Tensor done, @Nullable Tensor action) {
        // LOGGER.debug("--- Entering getActionAndValue ---");

        States states = this.getStates(observation, lstmState, done);
        Tensor sharedFeatures = states.newHiddenTensor;
        lstmState = states.lstmState;

        // LOGGER.debug("Shared features shape for actor/critic heads: {}", sharedFeatures.shape());

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
        forward_move_logits = self.actor_forward_move_keys(actor_critic_features)  # Shape: (B, 3)
        forward_move_dist = torch.distributions.Categorical(logits=forward_move_logits)
        strafing_move_logits = self.actor_strafing_move_keys(actor_critic_features)  # Shape: (B, 3)
        strafing_move_dist = torch.distributions.Categorical(logits=strafing_move_logits)
         */

        // Tensor lookMeans = torch.tanh(this.actorLookChangeMean.forward(sharedFeatures));
        /// Tensor lookMeans = this.actorLookChangeMean.forward(sharedFeatures);
        /// Tensor lookStd = torch.exp(this.actorLookChangeLogSTD.expand(lookMeans.shape()));
        /// Normal lookDist = new Normal(lookMeans, lookStd);
        // LOGGER.debug("Look means shape: {}, Look std shape: {}", lookMeans.shape(), lookStd.shape());

        /// Tensor sprintLogits = this.actorSprintKey.forward(sharedFeatures);
        /// Bernoulli sprintDist = new Bernoulli(sprintLogits);
        // LOGGER.debug("Sprint logits shape: {}", sprintLogits.shape());

        /// Tensor sneakLogits = this.actorSneakKey.forward(sharedFeatures);
        /// Bernoulli sneakDist = new Bernoulli(sneakLogits);
        // LOGGER.debug("Sneak logits shape: {}", sneakLogits.shape());

        /// Tensor jumpLogits = this.actorJumpKey.forward(sharedFeatures);
        /// Bernoulli jumpDist = new Bernoulli(jumpLogits);
        // LOGGER.debug("Jump logits shape: {}", jumpLogits.shape());

        Tensor forwardMoveLogits = this.actorForwardMoveKeys.forward(sharedFeatures); // Shape: (B, 3)
        Tensor strafingMoveLogits = this.actorStrafingMoveKeys.forward(sharedFeatures); // Shape: (B, 3)
        Categorical forwardMoveDist = new Categorical(forwardMoveLogits);
        Categorical strafingMoveDist = new Categorical(strafingMoveLogits);

        // LOGGER.debug("Move logits shape: {}", moveLogits.shape());

        // LOGGER.debug("Sprint logits values: {}", tensorString(sprintLogits));
        // LOGGER.debug("Sneak logits values: {}", tensorString(sneakLogits));
        // LOGGER.debug("Jump logits values: {}", tensorString(jumpLogits));
        // LOGGER.debug("Move logits values: {}", tensorString(moveLogits));

        /*
        if action is None:
            look_sample = look_dist.sample()  # (B, 2) e.g., pitch, yaw
            sprint_sample = sprint_dist.sample()  # (B, 1)
            sneak_sample = sneak_dist.sample()  # (B, 1)
            jump_sample = jump_dist.sample()  # (B, 1)
            forward_move_sample = forward_move_dist.sample()  # (B, 1)
            strafing_move_sample = strafing_move_dist.sample()  # (B, 1)

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
            // LOGGER.debug("Action is null, sampling new actions.");
            /// Tensor lookSample = lookDist.sample(); // Correctly sampling continuous values
            /// Tensor sprintSample = sprintDist.sample(); // Correctly sampling discrete values (0 or 1)
            /// Tensor sneakSample = sneakDist.sample();
            /// Tensor jumpSample = jumpDist.sample();
            Tensor forwardMoveSample = forwardMoveDist.sample();
            Tensor strafingMoveSample = strafingMoveDist.sample();
            // LOGGER.debug("Sampled actions shapes: look={}, sprint={}, sneak={}, jump={}, move={}",
            //     lookSample.shape(), sprintSample.shape(), sneakSample.shape(), jumpSample.shape(), moveSample.shape());

            // LOGGER.debug("Look sample: {}", tensorString(lookSample));
            // LOGGER.debug("Sprint sample: {}", tensorString(sprintSample));
            // LOGGER.debug("Sneak sample: {}", tensorString(sneakSample));
            // LOGGER.debug("Jump sample: {}", tensorString(jumpSample));
            // LOGGER.debug("Move sample: {}", tensorString(moveSample));

            // Concatenate along the last dimension -> flat per-sample vector
            // Resulting shape: (B, 7)
            // Jump(0, 1), Sprint(1, 1), Sneak(2, 1), Look(3, 2), ForwardMove(5, 1), StrafingMove(6, 1)
            action = torch.cat(new TensorVector(
                /// jumpSample,
                /// sprintSample,
                /// sneakSample,
                /// lookSample,
                forwardMoveSample,
                strafingMoveSample
            ), -1).to(torch.ScalarType.Float);
            // LOGGER.debug("Concatenated action: {}", tensorString(action));
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
        forward_move_action = action[..., 5:6].float()
        log_probs_forward_move = forward_move_dist.log_prob(forward_move_action).sum(dim=-1)
        strafing_move_action = action[..., 6:7].float()
        log_probs_strafing_move = strafing_move_dist.log_prob(strafing_move_action).sum(dim=-1)


        total_log_probs = (
            log_probs_look
            + log_probs_sprint
            + log_probs_sneak
            + log_probs_jump
            + log_probs_forward_move
            + log_probs_strafing_move
        )
         */

        /// Tensor jumpAction = action.narrow(-1, 0, 1);
        /// Tensor logProbsJump = jumpDist.logProb(jumpAction).squeeze(-1);
        /// Tensor sprintAction = action.narrow(-1, 1, 1);
        /// Tensor logProbsSprint = sprintDist.logProb(sprintAction).squeeze(-1);
        /// Tensor sneakAction = action.narrow(-1, 2, 1);
        /// Tensor logProbsSneak = sneakDist.logProb(sneakAction).squeeze(-1);
        /// Tensor lookAction = action.narrow(-1, 3, 2);
        /// Tensor logProbsLook = lookDist.logProb(lookAction).sum(-1);
        /// Tensor forwardMoveAction = action.narrow(-1, 5, 1);
        /// Tensor logProbsForwardMove = forwardMoveDist.logProb(forwardMoveAction);
        /// Tensor strafingMoveAction = action.narrow(-1, 6, 1);
        /// Tensor logProbsStrafingMove = strafingMoveDist.logProb(strafingMoveAction);
        Tensor forwardMoveAction = action.narrow(-1, 0, 1);
        Tensor logProbsForwardMove = forwardMoveDist.logProb(forwardMoveAction);
        Tensor strafingMoveAction = action.narrow(-1, 1, 1);
        Tensor logProbsStrafingMove = strafingMoveDist.logProb(strafingMoveAction);

        // LOGGER.debug("Log probs shapes: look={}, sprint={}, sneak={}, jump={}, move={}",
        //     logProbsLook.shape(), logProbsSprint.shape(), logProbsSneak.shape(), logProbsJump.shape(), logProbsMove.shape());

        /// Tensor totalLogProbs = logProbsLook.add(logProbsSprint).add(logProbsSneak)
        ///     .add(logProbsJump).add(logProbsForwardMove).add(logProbsStrafingMove);
        Tensor totalLogProbs = logProbsForwardMove.add(logProbsStrafingMove);
        // LOGGER.debug("Total log probs shape: {}", totalLogProbs.shape());

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

        /// Tensor entropyJump = jumpDist.entropy().squeeze(-1);
        /// Tensor entropySprint = sprintDist.entropy().squeeze(-1);
        /// Tensor entropySneak = sneakDist.entropy().squeeze(-1);
        /// Tensor lookEntropy = lookDist.entropy();
        /// Tensor entropyLook = lookEntropy.sum(-1);
        /// lookEntropy.close();
        Tensor entropyForwardMove = forwardMoveDist.entropy();
        Tensor entropyStrafingMove = strafingMoveDist.entropy();
        // LOGGER.debug("Entropy shapes: look={}, sprint={}, sneak={}, jump={}, move={}",
        //     entropyLook.shape(), entropySprint.shape(), entropySneak.shape(), entropyJump.shape(), entropyMove.shape());

        /// Tensor totalEntropy = entropyLook.add(entropySprint).add(entropySneak)
        ///     .add(entropyJump).add(entropyForwardMove).add(entropyStrafingMove);
        Tensor totalEntropy = entropyForwardMove.add(entropyStrafingMove);
        // LOGGER.debug("Total entropy shape: {}", totalEntropy.shape());

        /*
        # --- Critic Value ---
        value = self.critic(actor_critic_features)
         */

        Tensor value = this.critic.forward(sharedFeatures);
        // LOGGER.debug("Critic value shape: {}", value.shape());

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


        // LOGGER.debug("--- Exiting getActionAndValue ---");
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
                                 MinecraftRL.LSTMState lstmState) {
    }
}
