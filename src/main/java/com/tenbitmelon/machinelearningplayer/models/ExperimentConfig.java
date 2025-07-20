package com.tenbitmelon.machinelearningplayer.models;

public class ExperimentConfig {

    public static final ExperimentConfig config = new ExperimentConfig();
    /**
     * Seed of the experiment.
     */
    public final int seed = 1;
    /**
     * The learning rate of the optimizer.
     */
    public final float learningRate = 2.5e-4f; // Note the 'f' suffix for float literals
    /**
     * The number of parallel game environments.
     */
    public final int numEnvs = 8;
    /**
     * Toggle learning rate annealing for policy and value networks.
     */
    public final boolean annealLr = true;
    /**
     * The discount factor gamma.
     */
    public final float gamma = 0.99f;
    /**
     * The lambda for the general advantage estimation.
     */
    public final float gaeLambda = 0.95f;
    /**
     * The number of mini-batches.
     */
    public final int numMinibatches = 4;

    // Algorithm specific arguments
    /**
     * The K epochs to update the policy.
     */
    public final int updateEpochs = 4;
    /**
     * Total timesteps of the experiments.
     */
    // public int totalTimesteps = 5;
    // public int totalTimesteps = 10_000_000;
    /**
     * Toggles advantages normalization.
     */
    public final boolean normAdv = true;
    /**
     * The surrogate clipping coefficient.
     */
    public final float clipCoef = 0.1f;
    /**
     * Toggles whether or not to use a clipped loss for the value function, as per the paper.
     */
    public final boolean clipVloss = true;
    /**
     * Coefficient of the entropy.
     */
    public final float entCoef = 0.01f;
    /**
     * Coefficient of the value function.
     */
    public final float vfCoef = 0.5f;
    /**
     * The maximum norm for the gradient clipping.
     */
    public final float maxGradNorm = 0.5f;
    /**
     * The target KL divergence threshold. Can be null if not used.
     */
    public final Float targetKl = null; // Use wrapper class Float to allow null
    /**
     * The name of this experiment.
     * In Python, this was derived from the filename. In Java, we might use the class name
     * or a manually set string. For simplicity, let's use a default or allow it to be set.
     * A more direct equivalent to os.path.basename(__file__) is complex in Java without
     * knowing the execution context, so a simple string or class name is often used.
     * If this class itself represents the experiment, its name is a good candidate.
     */
    public String expName = "MinecraftRLExperiment"; // Default name, can be overridden
    /**
     * If toggled (true), PyTorch's `torch.backends.cudnn.deterministic` is typically set to true.
     * The original comment "if toggled, `torch.backends.cudnn.deterministic=False`" was a bit ambiguous.
     * Assuming true means more deterministic.
     */
    public boolean torchDeterministic = true;
    /**
     * If toggled (true), CUDA will be enabled by default if available.
     */
    public boolean cuda = true;
    /**
     * If toggled (true), this experiment will be tracked with Weights and Biases.
     */
    public boolean track = false;
    /**
     * The WandB's project name.
     */
    public String wandbProjectName = "cleanRL";
    /**
     * The entity (team) of WandB's project. Can be null.
     */
    public String wandbEntity = null;
    /**
     * Whether to capture videos of the agent performances (check out `videos` folder).
     */
    public boolean captureVideo = false;

    // to be filled in runtime
    /**
     * The number of steps to run in each environment per policy rollout.
     */
    // public int numSteps = 5;
    public int numSteps = 200;
    /**
     * The batch size (computed in runtime, e.g., numEnvs * numSteps).
     * Initialized to 0 or a sensible default, will be calculated later.
     */
    public int batchSize = 0;
    /**
     * The mini-batch size (computed in runtime, e.g., batchSize / numMinibatches).
     * Initialized to 0 or a sensible default, will be calculated later.
     */
    public int minibatchSize = 0;
    /**
     * The number of iterations (computed in runtime, e.g., totalTimesteps / batchSize).
     * Initialized to 0 or a sensible default, will be calculated later.
     */
    // public int numIterations = 0;
    public int numIterations = 5;

    private ExperimentConfig() {
        // Default constructor, can be used to initialize with default values
        // args.batch_size = int(args.num_envs * args.num_steps)
        // args.minibatch_size = int(args.batch_size // args.num_minibatches)
        // args.num_iterations = args.total_timesteps // args.batch_size

        batchSize = numEnvs * numSteps;
        System.out.println("Batch size: " + batchSize);
        minibatchSize = batchSize / numMinibatches;
        System.out.println("Minibatch size: " + minibatchSize);
        // numIterations = totalTimesteps / batchSize;
        System.out.println("Number of iterations: " + numIterations);
    }

    public static ExperimentConfig getInstance() {
        return config;
    }
}