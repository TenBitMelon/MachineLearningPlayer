package com.tenbitmelon.machinelearningplayer.models;

public class ExperimentConfig {

    public static final ExperimentConfig config = new ExperimentConfig();
    /**
     * The learning rate of the optimizer.
     */
    public final float learningRate = 3e-5f;
    /**
     * The number of parallel game environments.
     */
    public final int numEnvs = 64;
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
    public final int numMinibatches = 8;
    /**
     * The K epochs to update the policy.
     */
    public final int updateEpochs = 4;
    /**
     * Toggles advantages normalization.
     */
    public final boolean normAdv = true;
    /**
     * The surrogate clipping coefficient.
     */
    public final float clipCoef = 0.1f;
    /**
     * Toggles whether to use a clipped loss for the value function, as per the paper.
     */
    public final boolean clipVloss = true;
    /**
     * Coefficient of the entropy.
     */
    public final float entCoef = 0.005f;
    /**
     * Coefficient of the value function.
     */
    public final float vfCoef = 0.7f;
    /**
     * The maximum norm for the gradient clipping.
     */
    public final float maxGradNorm = 0.5f;
    /**
     * The target KL divergence threshold. Can be null if not used.
     */
    public final Float targetKl = 0.02f;
    /**
     * The number of steps to run in each environment per policy rollout.
     */
    public int numSteps = 260;
    /**
     * The batch size (computed in runtime, e.g., numEnvs * numSteps).
     * Initialized to 0 or a sensible default, will be calculated later.
     */
    public int batchSize = numEnvs * numSteps;
    /**
     * The number of iterations to run. One iteration is numEnvs * numSteps steps.
     */
    public int numIterations = 5000;

    private ExperimentConfig() {}

    public static ExperimentConfig getInstance() {
        return config;
    }
}