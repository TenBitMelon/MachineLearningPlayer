package com.tenbitmelon.machinelearningplayer;

import org.bytedeco.pytorch.global.torch_cuda;
import org.yaml.snakeyaml.DumperOptions;
import org.yaml.snakeyaml.LoaderOptions;
import org.yaml.snakeyaml.TypeDescription;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.Constructor;
import org.yaml.snakeyaml.nodes.Node;
import org.yaml.snakeyaml.nodes.Tag;
import org.yaml.snakeyaml.representer.Representer;

import javax.annotation.Nullable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class ExperimentConfig {

    public final String experimentId = UUID.randomUUID().toString();
    /**
     * The learning rate of the optimizer.
     */
    public final float learningRate = 1.8e-4f;
    /**
     * Toggle learning rate annealing for policy and value networks.
     */
    public final boolean annealLr = false;
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
    public final float clipCoef = 0.2f;
    /**
     * Toggles whether to use a clipped loss for the value function, as per the paper.
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
    public final Float targetKl = 0.02f;
    /**
     * The number of parallel game environments.
     */
    public int numEnvs = 32;
    /**
     * The number of steps to run in each environment per policy rollout.
     */
    public int numSteps = 200;
    /**
     * The number of steps to run in each environment before truncating the episode.
     */
    public int maxEnvironmentSteps = 600;
    /**
     * The batch size (computed in runtime, e.g., numEnvs * numSteps).
     * Initialized to 0 or a sensible default, will be calculated later.
     */
    public int batchSize = numEnvs * numSteps;
    /**
     * The number of iterations to run. One iteration is numEnvs * numSteps steps.
     */
    public int numIterations = 10000;
    /**
     * The checkpoint to start from.
     */
    @Nullable
    public Integer startingCheckpoint = null;

    /**
     * Feature flags for enabling or disabling specific features in the experiment.
     */
    public Set<FeatureFlag> featureFlags = EnumSet.noneOf(FeatureFlag.class);

    public ExperimentConfig() {}

    public static ExperimentConfig fromFile(String filePath) {
        if (filePath == null || filePath.isEmpty()) {
            throw new IllegalArgumentException("File path cannot be null or empty");
        }
        try {
            String yamlContent = Files.readString(Path.of(filePath));

            LoaderOptions loaderOptions = new LoaderOptions();
            Constructor constructor = new Constructor(ExperimentConfig.class, loaderOptions);

            TypeDescription configDescription = new TypeDescription(ExperimentConfig.class);
            configDescription.addPropertyParameters("featureFlags", FeatureFlag.class);
            constructor.addTypeDescription(configDescription);

            Yaml yaml = new Yaml(constructor);
            return yaml.load(yamlContent);

        } catch (IOException e) {
            throw new RuntimeException("Failed to read the configuration file: " + filePath, e);
        }
    }

    public void save() {
        String filePath = "training/" + this.experimentId + "/args.yaml";
        try {
            DumperOptions options = new DumperOptions();
            options.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK);
            options.setPrettyFlow(true);
            Representer representer = new Representer(options) {
                {
                    addClassTag(ExperimentConfig.class, Tag.MAP);
                    addClassTag(FeatureFlag.class, Tag.STR);

                    this.multiRepresenters.put(Set.class, data ->
                        representSequence(Tag.SEQ, (Set<?>) data, DumperOptions.FlowStyle.BLOCK)
                    );
                }
            };

            Yaml yaml = new Yaml(representer, options);
            String yamlContent = yaml.dump(this);
            Files.writeString(Path.of(filePath), yamlContent);
        } catch (IOException e) {
            throw new RuntimeException("Failed to write the configuration file: " + filePath, e);
        }


    }

    @Override
    public String toString() {
        return "ExperimentConfig{" +
            "learningRate=" + learningRate +
            ", annealLr=" + annealLr +
            ", gamma=" + gamma +
            ", gaeLambda=" + gaeLambda +
            ", numMinibatches=" + numMinibatches +
            ", updateEpochs=" + updateEpochs +
            ", normAdv=" + normAdv +
            ", clipCoef=" + clipCoef +
            ", clipVloss=" + clipVloss +
            ", entCoef=" + entCoef +
            ", vfCoef=" + vfCoef +
            ", maxGradNorm=" + maxGradNorm +
            ", targetKl=" + targetKl +
            ", numEnvs=" + numEnvs +
            ", numSteps=" + numSteps +
            ", maxEnvironmentSteps=" + maxEnvironmentSteps +
            ", batchSize=" + batchSize +
            ", numIterations=" + numIterations +
            ", startingCheckpoint=" + startingCheckpoint +
            '}';
    }

    public enum FeatureFlag {
        HEIGHT_MAP_CONV,
        LSTM_SIZE_128,
        LAYER_NORM,
        LEAK_PROBE,
        ALLOCATOR_SNAPSHOT,
    }
}