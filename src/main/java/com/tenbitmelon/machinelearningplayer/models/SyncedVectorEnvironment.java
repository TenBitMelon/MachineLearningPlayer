package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.environment.MinecraftEnvironment;
import com.tenbitmelon.machinelearningplayer.environment.Observation;
import com.tenbitmelon.machinelearningplayer.environment.ResetResult;
import com.tenbitmelon.machinelearningplayer.environment.StepResult;
import org.bytedeco.pytorch.Tensor;

import java.util.Arrays;

public class SyncedVectorEnvironment {

    // private final AutoresetMode autoresetMode;
    private final boolean[] terminated;
    private final boolean[] truncated;

    private final int numEnvs;
    private final MinecraftEnvironment[] environments;

    public SyncedVectorEnvironment(ExperimentConfig args) {
        this.numEnvs = args.numEnvs;

        if (this.numEnvs % 2 != 0) {
            throw new IllegalArgumentException("numEnvs must be even for SyncedVectorEnvironment");
        }


        this.environments = new MinecraftEnvironment[numEnvs];
        for (int i = 0; i < numEnvs; i++) {
            environments[i] = new MinecraftEnvironment(args);
            if (i % 2 == 1) {
                environments[i].setTarget(environments[i - 1].agent);
                environments[i - 1].setTarget(environments[i].agent);
            }
        }

        this.terminated = new boolean[numEnvs];
        this.truncated = new boolean[numEnvs];
    }

    public Observation[] getObservation() {
        return Arrays.stream(environments)
            .map(MinecraftEnvironment::getObservation)
            .toArray(Observation[]::new);
    }

    public VectorResetResult reset() {
        Observation[] observations = new Observation[numEnvs];

        for (int i = 0; i < numEnvs; i++) {
            ResetResult resetResult = environments[i].reset();
            observations[i] = resetResult.observation();
            terminated[i] = false;
            truncated[i] = false;
        }
        return new VectorResetResult(observations);
    }

    public void preTickStep(Tensor action) {
        // LOGGER.debug("Stepping in SyncedVectorEnvironment with action: {}", action);

        for (int i = 0; i < numEnvs; i++) {
            environments[i].preTickStep(action.get(i));
        }
    }

    public VectorStepResult postTickStep() {
        // LOGGER.debug("Post tick stepping in SyncedVectorEnvironment");
        Observation[] observations = new Observation[numEnvs];
        double[] rewards = new double[numEnvs];

        for (int i = 0; i < numEnvs; i += 2) {

            StepResult stepResult = environments[i].postTickStep();
            rewards[i] = stepResult.reward();
            terminated[i] = stepResult.terminated();
            truncated[i] = stepResult.truncated();

            StepResult oppositeStepResult = environments[i + 1].postTickStep();
            rewards[i + 1] = oppositeStepResult.reward();
            terminated[i + 1] = oppositeStepResult.terminated();
            truncated[i + 1] = oppositeStepResult.truncated();

            assert terminated[i] == terminated[i + 1] : "Terminated flags do not match for opposite environments";
            assert truncated[i] == truncated[i + 1] : "Truncated flags do not match for opposite environments";

            if (terminated[i] || truncated[i]) {
                ResetResult resetResult = environments[i].reset();
                observations[i] = resetResult.observation();
                ResetResult oppositeResetResult = environments[i + 1].reset();
                observations[i + 1] = oppositeResetResult.observation();
            } else {
                observations[i] = stepResult.observation();
                observations[i + 1] = oppositeStepResult.observation();
            }
        }
        return new VectorStepResult(observations, rewards, terminated, truncated);
    }

    public boolean isReady() {
        return Arrays.stream(environments)
            .allMatch(MinecraftEnvironment::isReady);
    }

    public MinecraftEnvironment getEnvironment(int index) {
        return environments[index];
    }
}
