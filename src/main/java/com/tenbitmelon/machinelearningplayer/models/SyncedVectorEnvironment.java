package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.environment.*;
import org.bytedeco.pytorch.Tensor;

import java.util.Arrays;

public class SyncedVectorEnvironment {

    // private final AutoresetMode autoresetMode;
    private final boolean[] terminated;
    private final boolean[] truncated;
    private final boolean[] autoresetEnvs;

    private final int numEnvs;
    private final MinecraftEnvironment[] environments;

    public SyncedVectorEnvironment(ExperimentConfig args) {
        this.numEnvs = args.numEnvs;
        this.environments = new MinecraftEnvironment[numEnvs];
        for (int i = 0; i < numEnvs; i++) {
            environments[i] = new MinecraftEnvironment(args);

        }

        // this.autoresetMode = AutoresetMode.NEXT_STEP;
        this.terminated = new boolean[numEnvs];
        this.truncated = new boolean[numEnvs];
        this.autoresetEnvs = new boolean[numEnvs];
    }

    public Observation[] getObservation() {
        return Arrays.stream(environments)
            .map(MinecraftEnvironment::getObservation)
            .toArray(Observation[]::new);
    }

    private Info[] getInfo() {
        return Arrays.stream(environments)
            .map(MinecraftEnvironment::getInfo)
            .toArray(Info[]::new);
    }

    public VectorResetResult reset() {
        Observation[] observations = new Observation[numEnvs];
        Info[] infos = new Info[numEnvs];

        for (int i = 0; i < numEnvs; i++) {
            ResetResult resetResult = environments[i].reset();
            observations[i] = resetResult.observation();
            infos[i] = resetResult.info();
            terminated[i] = false;
            truncated[i] = false;
            autoresetEnvs[i] = false;
        }
        return new VectorResetResult(observations, infos);
    }

    public void preTickStep(Tensor action) {
        // LOGGER.debug("Stepping in SyncedVectorEnvironment with action: {}", action);

        for (int i = 0; i < numEnvs; i++) {
            if (!autoresetEnvs[i]) {
                environments[i].preTickStep(action.get(i));
            }
        }
    }

    public VectorStepResult postTickStep() {
        // LOGGER.debug("Post tick stepping in SyncedVectorEnvironment");
        Observation[] observations = new Observation[numEnvs];
        double[] rewards = new double[numEnvs];
        Info[] infos = new Info[numEnvs];

        for (int i = 0; i < numEnvs; i++) {
            if (autoresetEnvs[i]) {
                ResetResult resetResult = environments[i].reset();
                observations[i] = resetResult.observation();
                infos[i] = resetResult.info();
                rewards[i] = 0.0;
                terminated[i] = false;
                truncated[i] = false;
            } else {
                StepResult stepResult = environments[i].postTickStep();
                observations[i] = stepResult.observation();
                rewards[i] = stepResult.reward();
                terminated[i] = stepResult.terminated();
                truncated[i] = stepResult.truncated();
                infos[i] = stepResult.info();
            }

            autoresetEnvs[i] = terminated[i] || truncated[i];
        }
        return new VectorStepResult(observations, rewards, terminated, truncated, infos);
    }

    public boolean isReady() {
        return Arrays.stream(environments)
            .allMatch(MinecraftEnvironment::isReady);
    }

    public enum AutoresetMode {
        DISABLED,
        NEXT_STEP,
        SAME_STEP
    }
}
