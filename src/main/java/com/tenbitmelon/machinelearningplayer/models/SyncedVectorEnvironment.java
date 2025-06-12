package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.environment.*;
import org.bytedeco.pytorch.Tensor;

import java.util.Arrays;

public class SyncedVectorEnvironment {

    private final int numEnvs;
    private final MinecraftEnvironment[] environments;

    public SyncedVectorEnvironment(int numEnvs) {
        this.numEnvs = numEnvs;
        this.environments = new MinecraftEnvironment[numEnvs];
        for (int i = 0; i < numEnvs; i++) {
            environments[i] = new MinecraftEnvironment();
        }
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

    public VectorResetResult reset(int seed) {
        Observation[] observations = new Observation[numEnvs];
        Info[] infos = new Info[numEnvs];
        for (int i = 0; i < numEnvs; i++) {
            ResetResult resetResult = environments[i].reset(seed);
            observations[i] = resetResult.observation();
            infos[i] = resetResult.info();
        }
        return new VectorResetResult(observations, infos);
    }

    public VectorStepResult step(Tensor action) {
        Observation[] observations = new Observation[numEnvs];
        double[] rewards = new double[numEnvs];
        boolean[] terminated = new boolean[numEnvs];
        boolean[] truncated = new boolean[numEnvs];
        Info[] infos = new Info[numEnvs];
        for (int i = 0; i < numEnvs; i++) {
            StepResult stepResult = environments[i].step(action.get(i));
            observations[i] = stepResult.observation();
            rewards[i] = stepResult.reward();
            terminated[i] = stepResult.terminated();
            truncated[i] = stepResult.truncated();
            infos[i] = stepResult.info();
        }
        return new VectorStepResult(observations, rewards, terminated, truncated, infos);
    }

    public boolean isReady() {
        return Arrays.stream(environments)
            .allMatch(MinecraftEnvironment::isReady);
    }
}
