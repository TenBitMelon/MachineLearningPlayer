package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.environment.Info;
import com.tenbitmelon.machinelearningplayer.environment.MinecraftEnvironment;
import com.tenbitmelon.machinelearningplayer.environment.Observation;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

public record VectorStepResult(Observation[] observations, double[] rewards, boolean[] terminated,
                               boolean[] truncated, Info[] infos) {
    public int[] logicalOrTerminationsAndTruncations() {
        // stepResult.terminated() || stepResult.truncated();
        int[] ors = new int[terminated.length];
        for (int i = 0; i < terminated.length; i++) {
            ors[i] = terminated[i] || truncated[i] ? 1 : 0;
        }
        return ors;
    }

    public Tensor observationsTensor() {
        Tensor nextObs = torch.zeros(observations.length, Observation.OBSERVATION_SPACE_SIZE);
        for (int i = 0; i < observations.length; i++) {
            Observation observation = observations[i];
            Tensor tensor = observation.toTensor();
            nextObs.get(i).put(tensor);
        }
        return nextObs;
    }
}
