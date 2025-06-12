package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.environment.Info;
import com.tenbitmelon.machinelearningplayer.environment.MinecraftEnvironment;
import com.tenbitmelon.machinelearningplayer.environment.Observation;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

public record VectorResetResult(Observation[] observations, Info[] infos) {
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
