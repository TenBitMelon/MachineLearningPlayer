package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.environment.Observation;
import org.bytedeco.pytorch.Tensor;

import static com.tenbitmelon.machinelearningplayer.models.VectorStepResult.createObservationTensor;

public record VectorResetResult(Observation[] observations) implements AutoCloseable {
    public Tensor observationsTensor() {
        return createObservationTensor(observations);
    }

    @Override
    public void close() {
        for (Observation observation : observations) {
            observation.close();
        }
    }
}
