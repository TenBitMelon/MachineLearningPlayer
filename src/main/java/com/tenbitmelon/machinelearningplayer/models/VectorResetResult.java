package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.environment.Observation;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

public record VectorResetResult(Observation[] observations) implements AutoCloseable {
    public Tensor observationsTensor() {
        TensorVector tensorVector = new TensorVector();
        for (Observation observation : observations) {
            Tensor tensor = observation.tensor();
            tensorVector.push_back(tensor);
        }
        Tensor stack = torch.stack(tensorVector, 0);
        tensorVector.close();
        return stack;  // Stack along batch dimension
    }

    @Override
    public void close() {
        for (Observation observation : observations) {
            observation.close();
        }
    }
}
