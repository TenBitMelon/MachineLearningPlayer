package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.environment.Info;
import com.tenbitmelon.machinelearningplayer.environment.Observation;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

public record VectorResetResult(Observation[] observations, Info[] infos) {
    public Tensor observationsTensor() {
        TensorVector tensorVector = new TensorVector();
        for (Observation observation : observations) {
            Tensor tensor = observation.tensor();
            tensorVector.push_back(tensor);
        }
        return torch.stack(tensorVector, 0);  // Stack along batch dimension

    }
}
