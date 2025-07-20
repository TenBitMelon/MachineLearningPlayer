package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.environment.Info;
import com.tenbitmelon.machinelearningplayer.environment.MinecraftEnvironment;
import com.tenbitmelon.machinelearningplayer.environment.Observation;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorIndex;
import org.bytedeco.pytorch.TensorIndexVector;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

public record VectorResetResult(Observation[] observations, Info[] infos) {
    public Tensor observationsTensor() {
        // Tensor nextObs = torch.zeros(observations.length, Observation.OBSERVATION_SPACE_SIZE);
        // for (int i = 0; i < observations.length; i++) {
        //     Observation observation = observations[i];
        //     Tensor tensor = observation.toTensor();
        //
        //     nextObs.index_put_(
        //         new TensorIndexVector(new TensorIndex(i)),
        //         tensor
        //     );
        // }
        // return nextObs;

        TensorVector tensorVector = new TensorVector();
        for (Observation observation : observations) {
            Tensor tensor = observation.toTensor();
            tensorVector.push_back(tensor);
        }
        return torch.stack(tensorVector, 0);  // Stack along batch dimension

    }
}
