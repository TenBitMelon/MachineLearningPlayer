package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.environment.Observation;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

public record VectorStepResult(Observation[] observations, double[] rewards, boolean[] terminated,
                               boolean[] truncated, int bowSelectedSteps, int bowDrawingSteps,
                               int bowFullyDrawnSteps, int shieldUsingSteps,
                               Tensor nextValuePreReset) implements AutoCloseable {
    // public int[] logicalOrTerminationsAndTruncations() {
    //     // stepResult.terminated() || stepResult.truncated();
    //     int[] ors = new int[terminated.length];
    //     for (int i = 0; i < terminated.length; i++) {
    //         ors[i] = terminated[i] || truncated[i] ? 1 : 0;
    //     }
    //     return ors;
    // }

    static Tensor createObservationTensor(Observation[] observations) {
        TensorVector tensorVector = new TensorVector();
        for (Observation observation : observations) {
            Tensor tensor = observation.tensor();
            tensorVector.push_back(tensor);
        }
        Tensor stack = torch.stack(tensorVector, 0);  // Stack along batch dimension
        tensorVector.close();
        return stack;
    }

    public Tensor observationsTensor() {
        return createObservationTensor(observations);
    }

    public int numTruncations() {
        int count = 0;
        for (boolean truncated : truncated) {
            if (truncated) {
                count++;
            }
        }
        return count;
    }

    public int numTerminations() {
        int count = 0;
        for (boolean terminated : terminated) {
            if (terminated) {
                count++;
            }
        }
        return count;
    }

    public int bowSelectedSteps() {
        return bowSelectedSteps;
    }

    public int bowDrawingSteps() {
        return bowDrawingSteps;
    }

    public int bowFullyDrawnSteps() {
        return bowFullyDrawnSteps;
    }

    public int shieldUsingSteps() {
        return shieldUsingSteps;
    }

    @Override
    public void close() {
        for (Observation observation : observations) {
            observation.close();
        }
    }
}
