package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.environment.MinecraftEnvironment;
import com.tenbitmelon.machinelearningplayer.environment.Observation;
import com.tenbitmelon.machinelearningplayer.environment.StepResult;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.util.Arrays;

import static com.tenbitmelon.machinelearningplayer.models.TrainingManager.zerosLikeNumEnvs;
import static com.tenbitmelon.machinelearningplayer.models.VectorStepResult.createObservationTensor;
import static com.tenbitmelon.machinelearningplayer.util.Utils.tensorString;

public class SyncedVectorEnvironment {

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
    }

    public Observation[] getObservation() {
        return Arrays.stream(environments)
            .map(MinecraftEnvironment::getObservation)
            .toArray(Observation[]::new);
    }

    public VectorResetResult reset() {
        Observation[] observations = new Observation[numEnvs];

        for (int i = 0; i < numEnvs; i++) {
            environments[i].reset();
        }
        for (int i = 0; i < numEnvs; i++) {
            observations[i] = environments[i].getObservation();
        }

        return new VectorResetResult(observations);
    }

    public void preTickStep(Tensor action) {
        // LOGGER.debug("Stepping in SyncedVectorEnvironment with action: {}", action);

        for (int i = 0; i < numEnvs; i++) {
            environments[i].preTickStep(action.get(i));
        }
    }

    public VectorStepResult postTickStep(MinecraftRL model, MinecraftRL.LSTMState nextLstmState, Device device) {
        // LOGGER.debug("Post tick stepping in SyncedVectorEnvironment");
        Observation[] observations = new Observation[numEnvs];
        double[] rewards = new double[numEnvs];

        boolean[] terminated = new boolean[numEnvs];
        boolean[] truncated = new boolean[numEnvs];

        int bowSelectedSteps = 0;
        int bowDrawingSteps = 0;
        int bowFullyDrawnSteps = 0;
        int shieldUsingSteps = 0;

        for (int i = 0; i < numEnvs; i += 2) {

            StepResult stepResult = environments[i].postTickStep();
            rewards[i] = stepResult.reward();
            terminated[i] = stepResult.terminated();
            truncated[i] = stepResult.truncated();
            bowSelectedSteps += stepResult.bowSelected() ? 1 : 0;
            bowDrawingSteps += stepResult.bowDrawing() ? 1 : 0;
            bowFullyDrawnSteps += stepResult.bowFullyDrawn() ? 1 : 0;
            shieldUsingSteps += stepResult.shieldUsing() ? 1 : 0;

            StepResult oppositeStepResult = environments[i + 1].postTickStep();
            rewards[i + 1] = oppositeStepResult.reward();
            terminated[i + 1] = oppositeStepResult.terminated();
            truncated[i + 1] = oppositeStepResult.truncated();
            bowSelectedSteps += oppositeStepResult.bowSelected() ? 1 : 0;
            bowDrawingSteps += oppositeStepResult.bowDrawing() ? 1 : 0;
            bowFullyDrawnSteps += oppositeStepResult.bowFullyDrawn() ? 1 : 0;
            shieldUsingSteps += oppositeStepResult.shieldUsing() ? 1 : 0;

            assert terminated[i] == terminated[i + 1] : "Terminated flags do not match for opposite environments";
            assert truncated[i] == truncated[i + 1] : "Truncated flags do not match for opposite environments";

            observations[i] = stepResult.observation();
            observations[i + 1] = oppositeStepResult.observation();
        }

        Tensor observationTensor = createObservationTensor(observations).to(device, torch.ScalarType.Float);

        Tensor nextValuePreReset = model.getValue(observationTensor, nextLstmState, zerosLikeNumEnvs);
        nextValuePreReset = nextValuePreReset.reshape(-1); // (numEnvs,1) -> (numEnvs,)

        for (int i = 0; i < numEnvs; i += 2) {
            if (terminated[i] || truncated[i]) {
                environments[i].reset();
                environments[i + 1].reset();
                observations[i] = environments[i].getObservation();
                observations[i + 1] = environments[i + 1].getObservation();
            }
        }

        return new VectorStepResult(
            observations,
            rewards,
            terminated,
            truncated,
            bowSelectedSteps,
            bowDrawingSteps,
            bowFullyDrawnSteps,
            shieldUsingSteps,
            nextValuePreReset
        );
    }

    public boolean isReady() {
        return Arrays.stream(environments)
            .allMatch(MinecraftEnvironment::isReady);
    }

    public MinecraftEnvironment getEnvironment(int index) {
        return environments[index];
    }
}
