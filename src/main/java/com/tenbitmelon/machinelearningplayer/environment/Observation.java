package com.tenbitmelon.machinelearningplayer.environment;

import net.minecraft.world.phys.Vec3;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

public class Observation {

    private static final int SIZE_PITCH = 1;
    private static final int SIZE_SPRINTING = 1;
    private static final int SIZE_SNEAKING = 1;
    private static final int SIZE_ON_GROUND = 1;
    private static final int SIZE_CENTER_DISTANCE = 1;
    private static final int SIZE_OPPONENT_DIRECTION_VEC = 3;
    private static final int SIZE_OPPONENT_DISTANCE = 1;

    private static final int OFFSET_PITCH = 0;
    private static final int OFFSET_SPRINTING = OFFSET_PITCH + SIZE_PITCH;
    private static final int OFFSET_SNEAKING = OFFSET_SPRINTING + SIZE_SPRINTING;
    private static final int OFFSET_ON_GROUND = OFFSET_SNEAKING + SIZE_SNEAKING;
    private static final int OFFSET_CENTER_DISTANCE = OFFSET_ON_GROUND + SIZE_ON_GROUND;
    private static final int OFFSET_OPPONENT_DIRECTION_VEC = OFFSET_CENTER_DISTANCE + SIZE_CENTER_DISTANCE;
    private static final int OFFSET_OPPONENT_DISTANCE = OFFSET_OPPONENT_DIRECTION_VEC + SIZE_OPPONENT_DIRECTION_VEC;

    public static final int OBSERVATION_SPACE_SIZE = OFFSET_OPPONENT_DISTANCE + SIZE_OPPONENT_DISTANCE;

    final Tensor data;

    public Observation(float pitch, boolean sprinting, boolean sneaking, boolean onGround, float centerDistance, Vec3 opponentDirectionVec, float opponentDistance) {
        float[] observationData = new float[Observation.OBSERVATION_SPACE_SIZE];
        observationData[OFFSET_PITCH] = pitch;
        observationData[OFFSET_SPRINTING] = sprinting ? 1.0f : 0.0f;
        observationData[OFFSET_SNEAKING] = sneaking ? 1.0f : 0.0f;
        observationData[OFFSET_ON_GROUND] = onGround ? 1.0f : 0.0f;
        observationData[OFFSET_CENTER_DISTANCE] = centerDistance;
        observationData[OFFSET_OPPONENT_DIRECTION_VEC] = (float) opponentDirectionVec.x;
        observationData[OFFSET_OPPONENT_DIRECTION_VEC + 1] = (float) opponentDirectionVec.y;
        observationData[OFFSET_OPPONENT_DIRECTION_VEC + 2] = (float) opponentDirectionVec.z;
        observationData[OFFSET_OPPONENT_DISTANCE] = opponentDistance;
        this.data = torch.tensor(observationData);
    }

    public Observation() {
        this.data = torch.zeros(OBSERVATION_SPACE_SIZE);
    }

    public Observation(Tensor data) {
        if (data.size(0) != OBSERVATION_SPACE_SIZE) {
            throw new IllegalArgumentException("Observation tensor must have size " + OBSERVATION_SPACE_SIZE);
        }
        this.data = data;
    }

    /**
     * Pitch:
     * - Shape: (2,)
     */
    public Tensor pitch() {
        return data.narrow(0, OFFSET_PITCH, SIZE_PITCH);
    }

    /**
     * Sprinting:
     * - Shape: (1,)
     */
    public Tensor sprinting() {
        return data.narrow(0, OFFSET_SPRINTING, SIZE_SPRINTING);
    }

    /**
     * Sneaking:
     * - Shape: (1,)
     */
    public Tensor sneaking() {
        return data.narrow(0, OFFSET_SNEAKING, SIZE_SNEAKING);
    }

    /**
     * On Ground:
     * - Shape: (1,)
     */
    public Tensor onGround() {
        return data.narrow(0, OFFSET_ON_GROUND, SIZE_ON_GROUND);
    }

    /**
     * Center Distance:
     * - Shape: (1,)
     */
    public Tensor centerDistance() {
        return data.narrow(0, OFFSET_CENTER_DISTANCE, SIZE_CENTER_DISTANCE);
    }

    /**
     * Opponent Direction Vec:
     * - Shape: (3,)
     */
    public Tensor opponentDirectionVec() {
        return data.narrow(0, OFFSET_OPPONENT_DIRECTION_VEC, SIZE_OPPONENT_DIRECTION_VEC);
    }

    /**
     * Opponent Distance:
     * - Shape: (1,)
     */
    public Tensor opponentDistance() {
        return data.narrow(0, OFFSET_OPPONENT_DISTANCE, SIZE_OPPONENT_DISTANCE);
    }


    /**
     * @return A tensor representation of the observation.
     */
    public Tensor tensor() {
        return data;
    }
}
