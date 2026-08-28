package com.tenbitmelon.machinelearningplayer.environment;

import net.minecraft.world.phys.Vec3;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

public class Observation implements AutoCloseable {

    public static final int SIZE_PITCH = 1; // up/down, don't need left/right because all directions are rotationally relative
    public static final int SIZE_SPRINTING = 1;
    public static final int SIZE_SNEAKING = 1;
    public static final int SIZE_ON_GROUND = 1;
    public static final int SIZE_ATTACK_COOLDOWN = 1;
    public static final int SIZE_HEALTH = 1;
    public static final int SIZE_VELOCITY_VEC = 3;
    public static final int SIZE_OPPONENT_DIRECTION_VEC = 3;
    public static final int SIZE_OPPONENT_DISTANCE = 1;
    public static final int SIZE_OPPONENT_VELOCITY_VEC = 3;
    public static final int SIZE_OPPONENT_LOOK_DIRECTION_VEC = 3;
    public static final int OFFSET_PITCH = 0;
    public static final int SIZE_LOCAL_HEIGHT_MAP = 7 * 7; // 7x7 grid of blocks around the agent, each block is represented by its height relative to the agent's feet
    public static final int OFFSET_SPRINTING = OFFSET_PITCH + SIZE_PITCH;
    public static final int OFFSET_SNEAKING = OFFSET_SPRINTING + SIZE_SPRINTING;
    public static final int OFFSET_ON_GROUND = OFFSET_SNEAKING + SIZE_SNEAKING;
    public static final int OFFSET_ATTACK_COOLDOWN = OFFSET_ON_GROUND + SIZE_ON_GROUND;
    public static final int OFFSET_HEALTH = OFFSET_ATTACK_COOLDOWN + SIZE_ATTACK_COOLDOWN;
    public static final int OFFSET_VELOCITY_VEC = OFFSET_HEALTH + SIZE_HEALTH;
    public static final int OFFSET_OPPONENT_DIRECTION_VEC = OFFSET_VELOCITY_VEC + SIZE_VELOCITY_VEC;
    public static final int OFFSET_OPPONENT_DISTANCE = OFFSET_OPPONENT_DIRECTION_VEC + SIZE_OPPONENT_DIRECTION_VEC;
    public static final int OFFSET_OPPONENT_VELOCITY_VEC = OFFSET_OPPONENT_DISTANCE + SIZE_OPPONENT_DISTANCE;
    public static final int OFFSET_OPPONENT_LOOK_DIRECTION_VEC = OFFSET_OPPONENT_VELOCITY_VEC + SIZE_OPPONENT_VELOCITY_VEC;
    public static final int OFFSET_LOCAL_HEIGHT_MAP = OFFSET_OPPONENT_LOOK_DIRECTION_VEC + SIZE_OPPONENT_LOOK_DIRECTION_VEC;

    public static final int OBSERVATION_SPACE_SIZE = OFFSET_LOCAL_HEIGHT_MAP + SIZE_LOCAL_HEIGHT_MAP;

    final Tensor data;

    public Observation(float pitch, boolean sprinting, boolean sneaking, boolean onGround, float attackCooldown, float health, Vec3 velocity, Vec3 opponentDirectionVec, float opponentDistance, Vec3 opponentVelocityVec, Vec3 opponentLookDirectionVec, float[] localHeightMap) {
        float[] observationData = new float[Observation.OBSERVATION_SPACE_SIZE];
        observationData[OFFSET_PITCH] = pitch;
        observationData[OFFSET_SPRINTING] = sprinting ? 1.0f : 0.0f;
        observationData[OFFSET_SNEAKING] = sneaking ? 1.0f : 0.0f;
        observationData[OFFSET_ON_GROUND] = onGround ? 1.0f : 0.0f;
        observationData[OFFSET_ATTACK_COOLDOWN] = attackCooldown;
        observationData[OFFSET_HEALTH] = health;
        observationData[OFFSET_VELOCITY_VEC] = (float) velocity.x;
        observationData[OFFSET_VELOCITY_VEC + 1] = (float) velocity.y;
        observationData[OFFSET_VELOCITY_VEC + 2] = (float) velocity.z;
        observationData[OFFSET_OPPONENT_DIRECTION_VEC] = (float) opponentDirectionVec.x;
        observationData[OFFSET_OPPONENT_DIRECTION_VEC + 1] = (float) opponentDirectionVec.y;
        observationData[OFFSET_OPPONENT_DIRECTION_VEC + 2] = (float) opponentDirectionVec.z;
        observationData[OFFSET_OPPONENT_DISTANCE] = opponentDistance;
        observationData[OFFSET_OPPONENT_VELOCITY_VEC] = (float) opponentVelocityVec.x;
        observationData[OFFSET_OPPONENT_VELOCITY_VEC + 1] = (float) opponentVelocityVec.y;
        observationData[OFFSET_OPPONENT_VELOCITY_VEC + 2] = (float) opponentVelocityVec.z;
        observationData[OFFSET_OPPONENT_LOOK_DIRECTION_VEC] = (float) opponentLookDirectionVec.x;
        observationData[OFFSET_OPPONENT_LOOK_DIRECTION_VEC + 1] = (float) opponentLookDirectionVec.y;
        observationData[OFFSET_OPPONENT_LOOK_DIRECTION_VEC + 2] = (float) opponentLookDirectionVec.z;
        System.arraycopy(localHeightMap, 0, observationData, OFFSET_LOCAL_HEIGHT_MAP, SIZE_LOCAL_HEIGHT_MAP);

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
     * - Shape: (1,)
     */
    public Tensor pitch() {
        return data.narrow(0, OFFSET_PITCH, SIZE_PITCH); // (OBSERVATION_SPACE_SIZE,) -> (1,)
    }

    /**
     * Sprinting:
     * - Shape: (1,)
     */
    public Tensor sprinting() {
        return data.narrow(0, OFFSET_SPRINTING, SIZE_SPRINTING); // (OBSERVATION_SPACE_SIZE,) -> (1,)
    }

    /**
     * Sneaking:
     * - Shape: (1,)
     */
    public Tensor sneaking() {
        return data.narrow(0, OFFSET_SNEAKING, SIZE_SNEAKING); // (OBSERVATION_SPACE_SIZE,) -> (1,)
    }

    /**
     * On Ground:
     * - Shape: (1,)
     */
    public Tensor onGround() {
        return data.narrow(0, OFFSET_ON_GROUND, SIZE_ON_GROUND); // (OBSERVATION_SPACE_SIZE,) -> (1,)
    }


    /**
     * Opponent Direction Vec:
     * - Shape: (3,)
     */
    public Tensor opponentDirectionVec() {
        return data.narrow(0, OFFSET_OPPONENT_DIRECTION_VEC, SIZE_OPPONENT_DIRECTION_VEC); // (OBSERVATION_SPACE_SIZE,) -> (3,)
    }

    /**
     * Opponent Distance:
     * - Shape: (1,)
     */
    public Tensor opponentDistance() {
        return data.narrow(0, OFFSET_OPPONENT_DISTANCE, SIZE_OPPONENT_DISTANCE); // (OBSERVATION_SPACE_SIZE,) -> (1,)
    }

    /**
     * Opponent Velocity Vec:
     * - Shape: (3,)
     */
    public Tensor opponentVelocityVec() {
        return data.narrow(0, OFFSET_OPPONENT_VELOCITY_VEC, SIZE_OPPONENT_VELOCITY_VEC); // (OBSERVATION_SPACE_SIZE,) -> (3,)
    }

    /**
     * Opponent Look Direction Vec (in agent-local angle space):
     * - Shape: (3,)
     */
    public Tensor opponentLookDirectionVec() {
        return data.narrow(0, OFFSET_OPPONENT_LOOK_DIRECTION_VEC, SIZE_OPPONENT_LOOK_DIRECTION_VEC); // (OBSERVATION_SPACE_SIZE,) -> (3,)
    }

    /**
     * Attack Cooldown:
     * - Shape: (1,)
     */
    public Tensor attackCooldown() {
        return data.narrow(0, OFFSET_ATTACK_COOLDOWN, SIZE_ATTACK_COOLDOWN); // (OBSERVATION_SPACE_SIZE,) -> (1,)
    }


    /**
     * Local Height Map:
     * - Shape: (49,) representing a 7x7 grid of blocks around the agent, each block is represented by its height relative to the agent's feet
     */
    public Tensor localHeightMap() {
        return data.narrow(0, OFFSET_LOCAL_HEIGHT_MAP, SIZE_LOCAL_HEIGHT_MAP); // (OBSERVATION_SPACE_SIZE,) -> (49,)
    }

    /**
     * @return A tensor representation of the observation.
     */
    public Tensor tensor() {
        return data;
    }

    @Override
    public void close() {
        data.close();
    }

}
