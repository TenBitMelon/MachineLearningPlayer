package com.tenbitmelon.machinelearningplayer.environment;

import net.minecraft.world.phys.Vec3;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

public class Observation implements AutoCloseable {

    private static final int SIZE_PITCH = 1; // up/down, don't need left/right because all directions are rotationally relative
    private static final int SIZE_SPRINTING = 1;
    private static final int SIZE_SNEAKING = 1;
    private static final int SIZE_ON_GROUND = 1;
    private static final int SIZE_ATTACK_COOLDOWN = 1;
    private static final int SIZE_HEALTH = 1;
    private static final int SIZE_VELOCITY_VEC = 3;
    private static final int SIZE_OPPONENT_DIRECTION_VEC = 3;
    private static final int SIZE_OPPONENT_DISTANCE = 1;
    private static final int SIZE_OPPONENT_VELOCITY_VEC = 3;

    private static final int OFFSET_PITCH = 0;
    private static final int OFFSET_SPRINTING = OFFSET_PITCH + SIZE_PITCH;
    private static final int OFFSET_SNEAKING = OFFSET_SPRINTING + SIZE_SPRINTING;
    private static final int OFFSET_ON_GROUND = OFFSET_SNEAKING + SIZE_SNEAKING;
    private static final int OFFSET_ATTACK_COOLDOWN = OFFSET_ON_GROUND + SIZE_ON_GROUND;
    private static final int OFFSET_HEALTH = OFFSET_ATTACK_COOLDOWN + SIZE_ATTACK_COOLDOWN;
    private static final int OFFSET_VELOCITY_VEC = OFFSET_HEALTH + SIZE_HEALTH;
    private static final int OFFSET_OPPONENT_DIRECTION_VEC = OFFSET_VELOCITY_VEC + SIZE_VELOCITY_VEC;
    private static final int OFFSET_OPPONENT_DISTANCE = OFFSET_OPPONENT_DIRECTION_VEC + SIZE_OPPONENT_DIRECTION_VEC;
    private static final int OFFSET_OPPONENT_VELOCITY_VEC = OFFSET_OPPONENT_DISTANCE + SIZE_OPPONENT_DISTANCE;


    public static final int OBSERVATION_SPACE_SIZE = OFFSET_OPPONENT_VELOCITY_VEC + SIZE_OPPONENT_VELOCITY_VEC;

    final Tensor data;

    public Observation(float pitch, boolean sprinting, boolean sneaking, boolean onGround, float attackCooldown, float health, Vec3 velocity, Vec3 opponentDirectionVec, float opponentDistance, Vec3 opponentVelocityVec) {
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
     * Opponent Velocity Vec:
     * - Shape: (3,)
     */
    public Tensor opponentVelocityVec() {
        return data.narrow(0, OFFSET_OPPONENT_VELOCITY_VEC, SIZE_OPPONENT_VELOCITY_VEC);
    }

    /**
     * Attack Cooldown:
     * - Shape: (1,)
     */
    public Tensor attackCooldown() {
        return data.narrow(0, OFFSET_ATTACK_COOLDOWN, SIZE_ATTACK_COOLDOWN);
    }


    /**
     * @return A tensor representation of the observation.
     */
    public Tensor tensor() {
        return data;
    }

    @Override
    public void close() {
        data.releaseReference();
    }

    public void retainReference() {
        data.retainReference();
    }
}
