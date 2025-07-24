package com.tenbitmelon.machinelearningplayer.environment;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

public class Observation {

    private static final int SIZE_VOXEL_GRID = MinecraftEnvironment.GRID_VOLUME;
    private static final int SIZE_POSITION_IN_BLOCK = 3;
    private static final int SIZE_VELOCITY = 3;
    private static final int SIZE_LOOK_DIRECTION = 3;
    private static final int SIZE_JUMPING = 1;
    private static final int SIZE_SPRINTING = 1;
    private static final int SIZE_SNEAKING = 1;
    private static final int SIZE_ON_GROUND = 1;
    private static final int SIZE_GOAL_DIRECTION = 3;

    private static final int OFFSET_VOXEL_GRID = 0;
    private static final int OFFSET_POSITION_IN_BLOCK = OFFSET_VOXEL_GRID + SIZE_VOXEL_GRID;
    private static final int OFFSET_VELOCITY = OFFSET_POSITION_IN_BLOCK + SIZE_POSITION_IN_BLOCK;
    private static final int OFFSET_LOOK_DIRECTION = OFFSET_VELOCITY + SIZE_VELOCITY;
    private static final int OFFSET_JUMPING = OFFSET_LOOK_DIRECTION + SIZE_LOOK_DIRECTION;
    private static final int OFFSET_SPRINTING = OFFSET_JUMPING + SIZE_JUMPING;
    private static final int OFFSET_SNEAKING = OFFSET_SPRINTING + SIZE_SPRINTING;
    private static final int OFFSET_ON_GROUND = OFFSET_SNEAKING + SIZE_SNEAKING;
    private static final int OFFSET_GOAL_DIRECTION = OFFSET_ON_GROUND + SIZE_ON_GROUND;

    public static final int OBSERVATION_SPACE_SIZE = OFFSET_GOAL_DIRECTION + SIZE_GOAL_DIRECTION;

    final Tensor data;

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
     * Voxel Grid:
     * - Shape: (GRID_SIZE_XZ, GRID_SIZE_XZ, GRID_SIZE_Y)
     */
    public Tensor voxelGrid() {
        return data.narrow(0, OFFSET_VOXEL_GRID, SIZE_VOXEL_GRID);
    }

    /**
     * Position in Block:
     * - Shape: (3,)
     */
    public Tensor positionInBlock() {
        return data.narrow(0, OFFSET_POSITION_IN_BLOCK, SIZE_POSITION_IN_BLOCK);
    }

    /**
     * Velocity:
     * - Shape: (3,)
     */
    public Tensor velocity() {
        return data.narrow(0, OFFSET_VELOCITY, SIZE_VELOCITY);
    }

    /**
     * Look Direction:
     * - Shape: (3,)
     */
    public Tensor lookDirection() {
        return data.narrow(0, OFFSET_LOOK_DIRECTION, SIZE_LOOK_DIRECTION);
    }

    /**
     * Jumping:
     * - Shape: (1,)
     */
    public Tensor jumping() {
        return data.narrow(0, OFFSET_JUMPING, SIZE_JUMPING);
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
     * Goal Direction:
     * - Shape: (3,)
     */
    public Tensor goalDirection() {
        return data.narrow(0, OFFSET_GOAL_DIRECTION, SIZE_GOAL_DIRECTION);
    }


    /**
     * Converts the observation to a tensor.
     * This is a placeholder implementation and should be replaced with actual logic.
     *
     * @return A tensor representation of the observation.
     */
    public Tensor toTensor() {
        return data;
    }

    public Tensor nonVoxelGridData() {
        return data.narrow(0, OFFSET_POSITION_IN_BLOCK, OBSERVATION_SPACE_SIZE - OFFSET_POSITION_IN_BLOCK);
    }
}
