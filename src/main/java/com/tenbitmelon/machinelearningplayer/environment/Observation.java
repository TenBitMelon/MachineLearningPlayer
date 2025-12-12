package com.tenbitmelon.machinelearningplayer.environment;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

public class Observation {

    // public static final int SIZE_VOXEL_GRID = MinecraftEnvironment.GRID_VOLUME;
    // public static final int SIZE_POSITION_IN_BLOCK = 3;
    // public static final int SIZE_VELOCITY = 3;
    // public static final int SIZE_YAW = 2;
    // public static final int SIZE_PITCH = 2;
    // public static final int SIZE_JUMPING = 1;
    // public static final int SIZE_SPRINTING = 1;
    // public static final int SIZE_SNEAKING = 1;
    // public static final int SIZE_ON_GROUND = 1;
    public static final int SIZE_GOAL_DIRECTION = 3;
    public static final int SIZE_GOAL_DISTANCE = 1;

    // public static final int OFFSET_VOXEL_GRID = 0;
    // public static final int OFFSET_POSITION_IN_BLOCK = OFFSET_VOXEL_GRID + SIZE_VOXEL_GRID;
    // public static final int OFFSET_VELOCITY = OFFSET_POSITION_IN_BLOCK + SIZE_POSITION_IN_BLOCK;
    // public static final int OFFSET_YAW = OFFSET_VELOCITY + SIZE_VELOCITY;
    // public static final int OFFSET_PITCH = OFFSET_YAW + SIZE_YAW;
    // public static final int OFFSET_JUMPING = OFFSET_PITCH + SIZE_PITCH;
    // public static final int OFFSET_SPRINTING = OFFSET_JUMPING + SIZE_JUMPING;
    // public static final int OFFSET_SNEAKING = OFFSET_SPRINTING + SIZE_SPRINTING;
    // public static final int OFFSET_ON_GROUND = OFFSET_SNEAKING + SIZE_SNEAKING;
    // public static final int OFFSET_GOAL_DIRECTION = OFFSET_ON_GROUND + SIZE_ON_GROUND;
    public static final int OFFSET_GOAL_DIRECTION = 0;
    public static final int OFFSET_GOAL_DISTANCE = OFFSET_GOAL_DIRECTION + SIZE_GOAL_DIRECTION;

    public static final int OBSERVATION_SPACE_SIZE = OFFSET_GOAL_DISTANCE + SIZE_GOAL_DISTANCE;

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

    /// /**
    ///  * Voxel Grid:
    ///  * - Shape: (GRID_SIZE_XZ, GRID_SIZE_XZ, GRID_SIZE_Y)
    ///  */
    /// public Tensor voxelGrid() {
    ///     return data.narrow(0, OFFSET_VOXEL_GRID, SIZE_VOXEL_GRID);
    /// }

    /// /**
    ///  * Position in Block:
    ///  * - Shape: (3,)
    ///  */
    /// public Tensor positionInBlock() {
    ///     return data.narrow(0, OFFSET_POSITION_IN_BLOCK, SIZE_POSITION_IN_BLOCK);
    /// }

    /// /**
    ///  * Velocity:
    ///  * - Shape: (3,)
    ///  */
    /// public Tensor velocity() {
    ///     return data.narrow(0, OFFSET_VELOCITY, SIZE_VELOCITY);
    /// }

    /// /**
    ///  * Yaw:
    ///  * - Shape: (2,)
    ///  */
    /// public Tensor yaw() {
    ///     return data.narrow(0, OFFSET_YAW, SIZE_YAW);
    /// }

    /// /**
    ///  * Pitch:
    ///  * - Shape: (2,)
    ///  */
    /// public Tensor pitch() {
    ///     return data.narrow(0, OFFSET_PITCH, SIZE_PITCH);
    /// }

    /// /**
    ///  * Jumping:
    ///  * - Shape: (1,)
    ///  */
    /// public Tensor jumping() {
    ///     return data.narrow(0, OFFSET_JUMPING, SIZE_JUMPING);
    /// }

    /// /**
    ///  * Sprinting:
    ///  * - Shape: (1,)
    ///  */
    /// public Tensor sprinting() {
    ///     return data.narrow(0, OFFSET_SPRINTING, SIZE_SPRINTING);
    /// }

    /// /**
    ///  * Sneaking:
    ///  * - Shape: (1,)
    ///  */
    /// public Tensor sneaking() {
    ///     return data.narrow(0, OFFSET_SNEAKING, SIZE_SNEAKING);
    /// }

    /// /**
    ///  * On Ground:
    ///  * - Shape: (1,)
    ///  */
    /// public Tensor onGround() {
    ///     return data.narrow(0, OFFSET_ON_GROUND, SIZE_ON_GROUND);
    /// }

    /**
     * Goal Direction:
     * - Shape: (3,)
     */
    public Tensor goalDirection() {
        return data.narrow(0, OFFSET_GOAL_DIRECTION, SIZE_GOAL_DIRECTION);
    }

    /**
     * Goal Distance:
     * - Shape: (1,)
     */
    public Tensor goalDistance() {
        return data.narrow(0, OFFSET_GOAL_DISTANCE, SIZE_GOAL_DISTANCE);
    }


    /**
     * @return A tensor representation of the observation.
     */
    public Tensor tensor() {
        return data;
    }
}
