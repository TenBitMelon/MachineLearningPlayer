package com.tenbitmelon.machinelearningplayer.environment;

import net.minecraft.world.phys.Vec2;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.pytorch.Tensor;

public class Action {
    // double[] lookChange,
    // int sprintKey,
    // int jumpKey,
    // int sneakKey,
    // double[] moveKeys,

    /// private static final int SIZE_JUMPING = 1;
    /// private static final int SIZE_SPRINTING = 1;
    /// private static final int SIZE_SNEAKING = 1;
    private static final int SIZE_LOOK_CHANGE = 2;
    private static final int SIZE_FORWARD_MOVE_KEY = 1;
    private static final int SIZE_STRAFE_MOVE_KEY = 1;

    /// private static final int OFFSET_JUMPING = 0;
    /// private static final int OFFSET_SPRINTING = OFFSET_JUMPING + SIZE_JUMPING;
    /// private static final int OFFSET_SNEAKING = OFFSET_SPRINTING + SIZE_SPRINTING;
    /// private static final int OFFSET_LOOK_CHANGE = OFFSET_SNEAKING + SIZE_SNEAKING;
    private static final int OFFSET_LOOK_CHANGE = 0;
    private static final int OFFSET_FORWARD_MOVE_KEY = OFFSET_LOOK_CHANGE + SIZE_LOOK_CHANGE;
    private static final int OFFSET_STRAFE_MOVE_KEY = OFFSET_FORWARD_MOVE_KEY + SIZE_FORWARD_MOVE_KEY;

    public static final int ACTION_SPACE_SIZE = OFFSET_STRAFE_MOVE_KEY + SIZE_STRAFE_MOVE_KEY;

    final Tensor data;
    private FloatPointer cachedData;

    public Action(Tensor data) {
        this.data = data;

        this.cachedData = data
            .data_ptr_float();
    }

    /// /**
    ///  * Jumping:
    ///  * - Shape: (1,)
    ///  */
    /// public int jumping() {
    ///     // return data.narrow(1, OFFSET_JUMPING, SIZE_JUMPING);
    ///     // return data.get(OFFSET_JUMPING).item_int();
    ///     return (int) cachedData.get(OFFSET_JUMPING);
    /// }
    ///
    /// /**
    ///  * Sprinting:
    ///  * - Shape: (1,)
    ///  */
    /// public int sprinting() {
    ///     // return data.narrow(1, OFFSET_SPRINTING, SIZE_SPRINTING);
    ///     // return data.get(OFFSET_SPRINTING).item_int();
    ///     return (int) cachedData.get(OFFSET_SPRINTING);
    /// }
    ///
    /// /**
    ///  * Sneaking:
    ///  * - Shape: (1,)
    ///  */
    /// public int sneaking() {
    ///     // return data.narrow(1, OFFSET_SNEAKING, SIZE_SNEAKING);
    ///     // return data.get(OFFSET_SNEAKING).item_int();
    ///     return (int) cachedData.get(OFFSET_SNEAKING);
    /// }

    /**
     * Look Change:
     * - Shape: (2,)
     */
    public Vec2 lookChange() {
        // return data.narrow(1, OFFSET_LOOK_CHANGE, SIZE_LOOK_CHANGE);
        // return new Vec2(
        //     (float) data.get(OFFSET_LOOK_CHANGE).item_double(),
        //     (float) data.get(OFFSET_LOOK_CHANGE + 1).item_double()
        // );
        return new Vec2(
            (float) cachedData.get(OFFSET_LOOK_CHANGE),
            (float) cachedData.get(OFFSET_LOOK_CHANGE + 1)
        );
    }

    /**
     * Forward Move Key:
     * - Shape: (1,)
     * 0 = no movement, 1 = forward, 2 = backward
     */
    public int forwardMoveKey() {
        // return data.narrow(1, OFFSET_FORWARD_MOVE_KEY, SIZE_FORWARD_MOVE_KEY);
        // return data.get(OFFSET_FORWARD_MOVE_KEY).item_int();
        int val = (int) cachedData.get(OFFSET_FORWARD_MOVE_KEY);
        return val;
    }

    /**
     * Strafe Move Key:
     * - Shape: (1,)
     * 0 = no movement, 1 = left, 2 = right
     */
    public int strafeMoveKey() {
        // return data.narrow(1, OFFSET_STRAFE_MOVE_KEY, SIZE_STRAFE_MOVE_KEY);
        // return data.get(OFFSET_STRAFE_MOVE_KEY).item_int();
        int val = (int) cachedData.get(OFFSET_STRAFE_MOVE_KEY);
        return val;
    }
}
