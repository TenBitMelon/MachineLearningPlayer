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

    private static final int SIZE_JUMPING = 1;
    private static final int SIZE_SPRINTING = 1;
    private static final int SIZE_SNEAKING = 1;
    private static final int SIZE_LOOK_CHANGE = 2;
    private static final int SIZE_MOVE_KEYS = 4;

    private static final int OFFSET_JUMPING = 0;
    private static final int OFFSET_SPRINTING = OFFSET_JUMPING + SIZE_JUMPING;
    private static final int OFFSET_SNEAKING = OFFSET_SPRINTING + SIZE_SPRINTING;
    private static final int OFFSET_LOOK_CHANGE = OFFSET_SNEAKING + SIZE_SNEAKING;
    private static final int OFFSET_MOVE_KEYS = OFFSET_LOOK_CHANGE + SIZE_LOOK_CHANGE;

    public static final int ACTION_SPACE_SIZE = OFFSET_MOVE_KEYS + SIZE_MOVE_KEYS;

    final Tensor data;
    private FloatPointer cachedData;

    public Action(Tensor data) {
        this.data = data;
        this.cachedData = data
            .cpu()
            .data_ptr_float();
    }

    /**
     * Jumping:
     * - Shape: (1,)
     */
    public int jumping() {
        // return data.narrow(1, OFFSET_JUMPING, SIZE_JUMPING);
        // return data.get(OFFSET_JUMPING).item_int();
        return (int) cachedData.get(OFFSET_JUMPING);
    }

    /**
     * Sprinting:
     * - Shape: (1,)
     */
    public int sprinting() {
        // return data.narrow(1, OFFSET_SPRINTING, SIZE_SPRINTING);
        // return data.get(OFFSET_SPRINTING).item_int();
        return (int) cachedData.get(OFFSET_SPRINTING);
    }

    /**
     * Sneaking:
     * - Shape: (1,)
     */
    public int sneaking() {
        // return data.narrow(1, OFFSET_SNEAKING, SIZE_SNEAKING);
        // return data.get(OFFSET_SNEAKING).item_int();
        return (int) cachedData.get(OFFSET_SNEAKING);
    }

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
     * Move Keys:
     * - Shape: (4,)
     */
    public MovementKeys moveKeys() {
        // return data.narrow(1, OFFSET_MOVE_KEYS, SIZE_MOVE_KEYS);
        // return new MovementKeys(
        //     data.get(OFFSET_MOVE_KEYS).item_int(),
        //     data.get(OFFSET_MOVE_KEYS + 1).item_int(),
        //     data.get(OFFSET_MOVE_KEYS + 2).item_int(),
        //     data.get(OFFSET_MOVE_KEYS + 3).item_int()
        // );
        return new MovementKeys(
            (int) cachedData.get(OFFSET_MOVE_KEYS),
            (int) cachedData.get(OFFSET_MOVE_KEYS + 1),
            (int) cachedData.get(OFFSET_MOVE_KEYS + 2),
            (int) cachedData.get(OFFSET_MOVE_KEYS + 3)
        );
    }

    public record MovementKeys(int forward, int backward, int left, int right) {
        @Override
        public String toString() {
            return "MovementKeys{" +
                "forward=" + forward +
                ", backward=" + backward +
                ", left=" + left +
                ", right=" + right +
                '}';
        }
    }
}
