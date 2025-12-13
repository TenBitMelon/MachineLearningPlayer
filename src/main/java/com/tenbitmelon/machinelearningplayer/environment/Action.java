package com.tenbitmelon.machinelearningplayer.environment;

import net.minecraft.world.phys.Vec2;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.pytorch.Tensor;

public class Action {

    private static final int SIZE_JUMPING = 1;
    private static final int SIZE_SPRINTING_SNEAKING = 1;
    private static final int SIZE_LOOK_CHANGE = 2;
    private static final int SIZE_FORWARD_MOVE_KEY = 1;
    private static final int SIZE_STRAFE_MOVE_KEY = 1;
    private static final int SIZE_ATTACK_USE_ITEM = 1;

    private static final int OFFSET_JUMPING = 0;
    private static final int OFFSET_SPRINTING_SNEAKING = OFFSET_JUMPING + SIZE_JUMPING;
    private static final int OFFSET_LOOK_CHANGE = OFFSET_SPRINTING_SNEAKING + SIZE_SPRINTING_SNEAKING;
    private static final int OFFSET_FORWARD_MOVE_KEY = OFFSET_LOOK_CHANGE + SIZE_LOOK_CHANGE;
    private static final int OFFSET_STRAFE_MOVE_KEY = OFFSET_FORWARD_MOVE_KEY + SIZE_FORWARD_MOVE_KEY;
    private static final int OFFSET_ATTACK_USE_ITEM = OFFSET_STRAFE_MOVE_KEY + SIZE_STRAFE_MOVE_KEY;

    public static final int ACTION_SPACE_SIZE = OFFSET_ATTACK_USE_ITEM + SIZE_ATTACK_USE_ITEM;

    final Tensor data;
    private FloatPointer cachedData;

    public Action(Tensor data) {
        this.data = data;

        this.cachedData = data
            .data_ptr_float();
    }

    /**
     * Jumping:
     * - Shape: (1,)
     */
    public int jumping() {
        return (int) cachedData.get(OFFSET_JUMPING);
    }

    /**
     * Sprinting Sneaking:
     * - Shape: (1,)
     * 0 = none, 1 = sprinting, 2 = sneaking
     */
    public int sprintingSneaking() {
        return (int) cachedData.get(OFFSET_SPRINTING_SNEAKING);
    }

    /**
     * Look Change:
     * - Shape: (2,)
     */
    public Vec2 lookChange() {
        return new Vec2(
            cachedData.get(OFFSET_LOOK_CHANGE),
            cachedData.get(OFFSET_LOOK_CHANGE + 1)
        );
    }

    /**
     * Forward Move Key:
     * - Shape: (1,)
     * 0 = no movement, 1 = forward, 2 = backward
     */
    public int forwardMoveKey() {
        int val = (int) cachedData.get(OFFSET_FORWARD_MOVE_KEY);
        return val;
    }

    /**
     * Strafe Move Key:
     * - Shape: (1,)
     * 0 = no movement, 1 = left, 2 = right
     */
    public int strafeMoveKey() {
        int val = (int) cachedData.get(OFFSET_STRAFE_MOVE_KEY);
        return val;
    }

    /**
     * Attack & Use Item:
     * - Shape: (1,)
     * 0 = no action, 1 = attack, 2 = use item
     */
    public int attackUseItem() {
        return (int) cachedData.get(OFFSET_ATTACK_USE_ITEM);
    }
}
