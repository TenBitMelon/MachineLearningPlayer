package com.tenbitmelon.machinelearningplayer.debugger.ui.controls;

import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.event.ClickEvent;
import org.jetbrains.annotations.NotNull;

public class ButtonControl extends Control {
    private final Runnable callback;

    public ButtonControl(Component title, Runnable action) {
        super(title);
        this.callback = action;
    }

    public @NotNull Component renderLabel() {
        return Component.empty();
    }

    public @NotNull Component renderValue() {
        return Component.text("[")
            .append(label.clickEvent(ClickEvent.callback((audience -> callback.run()))))
            .append(Component.text("]"));
    }
}
