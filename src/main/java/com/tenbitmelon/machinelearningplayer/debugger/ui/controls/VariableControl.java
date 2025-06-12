package com.tenbitmelon.machinelearningplayer.debugger.ui.controls;

import net.kyori.adventure.text.Component;
import org.jetbrains.annotations.NotNull;

import java.util.function.Supplier;

public class VariableControl extends Control {
    private final Supplier<Object> variable;

    public VariableControl(Component title, Supplier<Object> value) {
        super(title);
        this.variable = value;
    }

    @Override
    public @NotNull Component renderValue() {
        return Component.text(variable.get() != null ? variable.get().toString() : "null");
    }
}
