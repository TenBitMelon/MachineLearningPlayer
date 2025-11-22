package com.tenbitmelon.machinelearningplayer.debugger.ui.controls;

import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.event.ClickEvent;
import org.jetbrains.annotations.NotNull;

import java.util.function.Consumer;
import java.util.function.Supplier;

public class BooleanControl extends Control {
    private final Supplier<Boolean> getter;
    private final Consumer<Boolean> setter;

    public BooleanControl(Component title, Supplier<Boolean> value, Consumer<Boolean> setter) {
        super(title);
        this.getter = value;
        this.setter = setter;
    }

    @Override
    public @NotNull Component renderValue() {
        return Component.text(getter.get() ? "[True]" : "[False]")
            .clickEvent(ClickEvent.callback((audience) -> {
                boolean currentValue = getter.get();
                setter.accept(!currentValue);
                // System.out.println("BooleanControl clicked, new value: " + !currentValue);
            }));
    }
}
