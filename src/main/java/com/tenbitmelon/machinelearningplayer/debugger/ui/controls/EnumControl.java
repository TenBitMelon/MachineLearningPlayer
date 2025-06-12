package com.tenbitmelon.machinelearningplayer.debugger.ui.controls;

import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.event.ClickEvent;
import org.jetbrains.annotations.NotNull;

import java.util.function.Consumer;
import java.util.function.Supplier;

public class EnumControl<T extends Enum<T>> extends Control {
    private final Supplier<T> getter;
    private final Consumer<T> setter;

    public EnumControl(Component title, Supplier<T> value, Consumer<T> setter) {
        super(title);
        this.getter = value;
        this.setter = setter;
    }

    public @NotNull Component renderValue() {
        return Component.text(getter.get().name()).clickEvent(ClickEvent.callback((audience -> {
            T currentValue = getter.get();
            T[] enumConstants = currentValue.getDeclaringClass().getEnumConstants();
            int nextOrdinal = (currentValue.ordinal() + 1) % enumConstants.length;
            setter.accept(enumConstants[nextOrdinal]);
        })));
    }
}
