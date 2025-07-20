package com.tenbitmelon.machinelearningplayer.debugger.ui.controls;

import com.tenbitmelon.machinelearningplayer.debugger.Debugger;
import com.tenbitmelon.machinelearningplayer.util.BlockDisplayBuilder;
import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.event.ClickEvent;
import org.bukkit.Material;
import org.bukkit.entity.BlockDisplay;
import org.jetbrains.annotations.NotNull;
import org.joml.Matrix4f;

import java.util.function.Consumer;
import java.util.function.Supplier;

public class CounterControl extends Control {
    final BlockDisplay blockDisplay;
    private final Supplier<Integer> getter;
    private final Consumer<Integer> setter;

    public CounterControl(Component title, Supplier<Integer> value, Consumer<Integer> setter) {
        super(title);
        this.getter = value;
        this.setter = setter;
        blockDisplay = new BlockDisplayBuilder(Debugger.WORLD).block(Material.MELON.createBlockData()).build();
        blockDisplay.setTransformationMatrix(new Matrix4f().scale(0.2f));
    }

    public @NotNull Component renderValue() {
        return Component.text("←").clickEvent(ClickEvent.callback((audience -> {
                int currentValue = getter.get();
                setter.accept(currentValue - 1);
            })))
            .append(Component.text(getter.get().toString()))
            .append(Component.text("→")
                .clickEvent(ClickEvent.callback((audience -> {
                    int currentValue = getter.get();
                    setter.accept(currentValue + 1);
                }))));
    }
}
