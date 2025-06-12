package com.tenbitmelon.machinelearningplayer.debugger.ui.controls;

import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.TextComponent;
import org.jetbrains.annotations.NotNull;

public class TextControl extends Control {
    public TextControl(TextComponent text) {
        super(text);
    }

    @Override
    public @NotNull Component renderValue() {
        return Component.empty();
    }
}
