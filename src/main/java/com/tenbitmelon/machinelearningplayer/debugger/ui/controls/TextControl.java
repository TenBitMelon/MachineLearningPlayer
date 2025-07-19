package com.tenbitmelon.machinelearningplayer.debugger.ui.controls;

import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.TextComponent;
import org.jetbrains.annotations.NotNull;

public class TextControl extends Control {
    public TextComponent value;

    public TextControl(TextComponent text) {
        super(text);
        this.value = Component.empty();
    }

    public TextControl(TextComponent text, TextComponent value) {
        super(text);
        this.value = value;
    }

    public TextControl(String text) {
        this(Component.text(text));
    }

    public TextControl(String text, String value) {
        this(Component.text(text), Component.text(value));
    }

    @Override
    public @NotNull Component renderValue() {
        return value;
    }

    public void setText(String text) {
        this.label = Component.text(text);
    }

    public void setValue(String value) {
        this.value = Component.text(value);
    }
}
