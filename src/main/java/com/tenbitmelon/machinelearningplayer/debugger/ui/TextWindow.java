package com.tenbitmelon.machinelearningplayer.debugger.ui;

import com.tenbitmelon.machinelearningplayer.agent.Agent;
import com.tenbitmelon.machinelearningplayer.debugger.Debugger;
import com.tenbitmelon.machinelearningplayer.util.Utils;
import com.tenbitmelon.machinelearningplayer.util.TextDisplayBuilder;
import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.TextComponent;
import org.bukkit.Location;
import org.bukkit.block.BlockFace;
import org.bukkit.entity.Display;
import org.bukkit.entity.Entity;
import org.bukkit.entity.TextDisplay;
import org.jetbrains.annotations.NotNull;
import org.joml.Matrix4f;
import org.joml.Vector3d;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class TextWindow extends UIElement {

    ArrayList<Component> lines = new ArrayList<>();
    TextDisplay display = new TextDisplayBuilder(Debugger.WORLD).billboard(Display.Billboard.FIXED).alignment(TextDisplay.TextAlignment.LEFT).lineWidth(2000).teleportDuration(1).build();

    public TextWindow() {}

    public TextWindow(Entity anchor) {
        super(anchor);
    }

    public void addLine(String line) {
        lines.add(Component.text(line));
        dirty = true;
    }

    public void addLine(Component line) {
        lines.add(line);
        dirty = true;
    }

    public void setLine(int index, String line) {
        if (index < 0 || index >= lines.size()) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + lines.size());
        }
        lines.set(index, Component.text(line));
        dirty = true;
    }

    public void setLine(int index, Component line) {
        if (index < 0) {
            index = lines.size() + index;
        } else if (index >= lines.size()) {
            while (lines.size() <= index) {
                lines.add(Component.empty());
            }
        }
        lines.set(index, line);
        dirty = true;
    }

    public void update() {
        super.update();
        if (!ALLOW_UPDATES) return;

        if (!dirty) return;
        dirty = false;

        TextComponent.Builder text = Component.text();
        for (Component line : lines) {
            text.append(line);
            text.append(Component.newline());
        }
        display.text(text.build());
    }

    @Override
    public void setVisible(boolean b) {
        super.setVisible(b);
        if (!b) {
            display.setTransformationMatrix(new Matrix4f().scale(0));
        } else {
            display.setTransformationMatrix(new Matrix4f());
        }
    }

    @Override
    public void setPosition(Vector3d position, double direction) {
        super.setPosition(position, direction);
        display.teleport(new Location(Debugger.WORLD, position.x, position.y, position.z));
        display.setRotation((float) direction - 180, 0);
    }

    @Override
    public void destroy() {
        super.destroy();
        display.remove();
    }
}
