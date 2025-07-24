package com.tenbitmelon.machinelearningplayer.debugger.ui;

import com.tenbitmelon.machinelearningplayer.debugger.Debugger;
import com.tenbitmelon.machinelearningplayer.util.TextDisplayBuilder;
import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.TextComponent;
import org.bukkit.Location;
import org.bukkit.entity.Display;
import org.bukkit.entity.Entity;
import org.bukkit.entity.TextDisplay;
import org.joml.Matrix4f;
import org.joml.Vector3d;

import java.util.ArrayList;

public class TextWindow extends UIElement {

    final ArrayList<Component> lines = new ArrayList<>();
    final TextDisplay display;
    int maxLines = 0;

    public TextWindow() {
        this(Display.Billboard.FIXED, TextDisplay.TextAlignment.CENTER);
    }

    public TextWindow(Display.Billboard billboard, TextDisplay.TextAlignment alignment) {
        display = new TextDisplayBuilder(Debugger.WORLD)
            .billboard(billboard)
            .alignment(alignment)
            .lineWidth(2000)
            .teleportDuration(1)
            .build();
    }

    public void addLine(String line) {
        addLine(Component.text(line));
    }

    public void addLine(Component line) {
        if (maxLines != 0 && lines.size() >= maxLines) {
            lines.removeFirst();
        }
        lines.add(line);
        dirty = true;
    }

    public void clearLines() {
        lines.clear();
        dirty = true;
    }

    public void setMaxLines(int maxLines) {
        this.maxLines = maxLines;
        if (maxLines > 0) {
            while (lines.size() > maxLines) {
                lines.removeLast();
            }
        }
        dirty = true;
    }

    public void setLine(int index, String line) {
        setLine(index, Component.text(line));
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
