package com.tenbitmelon.machinelearningplayer.debugger.ui;

import com.tenbitmelon.machinelearningplayer.debugger.Debugger;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.Control;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.TextControl;
import com.tenbitmelon.machinelearningplayer.util.TextDisplayBuilder;
import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.TextComponent;
import org.bukkit.Location;
import org.bukkit.entity.Display;
import org.bukkit.entity.TextDisplay;
import org.bukkit.event.player.PlayerInteractEntityEvent;
import org.bukkit.util.Transformation;
import org.joml.AxisAngle4f;
import org.joml.Matrix4f;
import org.joml.Vector3d;
import org.joml.Vector3f;

import java.util.ArrayList;
import java.util.Iterator;

public class ControlsWindow extends UIElement {

    final TextDisplay display = new TextDisplayBuilder(Debugger.WORLD)
        .billboard(Display.Billboard.FIXED)
        .alignment(TextDisplay.TextAlignment.LEFT)
        .lineWidth(2000)
        .teleportDuration(1)
        .build();
    final TextDisplay displayBack = new TextDisplayBuilder(Debugger.WORLD)
        .billboard(Display.Billboard.FIXED)
        .alignment(TextDisplay.TextAlignment.LEFT)
        .lineWidth(2000)
        .teleportDuration(1)
        .textOpacity((byte) 4)
        .transformation(new Transformation(new Vector3f(1), new AxisAngle4f(0, 0, 0, 0), new Vector3f(1), new AxisAngle4f(0, 0, 0, 0)))
        .transformationMatrix(new Matrix4f().scale(-1, 1, 1))
        .build();
    private final ArrayList<Control> controls = new ArrayList<>();

    public ControlsWindow() {
        super();
    }

    public void addControl(Control control) {
        controls.add(control);
        dirty = true;
    }

    public void addText(String text) {
        controls.add(new TextControl(Component.text(text)));
        dirty = true;
    }

    public void addText(String text, String value) {
        controls.add(new TextControl(Component.text(text), Component.text(value)));
        dirty = true;
    }

    public void removeControl(Control control) {
        control.remove();
        controls.remove(control);
        dirty = true;
    }

    public void refresh() {
        if (!ALLOW_UPDATES) return;
        int longestLine = 0;
        int longestValue = 0;
        for (Control control : controls) {
            longestLine = Math.max(longestLine, control.getLabelWidth());
            longestValue = Math.max(longestValue, control.getValueWidth());
        }
        longestLine += 4;

        TextComponent.Builder builder = Component.text();
        int line = controls.size();
        for (Control control : controls) {
            Component component = control.placeAndRender(longestLine, longestValue, line, position, direction);
            line--;
            builder.append(component);
            builder.appendNewline();
        }

        display.text(builder.build());
        displayBack.text(builder.build());
    }

    public void update() {
        super.update();
        if (!ALLOW_UPDATES) return;

        if (!dirty) return;
        dirty = false;

        refresh();
    }

    @Override
    public void destroy() {
        super.destroy();
        display.remove();
        displayBack.remove();
    }

    @Override
    public boolean onInteract(PlayerInteractEntityEvent event) {
        if (super.onInteract(event)) return true;

        boolean handled = false;
        Iterator<Control> iterator = controls.iterator();
        while (iterator.hasNext()) {
            Control control = iterator.next();
            handled |= control.onInteract(event);
        }
        if (handled) {
            dirty = true; // Mark the window as dirty to update the display
        }
        return handled;
    }

    @Override
    public void setPosition(Vector3d position, double direction) {
        super.setPosition(position, direction);

        display.teleport(new Location(Debugger.WORLD, position.x, position.y, position.z));
        display.setRotation((float) direction - 180, 0);
        displayBack.teleport(new Location(Debugger.WORLD, position.x, position.y, position.z));
        displayBack.setRotation((float) direction - 180, 0);

        dirty = true;
    }

}
