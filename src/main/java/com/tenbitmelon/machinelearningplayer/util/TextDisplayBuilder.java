package com.tenbitmelon.machinelearningplayer.util;

import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.TextComponent;
import net.minecraft.world.phys.Vec3;
import org.bukkit.Color;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.entity.Display;
import org.bukkit.entity.TextDisplay;
import org.bukkit.util.Transformation;
import org.bukkit.util.Vector;
import org.jetbrains.annotations.NotNull;
import org.joml.Matrix4f;
import org.joml.Quaternionf;
import org.joml.Vector3f;

public class TextDisplayBuilder {

    private final TextDisplay textDisplay;

    public TextDisplayBuilder(World world) {
        textDisplay = world.spawn(new Location(world, 0, 0, 0), TextDisplay.class);
    }

    public TextDisplayBuilder lineWidth(int lineWidth) {
        textDisplay.setLineWidth(lineWidth);
        return this;
    }

    public TextDisplayBuilder backgroundColor(Color backgroundColor) {
        textDisplay.setBackgroundColor(backgroundColor);
        return this;
    }

    public TextDisplayBuilder textOpacity(byte textOpacity) {
        textDisplay.setTextOpacity(textOpacity);
        return this;
    }

    public TextDisplayBuilder shadowed(boolean shadowed) {
        textDisplay.setShadowed(shadowed);
        return this;
    }

    public TextDisplayBuilder seeThrough(boolean seeThrough) {
        textDisplay.setSeeThrough(seeThrough);
        return this;
    }

    public TextDisplayBuilder hasBackground(boolean hasBackground) {
        textDisplay.setDefaultBackground(hasBackground);
        return this;
    }

    public TextDisplayBuilder alignment(TextDisplay.TextAlignment alignment) {
        textDisplay.setAlignment(alignment);
        return this;
    }

    public TextDisplayBuilder text(String text) {
        return text(Component.text(text));
    }

    public TextDisplayBuilder text(Component text) {
        textDisplay.text(text);
        return this;
    }

    public TextDisplayBuilder transformation(Transformation transformation) {
        textDisplay.setTransformation(transformation);
        return this;
    }

    public TextDisplayBuilder transformationMatrix(Matrix4f transformationMatrix) {
        textDisplay.setTransformationMatrix(transformationMatrix);
        return this;
    }

    public TextDisplayBuilder interpolationDuration(int duration) {
        textDisplay.setInterpolationDuration(duration);
        return this;
    }

    public TextDisplayBuilder teleportDuration(int duration) {
        textDisplay.setTeleportDuration(duration);
        return this;
    }

    public TextDisplayBuilder viewRange(float range) {
        textDisplay.setViewRange(range);
        return this;
    }

    public TextDisplayBuilder shadowRadius(float radius) {
        textDisplay.setShadowRadius(radius);
        return this;
    }

    public TextDisplayBuilder shadowStrength(float strength) {
        textDisplay.setShadowStrength(strength);
        return this;
    }

    public TextDisplayBuilder interpolationDelay(int ticks) {
        textDisplay.setInterpolationDelay(ticks);
        return this;
    }

    public TextDisplayBuilder billboard(Display.Billboard billboard) {
        textDisplay.setBillboard(billboard);
        return this;
    }

    public TextDisplayBuilder glowColorOverride(Color color) {
        textDisplay.setGlowColorOverride(color);
        return this;
    }

    public TextDisplayBuilder brightness(Display.Brightness brightness) {
        textDisplay.setBrightness(brightness);
        return this;
    }

    public TextDisplayBuilder setTranslation(float x, float y, float z) {
        Transformation current = textDisplay.getTransformation();
        Transformation updated = new Transformation(
            new Vector3f(x, y, z),
            current.getLeftRotation(),
            current.getScale(),
            current.getRightRotation()
        );
        textDisplay.setTransformation(updated);
        return this;
    }

    public TextDisplayBuilder setScale(float x, float y, float z) {
        Transformation current = textDisplay.getTransformation();
        Transformation updated = new Transformation(
            current.getTranslation(),
            current.getLeftRotation(),
            new Vector3f(x, y, z),
            current.getRightRotation()
        );
        textDisplay.setTransformation(updated);
        return this;
    }

    public TextDisplayBuilder setRotation(Quaternionf leftRotation, Quaternionf rightRotation) {
        Transformation current = textDisplay.getTransformation();
        Transformation updated = new Transformation(
            current.getTranslation(),
            leftRotation,
            current.getScale(),
            rightRotation
        );
        textDisplay.setTransformation(updated);
        return this;
    }

    // Teleportation
    public TextDisplayBuilder teleport(Location location) {
        textDisplay.teleport(location);
        return this;
    }

    public TextDisplayBuilder teleport(Vec3 location) {
        textDisplay.teleport(new Location(
            textDisplay.getWorld(),
            location.x(),
            location.y(),
            location.z()
        ));
        return this;
    }

    public TextDisplay build() {
        return textDisplay;
    }

    public void addText(String string) {
        textDisplay.text(textDisplay.text().append(Component.newline()).append(Component.text(string)));
    }

    public TextDisplayBuilder teleport(Vector vector) {
        Location location = new Location(
            textDisplay.getWorld(),
            vector.getX(),
            vector.getY(),
            vector.getZ()
        );
        textDisplay.teleport(location);
        return this;
    }

    public void addText(TextComponent textComponent) {
        textDisplay.text(textDisplay.text().append(Component.newline(), textComponent));
    }
}



