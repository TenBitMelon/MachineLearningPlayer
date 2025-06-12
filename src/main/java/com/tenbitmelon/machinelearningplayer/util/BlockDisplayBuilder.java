package com.tenbitmelon.machinelearningplayer.util;

import org.bukkit.Color;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.block.data.BlockData;
import org.bukkit.entity.BlockDisplay;
import org.bukkit.entity.Display;
import org.bukkit.entity.TextDisplay;
import org.bukkit.util.Transformation;
import org.joml.Matrix4f;

public class BlockDisplayBuilder {

  private final BlockDisplay blockDisplay;

  public BlockDisplayBuilder(World world) {
    blockDisplay = world.spawn(new Location(world, 0, 0, 0), BlockDisplay.class);
  }

  public BlockDisplayBuilder block(BlockData block) {
    blockDisplay.setBlock(block);
    return this;
  }

  public BlockDisplayBuilder transformation(Transformation transformation) {
    blockDisplay.setTransformation(transformation);
    return this;
  }

  public BlockDisplayBuilder transformationMatrix(Matrix4f transformationMatrix) {
    blockDisplay.setTransformationMatrix(transformationMatrix);
    return this;
  }

  public BlockDisplayBuilder interpolationDuration(int duration) {
    blockDisplay.setInterpolationDuration(duration);
    return this;
  }

  public BlockDisplayBuilder teleportDuration(int duration) {
    blockDisplay.setTeleportDuration(duration);
    return this;
  }

  public BlockDisplayBuilder viewRange(float range) {
    blockDisplay.setViewRange(range);
    return this;
  }

  public BlockDisplayBuilder shadowRadius(float radius) {
    blockDisplay.setShadowRadius(radius);
    return this;
  }

  public BlockDisplayBuilder shadowStrength(float strength) {
    blockDisplay.setShadowStrength(strength);
    return this;
  }

  public BlockDisplayBuilder displayWidth(float width) {
    blockDisplay.setDisplayWidth(width);
    return this;
  }

  public BlockDisplayBuilder displayHeight(float height) {
    blockDisplay.setDisplayHeight(height);
    return this;
  }

  public BlockDisplayBuilder interpolationDelay(int ticks) {
    blockDisplay.setInterpolationDelay(ticks);
    return this;
  }

  public BlockDisplayBuilder billboard(Display.Billboard billboard) {
    blockDisplay.setBillboard(billboard);
    return this;
  }

  public BlockDisplayBuilder glowColorOverride(Color color) {
    blockDisplay.setGlowColorOverride(color);
    return this;
  }

  public BlockDisplayBuilder brightness(Display.Brightness brightness) {
    blockDisplay.setBrightness(brightness);
    return this;
  }

  public BlockDisplay build() {
    return blockDisplay;
  }
}
