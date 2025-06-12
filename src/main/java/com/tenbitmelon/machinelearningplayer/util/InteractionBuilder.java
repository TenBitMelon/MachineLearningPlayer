package com.tenbitmelon.machinelearningplayer.util;

import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.entity.Interaction;

public class InteractionBuilder {

  private final Interaction interaction;

  public InteractionBuilder(World world) {
    interaction = world.spawn(new Location(world, 0, 0, 0), Interaction.class);
  }

  public InteractionBuilder width(float width) {
    interaction.setInteractionWidth(width);
    return this;
  }

  public InteractionBuilder width(double width) {
    return width((float) width);
  }

  public InteractionBuilder height(float height) {
    interaction.setInteractionHeight(height);
    return this;
  }

  public InteractionBuilder height(double height) {
    return height((float) height);
  }

  public InteractionBuilder responsive(boolean responsive) {
    interaction.setResponsive(responsive);
    return this;
  }

  public Interaction build() {
    return interaction;
  }
}
