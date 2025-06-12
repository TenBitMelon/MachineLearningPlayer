package com.tenbitmelon.machinelearningplayer.events;

import com.tenbitmelon.machinelearningplayer.agent.Agent;
import io.papermc.paper.event.player.PlayerFailMoveEvent;
import org.bukkit.entity.Entity;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerInteractEntityEvent;
import org.bukkit.event.player.PlayerMoveEvent;

public class EntityInteractEvent implements Listener {

    @EventHandler
    public void onEntityInteract(PlayerInteractEntityEvent event) {

    }

    @EventHandler
    public void onPlayerMove(PlayerMoveEvent event) {
        // event.getPlayer().sendMessage(event.getPlayer().getLocation().getDirection().toString());
        // event.getPlayer().sendMessage(event.getPlayer().getVelocity().toString());
    }

    @EventHandler
    public void onPlayerFailMove(PlayerFailMoveEvent event) {
        event.setAllowed(true);
    }
}
