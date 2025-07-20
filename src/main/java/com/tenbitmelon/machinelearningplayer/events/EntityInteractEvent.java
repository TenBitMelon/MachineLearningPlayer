package com.tenbitmelon.machinelearningplayer.events;

import io.papermc.paper.event.player.PlayerFailMoveEvent;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;

public class EntityInteractEvent implements Listener {

    @EventHandler
    public void onPlayerFailMove(PlayerFailMoveEvent event) {
        event.setAllowed(true);
    }
}
