package com.tenbitmelon.machinelearningplayer.events;

import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerVelocityEvent;

public class PlayerVEvent implements Listener {

    @EventHandler
    public void onPlayerVelocity(PlayerVelocityEvent e) {
        e.setCancelled(true);
    }
}
