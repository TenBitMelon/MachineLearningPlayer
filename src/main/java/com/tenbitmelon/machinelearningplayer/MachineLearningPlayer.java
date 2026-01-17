package com.tenbitmelon.machinelearningplayer;

import com.tenbitmelon.machinelearningplayer.agent.Agent;
import com.tenbitmelon.machinelearningplayer.debugger.Debugger;
import com.tenbitmelon.machinelearningplayer.debugger.Logger;
import com.tenbitmelon.machinelearningplayer.models.EvaluationManager;
import com.tenbitmelon.machinelearningplayer.models.TrainingManager;
import io.papermc.paper.command.brigadier.Commands;
import io.papermc.paper.event.player.PlayerFailMoveEvent;
import io.papermc.paper.plugin.lifecycle.event.types.LifecycleEvents;
import net.kyori.adventure.text.Component;
import net.minecraft.server.level.ServerPlayer;
import org.bukkit.*;
import org.bukkit.craftbukkit.entity.CraftPlayer;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.player.PlayerVelocityEvent;
import org.bukkit.event.world.WorldLoadEvent;
import org.bukkit.plugin.PluginManager;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.scheduler.BukkitRunnable;
import org.bytedeco.cuda.presets.cudnn;
import org.bytedeco.cuda.presets.cupti;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.openblas.presets.openblas;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.presets.torch;
import org.bytedeco.pytorch.presets.torch_cuda;
import org.slf4j.event.Level;

@SuppressWarnings("UnstableApiUsage")
public final class MachineLearningPlayer extends JavaPlugin implements Listener {

    public static Mode CURRENT_MODE = null;
    public static Logger LOGGER = null;

    public static MachineLearningPlayer PLUGIN = null;

    static {
        System.setProperty("org.bytedeco.javacpp.pathsFirst", "true");
        // System.setProperty("org.bytedeco.javacpp.logger.debug", "true");
        // System.setProperty("org.bytedeco.javacpp.logger.level", "debug");
        // Loader.load(org.bytedeco.pytorch.presets.torch_cuda.class);

        // System.setProperty("org.bytedeco.javacpp.logger.debug", "true");
        // System.setProperty("org.bytedeco.javacpp.logger.level", "debug");
        // System.setProperty("org.bytedeco.openblas.load", "mkl");
        //
        Loader.load(javacpp.class);
        Loader.load(openblas.class);
        Loader.load(cupti.class);
        Loader.load(cudnn.class);
        Loader.load(torch.class);
        Loader.load(torch_cuda.class);
    }

    @Override
    public void onEnable() {
        PLUGIN = this;

        // System.out.println("org.bytedeco.openblas.global.openblas.blas_get_num_threads() = " + openblas.blas_get_num_threads());
        //
        // int info = openblas.LAPACKE_dgetrf(openblas.LAPACK_ROW_MAJOR, 2, 2, new double[]{1.0, 2.0, 3.0, 4.0}, 2, new int[]{0, 0});
        // System.out.println("openblas.LAPACKE_dgetrf() info = " + info);
        //
        // System.out.println("torch.cuda_is_available() = " + org.bytedeco.pytorch.global.torch.cuda_is_available());
        // System.out.println("torch.cuda_device_count() = " + org.bytedeco.pytorch.global.torch.cuda_device_count());
        // System.out.println("torch.hasCUDA() = " + torch.hasCUDA());
        // //
        // Device device = new Device("cuda:0");
        // System.out.println("device = " + device);
        // System.out.println("is_cuda = " + device.is_cuda());
        // System.out.println("is_cpu = " + device.is_cpu());
        // System.out.println("type = " + device.type().toString());


        LOGGER = new Logger();
        LOGGER.setEnabled(Level.DEBUG, false);

        PluginManager pluginManager = getServer().getPluginManager();
        pluginManager.registerEvents(this, this);
        pluginManager.registerEvents(Debugger.DebugListener.INSTANCE, this);

        this.getLifecycleManager().registerEventHandler(LifecycleEvents.COMMANDS, event -> {
            final Commands commands = event.registrar();
            commands.register(MachineLearningCommand.register(this), "Run the machine learning command");
        });

        Debugger.start();

        LOGGER.info("Hello, Machine Learning Player is starting!");

        new BukkitRunnable() {
            @Override
            public void run() {
                Debugger.update();
                EvaluationManager.evaluationStep();
                TrainingManager.trainingStep();
            }
        }.runTaskTimer(this, 0, 1);
    }

    @Override
    public void onDisable() {
        Debugger.stop();
        TrainingManager.shutdown();
        EvaluationManager.shutdown();

        for (World world : Bukkit.getWorlds()) {
            for (Entity entity : world.getEntities()) {
                if (entity instanceof Player player) {
                    if (player.getGameMode() == GameMode.CREATIVE) {
                        player.teleport(new Location(world, 16, 5, 20, 180, 0));
                    } else {
                        player.kick();
                    }
                } else {
                    entity.remove();
                }
            }
        }
    }

    @EventHandler
    public void onPlayerJoin(PlayerJoinEvent event) {
        event.getPlayer().sendMessage(Component.text("Hello, " + event.getPlayer().getName() + "!"));
    }

    @EventHandler
    public void onPlayerFailMove(PlayerFailMoveEvent event) {
        event.setAllowed(true);
    }

    @EventHandler
    public void onPlayerVelocity(PlayerVelocityEvent e) {
        ServerPlayer handle = ((CraftPlayer) e.getPlayer()).getHandle();
        if (handle instanceof Agent) {
            e.setCancelled(true);
        }
    }

    @EventHandler
    public void onWorldLoad(WorldLoadEvent event) {
        World world = event.getWorld();
        world.setTime(0);
        world.setGameRule(GameRule.DO_DAYLIGHT_CYCLE, false);
        world.setWeatherDuration(0);
        world.setStorm(false);
        world.setGameRule(GameRule.DO_WEATHER_CYCLE, false);
        world.setGameRule(GameRule.DO_MOB_SPAWNING, false);
        world.setGameRule(GameRule.RANDOM_TICK_SPEED, 0);
        world.setGameRule(GameRule.DO_ENTITY_DROPS, false);
        world.setGameRule(GameRule.DO_TILE_DROPS, false);
        world.setGameRule(GameRule.DO_INSOMNIA, false);
        world.setGameRule(GameRule.DO_IMMEDIATE_RESPAWN, true);

        for (Entity entity : world.getEntities()) {
            if (entity instanceof Player player) {
                if (player.getGameMode() == GameMode.CREATIVE) {
                    player.teleport(new Location(world, 16, 5, 20, 180, 0));
                } else {
                    player.kick();
                }
            } else {
                entity.remove();
            }
        }
    }

    public enum Mode {
        TRAINING,
        EVALUATION
    }

}
