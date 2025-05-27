package com.tenbitmelon.machinelearningplayer;

import com.mojang.brigadier.Command;
import com.mojang.brigadier.builder.LiteralArgumentBuilder;
import com.mojang.brigadier.context.CommandContext;
import com.mojang.brigadier.tree.LiteralCommandNode;
import io.papermc.paper.command.brigadier.CommandSourceStack;
import io.papermc.paper.command.brigadier.Commands;
import org.bukkit.*;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;

@SuppressWarnings("UnstableApiUsage")
public class MachineLearningCommand {

    public static LiteralCommandNode<CommandSourceStack> register() {
        LiteralArgumentBuilder<CommandSourceStack> commandBuilder = Commands.literal("ml")
                .executes(ctx -> {
                    ctx.getSource().getSender().sendPlainMessage("[STATS]");
                    return Command.SINGLE_SUCCESS;
                });

        commandBuilder.then(Commands.literal("setup")
            .executes(ctx -> {
                Location location = ctx.getSource().getLocation();
                World world = location.getWorld();

                world.setTime(0);
                world.setGameRule(GameRule.DO_DAYLIGHT_CYCLE, false);
                world.setWeatherDuration(0);
                world.setStorm(false);
                world.setGameRule(GameRule.DO_WEATHER_CYCLE, false);
                world.setGameRule(GameRule.DO_MOB_SPAWNING, false);

                for (Entity entity : world.getEntities()) {
                    if (entity instanceof Player) {
                        entity.teleport(new Location(world, 0, 1, 0));
                        continue;
                    }
                    entity.remove();
                }

                createRooms(ctx, world, 16);
                return Command.SINGLE_SUCCESS;
            }));

        commandBuilder.then(Commands.literal("train")
            .executes(ctx -> {
                ctx.getSource().getSender().sendPlainMessage("[TRAIN] Training started.");
                // Here you would add the logic to start training your model.
                return Command.SINGLE_SUCCESS;
            }).then(Commands.literal("stop")
                .executes(ctx -> {
                    ctx.getSource().getSender().sendPlainMessage("[TRAIN] Training stopped.");
                    // Here you would add the logic to stop training your model.
                    return Command.SINGLE_SUCCESS;
                })));

        return commandBuilder.build();
    }

    public static void createRooms(CommandContext<CommandSourceStack> ctx, World world, int count) {
        Location location = new Location(world, 0, 0, 0);
        int gridWidth = (int) Math.ceil(Math.sqrt(count));

        for (int i = 0; i < count; i++) {
            int offsetX = (i % gridWidth) * 16;
            int offsetZ = (i / gridWidth) * 16;

            Location roomLocation = location.clone().add(offsetX, 0, offsetZ);
            createRoom(ctx, roomLocation);
        }
    }

    public static void createRoom(CommandContext<CommandSourceStack> ctx, Location location) {
        Chunk chunk = location.getChunk();
        World world = location.getWorld();
        chunk.load();

        int startX = chunk.getX() * 16;
        int startZ = chunk.getZ() * 16;

        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                for (int y = -16; y < 64; y++) {
                    world.getBlockAt(startX + x, y, startZ + z).setType(Material.AIR);
                }
            }
        }

        for (int offsetX = 0; offsetX < 16; offsetX++) {
            for (int offsetZ = 0; offsetZ < 16; offsetZ++) {

                int worldX = startX + offsetX;
                int worldZ = startZ + offsetZ;

                world.getBlockAt(worldX, 0, worldZ).setType(Material.BARRIER);

                if (offsetX > 4 && offsetX < 12 && offsetZ > 4 && offsetZ < 12) {
                    world.getBlockAt(worldX, 0, worldZ).setType(Material.WHITE_CONCRETE);
                }

            }
        }
        world.getBlockAt(startX + 8, 0, startZ + 8).setType(Material.GOLD_BLOCK);
        world.getBlockAt(startX + 8, -1, startZ + 8).setType(Material.BEACON);
        for (int x = -1; x <= 1; x++) {
            for (int z = -1; z <= 1; z++) {
                world.getBlockAt(startX + 8 + x, -2, startZ + 8 + z).setType(Material.IRON_BLOCK);
            }
        }

        // world.spawn(new Location(location.getWorld(), startX + 8.5, 1.5, startZ + 8.5), ArmorStand.class);
        Agent.spawn(ctx, new Location(location.getWorld(), startX + 8.5, 1.5, startZ + 8.5));
    }
}
