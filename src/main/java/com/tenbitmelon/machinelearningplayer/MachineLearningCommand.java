package com.tenbitmelon.machinelearningplayer;

import com.mojang.brigadier.Command;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.builder.LiteralArgumentBuilder;
import com.mojang.brigadier.tree.LiteralCommandNode;
import io.papermc.paper.command.brigadier.CommandSourceStack;
import io.papermc.paper.command.brigadier.Commands;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.entity.Player;

@SuppressWarnings("UnstableApiUsage")
public class MachineLearningCommand {

    public static LiteralCommandNode<CommandSourceStack> register() {
        LiteralArgumentBuilder<CommandSourceStack> commandBuilder = Commands.literal("ml")
                .executes(ctx -> {
                    ctx.getSource().getSender().sendPlainMessage("[STATS]");
                    return Command.SINGLE_SUCCESS;
                });
        commandBuilder.then(Commands.literal("rooms")
            .executes(ctx -> {
                ctx.getSource().getSender().sendPlainMessage("[ROOMS]");
                return Command.SINGLE_SUCCESS;
            }).then(Commands.literal("create")
                .executes(ctx -> {
                    ctx.getSource().getSender().sendPlainMessage("[ROOMS] Creating...");
                    generateNoiseSquares(ctx.getSource().getLocation(), 50);
                    ctx.getSource().getSender().sendPlainMessage("[ROOMS] Created 1 room.");
                    return Command.SINGLE_SUCCESS;
                })
                .then(Commands.argument("amount", IntegerArgumentType.integer(1))
                    .executes(ctx -> {
                        int amount = IntegerArgumentType.getInteger(ctx, "amount");
                        generateNoiseSquares(ctx.getSource().getLocation(), amount);
                        ctx.getSource().getSender().sendPlainMessage("[ROOMS] Created " + amount + " rooms.");
                        return Command.SINGLE_SUCCESS;
                    })))).then(Commands.literal("edit").executes(ctx -> {
            ctx.getSource().getSender().sendPlainMessage("[ROOMS] Edit mode enabled.");
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

    private static final Material[] PASTEL_BLOCKS = {
        Material.PINK_CONCRETE,          // Light pink
        Material.LIME_CONCRETE,          // Light green
        Material.LIGHT_BLUE_CONCRETE,    // Light blue
        Material.YELLOW_CONCRETE,        // Light yellow
        Material.MAGENTA_CONCRETE,       // Light magenta
        Material.CYAN_CONCRETE,          // Light cyan
        Material.ORANGE_CONCRETE,        // Light peach/orange
        Material.PURPLE_CONCRETE         // Light lavender/purple
    };

    /**
     * Simple noise function equivalent to the Python version
     * @param x X coordinate
     * @param y Y coordinate
     * @param scale Scale factor for the noise
     * @return Noise value between 0 and 1
     */
    private static double simpleNoise(int x, int y, double scale) {
        return (Math.sin(x * scale) * Math.cos(y * scale) + 1) / 2;
    }

    public static void generateNoiseSquares(Location playerLoc, int halfSize) {
        World world = playerLoc.getWorld();

        if (world == null) {
            return;
        }

        int centerX = playerLoc.getBlockX();
        int centerZ = playerLoc.getBlockZ();
        int playerY = playerLoc.getBlockY();

        for (int offsetX = -halfSize; offsetX < halfSize; offsetX++) {
            for (int offsetZ = -halfSize; offsetZ < halfSize; offsetZ++) {

                int worldX = centerX + offsetX;
                int worldZ = centerZ + offsetZ;

                double sizeNoise = simpleNoise(worldX, worldZ, 0.02);
                int squareSize = (int)(4 + sizeNoise * 20);

                int gridX = (worldX / squareSize);
                int gridZ = (worldZ / squareSize);

                int colorIndex = ((gridX + gridZ) % PASTEL_BLOCKS.length + PASTEL_BLOCKS.length) % PASTEL_BLOCKS.length;
                Material blockType = PASTEL_BLOCKS[colorIndex];

                // Block highestBlockAt = world.getHighestBlockAt(worldX, worldZ);

                Location blockLoc = new Location(world, worldX, playerY, worldZ);
                world.getBlockAt(blockLoc).setType(blockType);
            }
        }
    }

    /**
     * Alternative version that generates on multiple Y levels (like a 3D area)
     * @param player The player to center the generation around
     * @param height How many blocks high to generate (default: 1)
     */
    public static void generateNoiseSquares3D(Player player, int height) {
        Location playerLoc = player.getLocation();
        World world = playerLoc.getWorld();

        if (world == null) {
            return;
        }

        int centerX = playerLoc.getBlockX();
        int centerZ = playerLoc.getBlockZ();
        int startY = playerLoc.getBlockY();

        int halfSize = 50;

        // Generate for multiple Y levels
        for (int y = 0; y < height; y++) {
            for (int offsetX = -halfSize; offsetX < halfSize; offsetX++) {
                for (int offsetZ = -halfSize; offsetZ < halfSize; offsetZ++) {

                    int worldX = centerX + offsetX;
                    int worldZ = centerZ + offsetZ;
                    int worldY = startY + y;

                    int localX = offsetX + halfSize;
                    int localZ = offsetZ + halfSize;

                    // Use Y level to slightly modify the noise for variation
                    double sizeNoise = simpleNoise(localX, localZ + y * 10, 0.02);
                    int squareSize = (int)(4 + sizeNoise * 20);

                    int gridX = (localX / squareSize) * squareSize;
                    int gridZ = (localZ / squareSize) * squareSize;

                    // Include Y level in color selection for 3D variation
                    int colorIndex = (gridX / squareSize + gridZ / squareSize + y) % PASTEL_BLOCKS.length;
                    Material blockType = PASTEL_BLOCKS[colorIndex];

                    Location blockLoc = new Location(world, worldX, worldY, worldZ);
                    world.getBlockAt(blockLoc).setType(blockType);
                }
            }
        }

        player.sendMessage("§aGenerated 3D noise-based squares in 100x100x" + height + " area!");
    }
}
