package com.tenbitmelon.machinelearningplayer;

import com.mojang.brigadier.Command;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.builder.LiteralArgumentBuilder;
import com.mojang.brigadier.tree.LiteralCommandNode;
import com.tenbitmelon.machinelearningplayer.agent.Agent;
import com.tenbitmelon.machinelearningplayer.debugger.Debugger;
import com.tenbitmelon.machinelearningplayer.debugger.ui.UIElement;
import com.tenbitmelon.machinelearningplayer.models.ExperimentConfig;
import io.papermc.paper.adventure.providers.ClickCallbackProviderImpl;
import io.papermc.paper.command.brigadier.CommandSourceStack;
import io.papermc.paper.command.brigadier.Commands;
import net.minecraft.server.MinecraftServer;
import org.bukkit.Chunk;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.command.CommandSender;
import org.slf4j.event.Level;

import java.lang.reflect.Field;
import java.util.Map;

import static com.tenbitmelon.machinelearningplayer.MachineLearningPlayer.LOGGER;

@SuppressWarnings("UnstableApiUsage")
public class MachineLearningCommand {

    public static LiteralCommandNode<CommandSourceStack> register() {
        LiteralArgumentBuilder<CommandSourceStack> commandBuilder = Commands.literal("ml")
            .executes(ctx -> {
                ctx.getSource().getSender().sendPlainMessage("[STATS]");
                return Command.SINGLE_SUCCESS;
            })
            .then(Commands.literal("clearUiCallbacks").executes(ctx -> {
                ClickCallbackProviderImpl.CallbackManager callbackManager = ClickCallbackProviderImpl.CALLBACK_MANAGER;

                try {
                    Field callbacksField = callbackManager.getClass().getDeclaredField("callbacks");
                    callbacksField.setAccessible(true);
                    Map<?, ?> callbacksMap = (Map<?, ?>) callbacksField.get(callbackManager);
                    ctx.getSource().getSender().sendPlainMessage("UI callbacks Size: " + callbacksMap.size());

                    callbacksMap.clear();
                    ctx.getSource().getSender().sendPlainMessage("UI callbacks cleared. (Size: " + callbacksMap.size() + ")");
                } catch (NoSuchFieldException | IllegalAccessException e) {
                    throw new RuntimeException(e);
                }
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("uiupdates").executes(ctx -> {
                boolean wasEnabled = UIElement.ALLOW_UPDATES;
                UIElement.ALLOW_UPDATES = !wasEnabled;
                ctx.getSource().getSender().sendPlainMessage("UI updates " + (wasEnabled ? "disabled" : "enabled"));
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("removeDisplays").executes(ctx -> {
                Debugger.stop();
                return Command.SINGLE_SUCCESS;
            }))
            .then(logLevels())
            .then(args());

        return commandBuilder.build();
    }

    public static LiteralArgumentBuilder<CommandSourceStack> logLevels() {
        return Commands.literal("loglevel")
            .then(Commands.literal("debug")
                .executes(ctx -> {
                    boolean wasEnabled = LOGGER.isEnabled(Level.DEBUG);
                    LOGGER.setEnabled(Level.DEBUG, !wasEnabled);
                    ctx.getSource().getSender().sendPlainMessage("Debug logging " + (wasEnabled ? "disabled" : "enabled"));
                    return Command.SINGLE_SUCCESS;
                }))
            .then(Commands.literal("info")
                .executes(ctx -> {
                    LOGGER.setEnabled(Level.INFO, true);
                    ctx.getSource().getSender().sendPlainMessage("Info logging enabled");
                    return Command.SINGLE_SUCCESS;
                }))
            .then(Commands.literal("warn")
                .executes(ctx -> {
                    LOGGER.setEnabled(Level.WARN, true);
                    ctx.getSource().getSender().sendPlainMessage("Warn logging enabled");
                    return Command.SINGLE_SUCCESS;
                }))
            .then(Commands.literal("error")
                .executes(ctx -> {
                    LOGGER.setEnabled(Level.ERROR, true);
                    ctx.getSource().getSender().sendPlainMessage("Error logging enabled");
                    return Command.SINGLE_SUCCESS;
                }))
            .then(Commands.literal("trace")
                .executes(ctx -> {
                    LOGGER.setEnabled(Level.TRACE, true);
                    ctx.getSource().getSender().sendPlainMessage("Trace logging enabled");
                    return Command.SINGLE_SUCCESS;
                }));
    }

    public static LiteralArgumentBuilder<CommandSourceStack> args() {
        ExperimentConfig config = ExperimentConfig.getInstance();
        return Commands.literal("args").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("=== Arguments:");
                sender.sendPlainMessage("Seed: " + config.seed);
                sender.sendPlainMessage("Learning Rate: " + config.learningRate);
                sender.sendPlainMessage("Number of Environments: " + config.numEnvs);
                sender.sendPlainMessage("Anneal Learning Rate: " + config.annealLr);
                sender.sendPlainMessage("Gamma: " + config.gamma);
                sender.sendPlainMessage("GAE Lambda: " + config.gaeLambda);
                sender.sendPlainMessage("Number of Minibatches: " + config.numMinibatches);
                sender.sendPlainMessage("Update Epochs: " + config.updateEpochs);
                sender.sendPlainMessage("Normalize Advantages: " + config.normAdv);
                sender.sendPlainMessage("Clip Coefficient: " + config.clipCoef);
                sender.sendPlainMessage("Clip Value Loss: " + config.clipVloss);
                sender.sendPlainMessage("Entropy Coefficient: " + config.entCoef);
                sender.sendPlainMessage("Value Function Coefficient: " + config.vfCoef);
                sender.sendPlainMessage("Max Gradient Norm: " + config.maxGradNorm);
                sender.sendPlainMessage("Target KL: " + config.targetKl);
                sender.sendPlainMessage("Experiment Name: " + config.expName);
                sender.sendPlainMessage("Torch Deterministic: " + config.torchDeterministic);
                sender.sendPlainMessage("CUDA Enabled: " + config.cuda);
                sender.sendPlainMessage("Track with WandB: " + config.track);
                sender.sendPlainMessage("WandB Project Name: " + config.wandbProjectName);
                sender.sendPlainMessage("WandB Entity: " + (config.wandbEntity != null ? config.wandbEntity : "None"));
                sender.sendPlainMessage("Capture Video: " + config.captureVideo);
                sender.sendPlainMessage("Number of Steps: " + config.numSteps);
                sender.sendPlainMessage("Batch Size: " + config.batchSize);
                sender.sendPlainMessage("Mini-batch Size: " + config.minibatchSize);
                sender.sendPlainMessage("Number of Iterations: " + config.numIterations);
                return Command.SINGLE_SUCCESS;
            })
            .then(Commands.literal("seed").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Seed: " + config.seed);
                sender.sendPlainMessage("Seed of the experiment.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("learningRate").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Learning Rate: " + config.learningRate);
                sender.sendPlainMessage("The learning rate of the optimizer.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("numEnvs").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Number of Environments: " + config.numEnvs);
                sender.sendPlainMessage("The number of parallel game environments.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("annealLr").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Anneal Learning Rate: " + config.annealLr);
                sender.sendPlainMessage("Toggle learning rate annealing for policy and value networks.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("gamma").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Gamma: " + config.gamma);
                sender.sendPlainMessage("The discount factor gamma.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("gaeLambda").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("GAE Lambda: " + config.gaeLambda);
                sender.sendPlainMessage("The lambda for the general advantage estimation.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("numMinibatches").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Number of Minibatches: " + config.numMinibatches);
                sender.sendPlainMessage("The number of mini-batches.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("updateEpochs").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Update Epochs: " + config.updateEpochs);
                sender.sendPlainMessage("The K epochs to update the policy.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("normAdv").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Normalize Advantages: " + config.normAdv);
                sender.sendPlainMessage("Toggles advantages normalization.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("clipCoef").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Clip Coefficient: " + config.clipCoef);
                sender.sendPlainMessage("The surrogate clipping coefficient.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("clipVloss").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Clip Value Loss: " + config.clipVloss);
                sender.sendPlainMessage("Toggles whether or not to use a clipped loss for the value function, as per the paper.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("entCoef").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Entropy Coefficient: " + config.entCoef);
                sender.sendPlainMessage("Coefficient of the entropy.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("vfCoef").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Value Function Coefficient: " + config.vfCoef);
                sender.sendPlainMessage("Coefficient of the value function.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("maxGradNorm").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Max Gradient Norm: " + config.maxGradNorm);
                sender.sendPlainMessage("The maximum norm for the gradient clipping.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("targetKl").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Target KL: " + config.targetKl);
                sender.sendPlainMessage("The target KL divergence threshold. Can be null if not used.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("expName").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Experiment Name: " + config.expName);
                sender.sendPlainMessage("The name of this experiment.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("torchDeterministic").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Torch Deterministic: " + config.torchDeterministic);
                sender.sendPlainMessage("If toggled (true), PyTorch's `torch.backends.cudnn.deterministic` is typically set to true.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("cuda").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("CUDA Enabled: " + config.cuda);
                sender.sendPlainMessage("If toggled (true), CUDA will be enabled by default if available.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("track").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Track with WandB: " + config.track);
                sender.sendPlainMessage("If toggled (true), this experiment will be tracked with Weights and Biases.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("wandbProjectName").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("WandB Project Name: " + config.wandbProjectName);
                sender.sendPlainMessage("The WandB's project name.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("wandbEntity").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("WandB Entity: " + (config.wandbEntity != null ? config.wandbEntity : "None"));
                sender.sendPlainMessage("The entity (team) of WandB's project. Can be null.");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("captureVideo").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Capture Video: " + config.captureVideo);
                sender.sendPlainMessage("Whether to capture videos of the agent performances (check out `videos` folder).");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("numSteps").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Number of Steps: " + config.numSteps);
                sender.sendPlainMessage("The number of steps to run in each environment per policy rollout.");
                return Command.SINGLE_SUCCESS;
            }).then(Commands.argument("value", IntegerArgumentType.integer()).executes(ctx -> {
                config.numSteps = ctx.getArgument("value", Integer.class);
                return Command.SINGLE_SUCCESS;
            })))
            .then(Commands.literal("batchSize").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Batch Size: " + config.batchSize);
                sender.sendPlainMessage("The batch size (computed in runtime, e.g., numEnvs * numSteps).");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("minibatchSize").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Mini-batch Size: " + config.minibatchSize);
                sender.sendPlainMessage("The mini-batch size (computed in runtime, e.g., batchSize / numMinibatches).");
                return Command.SINGLE_SUCCESS;
            }))
            .then(Commands.literal("numIterations").executes(ctx -> {
                CommandSender sender = ctx.getSource().getSender();
                sender.sendPlainMessage("Number of Iterations: " + config.numIterations);
                sender.sendPlainMessage("The number of iterations (computed in runtime, e.g., totalTimesteps / batchSize).");
                return Command.SINGLE_SUCCESS;
            }).then(Commands.argument("value", IntegerArgumentType.integer()).executes(ctx -> {
                config.numIterations = ctx.getArgument("value", Integer.class);
                return Command.SINGLE_SUCCESS;
            })));
    }

    public static void createRooms(MinecraftServer server, World world, int count) {
        Location location = new Location(world, 0, 0, 0);
        int gridWidth = (int) Math.ceil(Math.sqrt(count));

        for (int i = 0; i < count; i++) {
            int offsetX = (i % gridWidth) * 16;
            int offsetZ = (i / gridWidth) * 16;

            Location roomLocation = location.clone().add(offsetX, 0, offsetZ);
            createRoom(server, roomLocation);
        }
    }

    public static void createRoom(MinecraftServer server, Location location) {
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
        Agent.spawn(server, new Location(location.getWorld(), startX + 8.5, 1.5, startZ + 8.5));
    }
}
