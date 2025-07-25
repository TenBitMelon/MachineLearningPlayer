package com.tenbitmelon.machinelearningplayer.environment;

import com.tenbitmelon.machinelearningplayer.agent.Agent;
import com.tenbitmelon.machinelearningplayer.agent.EntityPlayerActionPack;
import com.tenbitmelon.machinelearningplayer.debugger.ui.TextWindow;
import com.tenbitmelon.machinelearningplayer.models.ExperimentConfig;
import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.format.NamedTextColor;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.phys.Vec2;
import net.minecraft.world.phys.Vec3;
import org.bukkit.*;
import org.bukkit.craftbukkit.CraftServer;
import org.bukkit.entity.Display;
import org.bukkit.entity.TextDisplay;
import org.bukkit.util.BoundingBox;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;

import java.util.concurrent.CompletableFuture;

import static com.tenbitmelon.machinelearningplayer.util.Utils.szudzikUnpairing;
import static com.tenbitmelon.machinelearningplayer.util.Utils.tensorString;

public class MinecraftEnvironment {

    public static final int GRID_SIZE_XZ = 5;
    public static final int GRID_SIZE_Y = 5;
    public static final int GRID_VOLUME = GRID_SIZE_XZ * GRID_SIZE_XZ * GRID_SIZE_Y;

    public static final double GOAL_THRESHOLD = 0.75;
    private static int nextEnvironmentId = 0;

    private final ExperimentConfig args;
    private final Location roomLocation;
    private final BoundingBox boundingBox;
    TextWindow environmentLog;
    private Agent agent;
    private int currentStep = 0;
    private Vec3 goalPosition;
    private double previousDistanceToGoal;
    private Material randomConcrete;
    private boolean isInitialSetup = true;

    public MinecraftEnvironment(ExperimentConfig args) {
        this.args = args;
        // LOGGER.debug("Initializing Minecraft environment");
        // int w = (int) Math.floor((Math.sqrt(8 * nextEnvironmentId + 1) - 1) / 2);
        // int t = (w * w + w) / 2;
        // roomLocation = new Location(world, (w - (nextEnvironmentId - t)) * 16, 0, (nextEnvironmentId - t) * 16);

        World world = Bukkit.getWorlds().getFirst();

        int[] coords = szudzikUnpairing(nextEnvironmentId);
        roomLocation = new Location(world, coords[0] * 16, 0, coords[1] * 16);

        environmentLog = new TextWindow(Display.Billboard.VERTICAL, TextDisplay.TextAlignment.LEFT);
        environmentLog.setPosition(roomLocation.toVector().toVector3d().mul(1, 0, 1).add(5.0f, 4.0f, 5.0f));

        Chunk chunk = roomLocation.getChunk();
        chunk.load();

        int startX = chunk.getX() * 16;
        int startZ = chunk.getZ() * 16;

        boundingBox = BoundingBox.of(
            new Location(world, startX + 5, 0, startZ + 5),
            new Location(world, startX + 11, 5, startZ + 11)
        );

        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                for (int y = -16; y < 64; y++) {
                    world.getBlockAt(startX + x, y, startZ + z).setType(Material.AIR);
                }
                world.getBlockAt(startX + x, 0, startZ + z).setType(Material.BARRIER);
            }
        }

        // world.getBlockAt(startX + 8, -3, startZ + 8).setType(Material.BEACON);
        // for (int x = -1; x <= 1; x++) {
        //     for (int z = -1; z <= 1; z++) {
        //         world.getBlockAt(startX + 8 + x, -4, startZ + 8 + z).setType(Material.IRON_BLOCK);
        //     }
        // }

        randomConcrete = new Material[]{
            Material.WHITE_CONCRETE,
            Material.ORANGE_CONCRETE,
            Material.MAGENTA_CONCRETE,
            Material.LIGHT_BLUE_CONCRETE,
            Material.YELLOW_CONCRETE,
            Material.LIME_CONCRETE,
            Material.PINK_CONCRETE,
            Material.GRAY_CONCRETE,
            Material.LIGHT_GRAY_CONCRETE,
            Material.CYAN_CONCRETE,
            Material.PURPLE_CONCRETE,
            Material.BLUE_CONCRETE,
            Material.BROWN_CONCRETE,
            Material.GREEN_CONCRETE,
            Material.RED_CONCRETE
        }[(int) (Math.random() * 15)];

        final int roomSize = 3; // inset 3 blocks from each side
        for (int offsetX = 0; offsetX < 16; offsetX++) {
            for (int offsetZ = 0; offsetZ < 16; offsetZ++) {

                int worldX = roomLocation.getBlockX() + offsetX;
                int worldZ = roomLocation.getBlockZ() + offsetZ;

                roomLocation.getWorld().getBlockAt(worldX, 0, worldZ).setType(Material.BARRIER);
                roomLocation.getWorld().getBlockAt(worldX, -1, worldZ).setType(Material.BARRIER);

                if (offsetX > roomSize && offsetX < (16 - roomSize) && offsetZ > roomSize && offsetZ < (16 - roomSize)) {
                    roomLocation.getWorld().getBlockAt(worldX, 0, worldZ).setType(randomConcrete);
                    roomLocation.getWorld().getBlockAt(worldX, -1, worldZ).setType(randomConcrete);
                }

            }
        }

        roomLocation.getWorld().getBlockAt(roomLocation.getBlockX() + 8, 0, roomLocation.getBlockZ() + 8).setType(Material.GOLD_BLOCK);

        MinecraftServer server = ((CraftServer) Bukkit.getServer()).getServer();
        CompletableFuture<Agent> completableFuture = Agent.spawn(server, new Location(world, startX + 8.5, 1.5, startZ + 8.5));
        completableFuture.whenComplete((agent, throwable) -> {
            if (throwable == null) {
                this.agent = agent;
            }
        });

        nextEnvironmentId++;
    }

    public static double[] getRandomPointInCircle(double minDist, double maxDist) {
        if (minDist < 0 || maxDist <= minDist) {
            throw new IllegalArgumentException("Invalid min/max distance");
        }

        // Uniform distribution over the annulus area
        double angle = Math.random() * 2 * Math.PI;
        double radius = Math.sqrt(Math.random() * (maxDist * maxDist - minDist * minDist) + minDist * minDist);

        double x = radius * Math.cos(angle);
        double y = radius * Math.sin(angle);
        return new double[]{x, y};
    }

    public Observation getObservation() {
        // Observation observation = new Observation();
        float[] observationData = new float[Observation.OBSERVATION_SPACE_SIZE];
        int offset = 0;

        // 1. Fill voxel grid data
        Vec3 position = agent.position();
        World world = roomLocation.getWorld();

        /*for (int x = -GRID_SIZE_XZ / 2; x < GRID_SIZE_XZ / 2; x++) {
            for (int z = -GRID_SIZE_XZ / 2; z < GRID_SIZE_XZ / 2; z++) {
                for (int y = -GRID_SIZE_Y / 2; y < GRID_SIZE_Y / 2; y++) {
                    int blockX = (int) (position.x() + x);
                    int blockY = (int) (position.y() + y);
                    int blockZ = (int) (position.z() + z);

                    if (boundingBox.contains(blockX, blockY, blockZ)) {
                        int dataIndex = offset + (x + GRID_SIZE_XZ / 2) * GRID_SIZE_XZ * GRID_SIZE_Y +
                            (z + GRID_SIZE_XZ / 2) * GRID_SIZE_Y + y;

                        Material blockType = world.getBlockAt(blockX, blockY, blockZ).getType();
                        observationData[dataIndex] = blockType.isSolid() ? 1.0f : 0.0f;
                    }
                    // else stays 0.0f (default)
                }
            }
        }*/
        for (int y = -GRID_SIZE_Y / 2; y <= GRID_SIZE_Y / 2; y++) {
            for (int x = -GRID_SIZE_XZ / 2; x <= GRID_SIZE_XZ / 2; x++) {
                for (int z = -GRID_SIZE_XZ / 2; z <= GRID_SIZE_XZ / 2; z++) {
                    // new TextDisplayBuilder(roomLocation.getWorld())
                    //     .text(offset++ + "").build().teleport(new Location(world, position.x() + x, position.y() + y, position.z() + z));

                    int index = offset++;
                    if (offset < 50) {
                        observationData[index] = 1.0f;
                    }
                }
            }
        }

        // 2. Position in block (3 values)
        observationData[offset++] = (float) (position.x() - (int) (position.x()));
        observationData[offset++] = (float) (position.y() - (int) (position.y()));
        observationData[offset++] = (float) (position.z() - (int) (position.z()));

        // 3. Velocity (3 values)
        Vec3 agentVelocity = agent.getDeltaMovement();
        observationData[offset++] = (float) agentVelocity.x();
        observationData[offset++] = (float) agentVelocity.y();
        observationData[offset++] = (float) agentVelocity.z();

        // 4. Look direction (3 values)
        Vec3 lookDirectionVec = agent.getLookAngle().normalize();
        observationData[offset++] = (float) lookDirectionVec.x();
        observationData[offset++] = (float) lookDirectionVec.y();
        observationData[offset++] = (float) lookDirectionVec.z();

        // 5. Boolean flags (4 values)
        observationData[offset++] = agent.jumping ? 1.0f : 0.0f;
        observationData[offset++] = agent.actionPack.sprinting ? 1.0f : 0.0f;
        observationData[offset++] = agent.actionPack.sneaking ? 1.0f : 0.0f;
        observationData[offset++] = agent.onGround ? 1.0f : 0.0f;

        // 6. Goal direction (3 values)
        Vec3 goalDirectionVec = goalPosition.subtract(position).normalize();
        observationData[offset++] = (float) goalDirectionVec.x;
        observationData[offset++] = (float) goalDirectionVec.y;
        observationData[offset++] = (float) goalDirectionVec.z;

        // Single tensor operation to set all data at once
        Observation observation = new Observation(torch.tensor(observationData)); // TODO: Construct on device

        // Log the observation to the agent's debug log
        agent.displayObservation(observation);

        return observation;
    }

    public Info getInfo() {
        double distanceToGoal = agent.position().distanceTo(goalPosition);

        Info info = new Info(distanceToGoal);
        agent.displayInfo(info);

        return info;
    }


    public ResetResult reset() {
        environmentLog.clearLines();
        this.currentStep = 0;
        this.agent.reset(roomLocation.clone().add(8.5, 1.5, 8.5));

        if (goalPosition != null) {
            roomLocation.getWorld().getBlockAt((int) goalPosition.x, roomLocation.getBlockY(), (int) goalPosition.z).setType(randomConcrete);
        }

        double[] randomPoint = getRandomPointInCircle(GRID_SIZE_XZ / 2f, GRID_SIZE_XZ);
        this.goalPosition = new Vec3(
            roomLocation.getX() + 8.5 + randomPoint[0],
            roomLocation.getY() + 1,
            roomLocation.getZ() + 8.5 + randomPoint[1]
        );

        roomLocation.getWorld().getBlockAt((int) goalPosition.x, roomLocation.getBlockY(), (int) goalPosition.z).setType(Material.EMERALD_BLOCK);

        Observation observation = getObservation();
        Info info = getInfo();
        previousDistanceToGoal = info.distanceToGoal();
        return new ResetResult(observation, info);
    }

    public void preTickStep(Tensor actionTensor) {
        // LOGGER.debug("Stepping in MinecraftEnvironment with actions");
        Action action = new Action(actionTensor);

        // LOGGER.debug("Current step: {}, Action: {}", this.currentStep, tensorString(action.data));

        this.currentStep++;

        // LOGGER.debug("Updating agent action pack with action before: {}", agent.actionPack);
        agent.actionPack.setSprinting(action.sprinting() == 1);
        agent.actionPack.setSneaking(action.sneaking() == 1);
        if (action.jumping() == 1) {
            agent.actionPack.start(EntityPlayerActionPack.ActionType.JUMP, EntityPlayerActionPack.Action.once());
        }

        Vec2 rotation = action.lookChange();
        // LOGGER.debug("Setting agent rotation: [{}, {}]", rotation.x, rotation.y);
        agent.actionPack.turn(rotation);


        /*
        table:
        forward | forwardPressed | backwardPressed
        --------|----------------|----------------
        0       | false          | false
        1       | true           | false
        -1      | false          | true
        0       | true           | true
         */
        Action.MovementKeys movementKeys = action.moveKeys();
        // LOGGER.debug("Movement keys: {}", movementKeys);
        int moveForward = (movementKeys.forward() == 1 ? 1 : 0) - (movementKeys.backward() == 1 ? 1 : 0);
        int moveRight = (movementKeys.right() == 1 ? 1 : 0) - (movementKeys.left() == 1 ? 1 : 0);

        // LOGGER.debug("Setting agent movement: forward={}, right={}", moveForward, moveRight);

        agent.actionPack.setForward(moveForward);
        agent.actionPack.setStrafing(moveRight);

        // LOGGER.debug("Updating agent action pack with action after: {}", agent.actionPack);
    }

    public StepResult postTickStep() {
        // --- Calculate rewards
        Info info = getInfo();

        // LOGGER.debug("Calculating reward based on distance to goal: {}", info.distanceToGoal());

        // Positive is closer to goal, negative is further from goal
        double deltaDistanceToGoal = info.distanceToGoal() - previousDistanceToGoal;
        double reward = (deltaDistanceToGoal) * 10.0; // Reward based on distance to goal
        previousDistanceToGoal = info.distanceToGoal();

        reward -= 0.05; // Small penalty for each step taken

        // LOGGER.debug("Current step: {}, Reward: {}", this.currentStep, reward);

        boolean terminated = false;
        if (info.distanceToGoal() < GOAL_THRESHOLD) {
            // LOGGER.debug("Goal reached! Distance to goal: {}", info.distanceToGoal());
            reward += 100.0;
            terminated = true;
        }

        boolean truncated = this.currentStep > this.args.numSteps;

        // LOGGER.debug("Truncated step: {}", truncated);


        environmentLog.addLine(Component.newline().append(
            Component.text(" S: " + currentStep).color(NamedTextColor.YELLOW),
            Component.newline(),
            Component.text(" D: "),
            Component.text(String.format("%.2f", info.distanceToGoal())),
            Component.newline(),
            Component.text("ΔD: "),
            Component.text(String.format("%.2f", deltaDistanceToGoal))
                .color(deltaDistanceToGoal > 0 ? NamedTextColor.GREEN : NamedTextColor.RED),
            Component.newline(),
            Component.text(" R: "),
            Component.text(String.format("%.2f", reward))
                .color(reward > 50 ? NamedTextColor.GOLD : reward > 0 ? NamedTextColor.GREEN : NamedTextColor.RED)
        ));


        return new StepResult(getObservation(), reward, terminated, truncated, info);
    }

    public boolean isReady() {
        return agent != null && agent.isReady();
    }
}
