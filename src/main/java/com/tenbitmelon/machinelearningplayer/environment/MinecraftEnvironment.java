package com.tenbitmelon.machinelearningplayer.environment;

import com.tenbitmelon.machinelearningplayer.agent.Agent;
import com.tenbitmelon.machinelearningplayer.agent.EntityPlayerActionPack;
import com.tenbitmelon.machinelearningplayer.debugger.ui.TextWindow;
import com.tenbitmelon.machinelearningplayer.models.ExperimentConfig;
import com.tenbitmelon.machinelearningplayer.util.TextDisplayBuilder;
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
import org.bytedeco.pytorch.global.torch;

import java.util.concurrent.CompletableFuture;

import static com.tenbitmelon.machinelearningplayer.util.Utils.szudzikUnpairing;
import static com.tenbitmelon.machinelearningplayer.util.Utils.tensorString;

public class MinecraftEnvironment {

    public static final int GRID_SIZE_XZ = 5;
    public static final int HALF_GRID_SIZE_XZ = GRID_SIZE_XZ / 2;
    public static final int GRID_SIZE_Y = 5;
    public static final int HALF_GRID_SIZE_Y = GRID_SIZE_Y / 2;
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

        // Uniform distribution over the area
        double angle = Math.random() * 2 * Math.PI;
        double radius = Math.sqrt(Math.random() * (maxDist * maxDist - minDist * minDist) + minDist * minDist);

        double x = radius * Math.cos(angle);
        double y = radius * Math.sin(angle);
        return new double[]{x, y};
    }

    public Observation getObservation() {
        // Observation observation = new Observation();
        float[] observationData = new float[Observation.OBSERVATION_SPACE_SIZE];

        // 1. Fill voxel grid data
        Vec3 position = agent.position();
        // World world = roomLocation.getWorld();

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
        }
        // int index = Observation.OFFSET_VOXEL_GRID;
        // for (int x = -HALF_GRID_SIZE_XZ; x < HALF_GRID_SIZE_XZ; x++) {
        //     for (int z = -HALF_GRID_SIZE_XZ; z < HALF_GRID_SIZE_XZ; z++) {
        //         for (int y = -HALF_GRID_SIZE_Y; y < HALF_GRID_SIZE_Y; y++) {
        //
        //             if (y < 0) {
        //                 observationData[index] = 0.5f;
        //                 // TODO: Adapt this when the goal eventually has different y positions
        //             } else if (goalPosition != null && (int) (position.x() + x) == (int) (goalPosition.x) && (int) (position.y() + y) == (int) (goalPosition.y) && (int) (position.z() + z) == (int) (goalPosition.z)) {
        //                 observationData[index] = 1.0f;
        //             } else {
        //                 observationData[index] = 0.0f;
        //             }
        //
        //             index++;
        //
        //             // new TextDisplayBuilder(roomLocation.getWorld())
        //             //     .text(observationData[index] + " " + index).build().teleport(new Location(world, position.x() + x, position.y() + y, position.z() + z));
        //         }
        //     }
        // } */

        // // 2. Position in block (3 values)
        // observationData[Observation.OFFSET_POSITION_IN_BLOCK + 0] = (float) (position.x() - (int) (position.x()));
        // observationData[Observation.OFFSET_POSITION_IN_BLOCK + 1] = (float) (position.y() - (int) (position.y()));
        // observationData[Observation.OFFSET_POSITION_IN_BLOCK + 2] = (float) (position.z() - (int) (position.z()));

        // // 3. Velocity (3 values)
        // Vec3 agentVelocity = agent.getDeltaMovement();
        // observationData[Observation.OFFSET_VELOCITY + 0] = (float) agentVelocity.x() / 5.0f; // TODO: Normalize based on max expected velocity
        // observationData[Observation.OFFSET_VELOCITY + 1] = (float) agentVelocity.y() / 20.0f; // falling from 15 blocks is 26.41 m/s (8 blocks is 20.95 m/s)
        // observationData[Observation.OFFSET_VELOCITY + 2] = (float) agentVelocity.z() / 5.0f;

        // // 4. Look direction (3 values)
        // float yawRad = (float) Math.toRadians(agent.getYRot());
        // float pitchRad = (float) Math.toRadians(agent.getXRot());
        // observationData[Observation.OFFSET_YAW + 0] = (float) Math.sin(yawRad);
        // observationData[Observation.OFFSET_YAW + 1] = (float) Math.cos(yawRad);
        // observationData[Observation.OFFSET_PITCH + 0] = (float) Math.sin(pitchRad);
        // observationData[Observation.OFFSET_PITCH + 1] = (float) Math.cos(pitchRad);

        // 5. Boolean flags (4 values)
        observationData[Observation.OFFSET_SPRINTING] = agent.actionPack.sprinting ? 1.0f : 0.0f;
        observationData[Observation.OFFSET_SNEAKING] = agent.actionPack.sneaking ? 1.0f : 0.0f;
        observationData[Observation.OFFSET_ON_GROUND] = agent.onGround ? 1.0f : 0.0f;

        // 6. Goal direction (3 values) // TODO: Make this relative to the agent's look direction
        Vec3 goalDirectionVec = goalPosition.subtract(position).normalize();
        // getYRot is in Degrees
        // yRot expects Radians
        float yawRadians = (float) Math.toRadians(agent.getYRot());
        // yRot rotates clockwise around the Y axis, which is the opposite of what I expected
        // so we don't need to negate the angle because its already doing that
        goalDirectionVec = goalDirectionVec.yRot(yawRadians);

        observationData[Observation.OFFSET_GOAL_DIRECTION + 0] = (float) goalDirectionVec.x;
        observationData[Observation.OFFSET_GOAL_DIRECTION + 1] = (float) goalDirectionVec.y;
        observationData[Observation.OFFSET_GOAL_DIRECTION + 2] = (float) goalDirectionVec.z;

        // 7. Goal distance (1 value)
        float goalDistance = (float) Math.clamp(position.distanceTo(goalPosition) / 20.0f, 0.0, 1.0); // Normalize by a max expected distance // TODO: Make this a variable
        observationData[Observation.OFFSET_GOAL_DISTANCE] = goalDistance;

        Tensor observationTensor = torch.tensor(observationData);
        Observation observation = new Observation(observationTensor);

        agent.displayObservation(observation);

        return observation;
    }


    public ResetResult reset() {
        environmentLog.clearLines();
        this.currentStep = 0;

        this.agent.reset(roomLocation.clone().add(8.5, 1.0, 8.5));

        if (goalPosition != null) {
            roomLocation.getWorld().getBlockAt((int) goalPosition.x, roomLocation.getBlockY(), (int) goalPosition.z).setType(randomConcrete);
        }

        double[] randomPoint = getRandomPointInCircle(GRID_SIZE_XZ / 2f, GRID_SIZE_XZ);
        this.goalPosition = new Vec3(
            roomLocation.getX() + 8.5 + randomPoint[0],
            roomLocation.getY() + 1.0,
            roomLocation.getZ() + 8.5 + randomPoint[1]
        );

        roomLocation.getWorld().getBlockAt((int) goalPosition.x, roomLocation.getBlockY(), (int) goalPosition.z).setType(Material.EMERALD_BLOCK);

        Observation observation = getObservation();
        Vec2 goalDirection = new Vec2((float) randomPoint[0], (float) randomPoint[1]);
        previousDistanceToGoal = goalDirection.length();
        return new ResetResult(observation, new Info(previousDistanceToGoal));
    }

    public void preTickStep(Tensor actionTensor) {
        Action action = new Action(actionTensor);

        this.currentStep++;
        agent.actionPack.stopAll();

        int sprintingSneaking = action.sprintingSneaking();
        if (sprintingSneaking == 1) {
            // Sprinting
            agent.actionPack.setSprinting(true);
            agent.actionPack.setSneaking(false);
        } else if (sprintingSneaking == 2) {
            // Sneaking
            agent.actionPack.setSprinting(false);
            agent.actionPack.setSneaking(true);
        } else {
            agent.actionPack.setSprinting(false);
            agent.actionPack.setSneaking(false);
        }

        if (action.jumping() == 1) {
            agent.actionPack.start(EntityPlayerActionPack.ActionType.JUMP, EntityPlayerActionPack.Action.once());
        }

        Vec2 rotation = action.lookChange().scale(15.0f); // Scale to a reasonable rotation speed
        agent.actionPack.turn(rotation); // Yaw, Pitch

        int moveForward = action.forwardMoveKey();
        if (moveForward == 1) {
            moveForward = 1;
        } else if (moveForward == 2) {
            moveForward = -1;
        }
        int moveRight = action.strafeMoveKey();
        if (moveRight == 1) {
            moveRight = 1;
        } else if (moveRight == 2) {
            moveRight = -1;
        }

        agent.actionPack.setForward(moveForward);
        agent.actionPack.setStrafing(moveRight);
    }

    public StepResult postTickStep() {
        Vec3 position = agent.position();
        Vec3 goalDirectionVec = goalPosition.subtract(position);

        float distanceToGoal = (float) goalDirectionVec.length();

        boolean reachedGoal = distanceToGoal < GOAL_THRESHOLD;
        boolean terminated = false;

        float reward = 0.0f;
        if (reachedGoal) {
            reward = 100.0f;
            terminated = true;
        } else {
            reward = -distanceToGoal * 0.1f;
            terminated = false;
        }

        boolean truncated = this.currentStep > this.args.numSteps;

        Observation observation = getObservation();
        Info info = new Info(distanceToGoal);
        previousDistanceToGoal = distanceToGoal;

        return new StepResult(observation, reward, terminated, truncated, info);


        // double reward = 0.0;
        // boolean terminated = false;
        //
        // // 1. Calculate improvement (positive if closer, negative if farther)
        // // Scale it up so the signal is strong (e.g., 1 block = 1.0 reward)
        // double delta = (previousDistanceToGoal - distanceToGoal);
        // reward += delta; // Scaling factor
        //
        // // 2. Goal Achievement
        // if (distanceToGoal < GOAL_THRESHOLD) {
        //     reward += 20.0; // Bonus for finishing
        //     terminated = true;
        // }
        //
        // // 3. Optional: Tiny time penalty to encourage speed (e.g., -0.01)
        // reward -= 0.01;
        //
        // boolean truncated = this.currentStep > this.args.numSteps;
        // previousDistanceToGoal = distanceToGoal;
        //
        // return new StepResult(getObservation(), reward, terminated, truncated, info);
        //
        //
        // // environmentLog.addLine(Component.newline().append(
        // //     Component.text(" S: " + currentStep).color(NamedTextColor.YELLOW),
        // //     Component.newline(),
        // //     Component.text(" D: "),
        // //     Component.text(String.format("%.2f", info.distanceToGoal())),
        // //     Component.newline(),
        // //     Component.text("ΔD: "),
        // //     Component.text(String.format("%.2f (%.2f)", deltaDistanceToGoal, deltaDistanceToGoal * 5.0))
        // //         .color(deltaDistanceToGoal > 0 ? NamedTextColor.GREEN : NamedTextColor.RED),
        // //     Component.newline(),
        // //     Component.text(" R: "),
        // //     Component.text(String.format("%.2f", reward))
        // //         .color(reward > 50 ? NamedTextColor.GOLD : reward > 0 ? NamedTextColor.GREEN : NamedTextColor.RED)
        // // ));
        //
        // // new TextDisplayBuilder(roomLocation.getWorld())
        // //     .teleport(agent.position().multiply(1.0, 0.0, 1.0).add(0, 3 + currentStep / 1000.0, 0))
        // //     // Left quaternion and right quaternion. Need to rotate the text display to face directly up (ie around the x axis 90 degrees)
        // //     .text(String.format("%.2f", reward))
        // //     .billboard(Display.Billboard.CENTER);
        //
        // // System.out.println(agent.position().x() + ", " +
        // //     agent.position().z() + ", " +
        // //     String.format("%.5f", info.distanceToGoal()) + ", " +
        // //     String.format("%.5f", deltaDistanceToGoal) + ", " +
        // //     String.format("%.5f", reward));
    }

    public boolean isReady() {
        return agent != null && agent.isReady();
    }
}
