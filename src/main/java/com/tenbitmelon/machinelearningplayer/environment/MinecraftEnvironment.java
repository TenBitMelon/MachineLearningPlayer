package com.tenbitmelon.machinelearningplayer.environment;

import com.tenbitmelon.machinelearningplayer.agent.Agent;
import com.tenbitmelon.machinelearningplayer.agent.EntityPlayerActionPack;
import com.tenbitmelon.machinelearningplayer.debugger.Debugger;
import com.tenbitmelon.machinelearningplayer.models.ExperimentConfig;
import com.tenbitmelon.machinelearningplayer.util.BlockDisplayBuilder;
import net.kyori.adventure.text.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.phys.Vec2;
import net.minecraft.world.phys.Vec3;
import org.bukkit.*;
import org.bukkit.craftbukkit.CraftServer;
import org.bukkit.entity.BlockDisplay;
import org.bukkit.util.BoundingBox;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorIndex;
import org.bytedeco.pytorch.TensorIndexVector;
import org.joml.Matrix4f;

import java.util.concurrent.CompletableFuture;

import static com.tenbitmelon.machinelearningplayer.MachineLearningPlayer.LOGGER;

public class MinecraftEnvironment {

    public static final int GRID_SIZE_XZ = 5;
    public static final int GRID_SIZE_Y = 5;
    public static final int GRID_VOLUME = GRID_SIZE_XZ * GRID_SIZE_XZ * GRID_SIZE_Y;

    public static final double GOAL_THRESHOLD = 0.75;
    private static int nextEnvironmentId = 0;

    private final ExperimentConfig args;

    private Agent agent;
    private Location roomLocation;
    private BoundingBox boundingBox;
    private int currentStep = 0;
    private Vec3 goalPosition;
    private double previousDistanceToGoal;

    public MinecraftEnvironment(ExperimentConfig args) {
        this.args = args;
        LOGGER.info("Initializing Minecraft environment");
        int w = (int) Math.floor((Math.sqrt(8 * nextEnvironmentId + 1) - 1) / 2);
        int t = (w * w + w) / 2;

        World world = Bukkit.getWorlds().getFirst();

        roomLocation = new Location(world, (w - (nextEnvironmentId - t)) * 16, 0, (nextEnvironmentId - t) * 16);
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

        world.getBlockAt(startX + 8, -1, startZ + 8).setType(Material.BEACON);
        for (int x = -1; x <= 1; x++) {
            for (int z = -1; z <= 1; z++) {
                world.getBlockAt(startX + 8 + x, -2, startZ + 8 + z).setType(Material.IRON_BLOCK);
            }
        }

        roomLocation.getWorld().getBlockAt(roomLocation.getBlockX() + 8, 0, roomLocation.getBlockZ() + 8).setType(Material.GOLD_BLOCK);

        MinecraftServer server = ((CraftServer) Bukkit.getServer()).getServer();
        CompletableFuture<Agent> completableFuture = Agent.spawn(server, new Location(world, startX + 8.5, 1.5, startZ + 8.5));
        completableFuture.whenComplete((agent, throwable) -> {
            if (throwable != null) {
                throwable.printStackTrace();
            } else {
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
        Observation observation = new Observation();

        // Create voxel grid
        Tensor voxelGrid = observation.voxelGrid();
        Vec3 position = agent.position();
        World world = roomLocation.getWorld();
        // Loop over the GRID_SIZE_XZ x GRID_SIZE_XZ x GRID_SIZE_Y grid centered at the agent's position
        for (int x = -GRID_SIZE_XZ / 2; x < GRID_SIZE_XZ / 2; x++) {
            for (int z = -GRID_SIZE_XZ / 2; z < GRID_SIZE_XZ / 2; z++) {
                for (int y = 0; y < GRID_SIZE_Y; y++) {
                    int blockX = (int) Math.floor(position.x() + x);
                    int blockY = (int) Math.floor(position.y() + y);
                    int blockZ = (int) Math.floor(position.z() + z);

                    if (!boundingBox.contains(blockX, blockY, blockZ))
                        continue;

                    Material blockType = world.getBlockAt(blockX, blockY, blockZ).getType();

                    // voxelData = voxelData.view(
                    //     observation.size(0),
                    //     MinecraftEnvironment.GRID_SIZE_XZ,
                    //     MinecraftEnvironment.GRID_SIZE_XZ,
                    //     MinecraftEnvironment.GRID_SIZE_Y
                    // ).unsqueeze(1); // (B, 1, X, Z, Y)

                    int index = (x + GRID_SIZE_XZ / 2) * GRID_SIZE_XZ * GRID_SIZE_Y +
                        (z + GRID_SIZE_XZ / 2) * GRID_SIZE_Y +
                        y;

                    if (blockType.isSolid()) {
                        voxelGrid.index_put_(
                            new TensorIndexVector(new TensorIndex(index)),
                            new Scalar(1)
                        );
                    }
                }
            }
        }

        // Set position in block
        Tensor positionInBlock = observation.positionInBlock();
        float x = (float) (position.x() - (int) (position.x()));
        float y = (float) (position.y() - (int) (position.y()));
        float z = (float) (position.z() - (int) (position.z()));
        positionInBlock.index_put_(new TensorIndexVector(new TensorIndex(0)), new Scalar(x));
        positionInBlock.index_put_(new TensorIndexVector(new TensorIndex(1)), new Scalar(y));
        positionInBlock.index_put_(new TensorIndexVector(new TensorIndex(2)), new Scalar(z));

        // Set velocity
        Tensor velocity = observation.velocity();
        Vec3 agentVelocity = agent.getDeltaMovement();
        velocity.index_put_(new TensorIndexVector(new TensorIndex(0)), new Scalar(agentVelocity.x()));
        velocity.index_put_(new TensorIndexVector(new TensorIndex(1)), new Scalar(agentVelocity.y()));
        velocity.index_put_(new TensorIndexVector(new TensorIndex(2)), new Scalar(agentVelocity.z()));

        // Set look direction
        Tensor lookDirection = observation.lookDirection();
        Vec3 lookDirectionVec = agent.getLookAngle();
        lookDirection.index_put_(new TensorIndexVector(new TensorIndex(0)), new Scalar(lookDirectionVec.x()));
        lookDirection.index_put_(new TensorIndexVector(new TensorIndex(1)), new Scalar(lookDirectionVec.y()));
        lookDirection.index_put_(new TensorIndexVector(new TensorIndex(2)), new Scalar(lookDirectionVec.z()));

        // Set control states
        observation.jumping().index_put_(new TensorIndexVector(new TensorIndex(0)), new Scalar(agent.jumping ? 1 : 0));
        observation.sprinting().index_put_(new TensorIndexVector(new TensorIndex(0)), new Scalar(agent.actionPack.sprinting ? 1 : 0));
        observation.sneaking().index_put_(new TensorIndexVector(new TensorIndex(0)), new Scalar(agent.actionPack.sneaking ? 1 : 0));
        observation.onGround().index_put_(new TensorIndexVector(new TensorIndex(0)), new Scalar(agent.onGround ? 1 : 0));

        // Set goal direction
        Tensor goalDirection = observation.goalDirection();
        Vec3 goalDirectionVec = goalPosition.subtract(position);
        goalDirectionVec.normalize();
        goalDirection.index_put_(new TensorIndexVector(new TensorIndex(0)), new Scalar(goalDirectionVec.x));
        goalDirection.index_put_(new TensorIndexVector(new TensorIndex(1)), new Scalar(goalDirectionVec.y));
        goalDirection.index_put_(new TensorIndexVector(new TensorIndex(2)), new Scalar(goalDirectionVec.z));


        // Log the observation to the agent's debug log
        agent.displayObservation(observation);

        return observation;
    }

    public Info getInfo() {
        double distanceToGoal = agent.position().distanceTo(goalPosition);

        BlockDisplay bblock = new BlockDisplayBuilder(Debugger.WORLD).block(Material.RED_STAINED_GLASS.createBlockData()).build();
        bblock.setTransformationMatrix(new Matrix4f().scale(0.1f));
        bblock.teleport(new Location(Debugger.WORLD, agent.position().x, agent.position().y, agent.position().z));


        Info info = new Info(distanceToGoal);
        agent.displayInfo(info);

        return info;
    }

    public ResetResult reset() {
        this.currentStep = 0;
        this.agent.reset(roomLocation.clone().add(8.5, 1.5, 8.5));

        // this.goalPosition = new Location(
        //     roomLocation.getWorld(),
        //     roomLocation.getX() + 8.5 + (Math.random() - 0.5) * GRID_SIZE_XZ,
        //     roomLocation.getY() + 1.5 + (Math.random() - 0.5) * GRID_SIZE_Y,
        //     roomLocation.getZ() + 8.5 + (Math.random() - 0.5) * GRID_SIZE_XZ
        // );
        // this.goalPosition = new Vec3(
        //     (roomLocation.getX() + 8.5 + (Math.random() - 0.5) * GRID_SIZE_XZ),
        //     (roomLocation.getY() + 1),
        //     (roomLocation.getZ() + 8.5 + (Math.random() - 0.5) * GRID_SIZE_XZ)
        // );
        // if (Math.random() < 0.5) {
        //     this.goalPosition = new Vec3(
        //         roomLocation.getX() + (Math.random() - 0.5) * GRID_SIZE_XZ,
        //         (roomLocation.getY() + 1),
        //         roomLocation.getZ() + (Math.random() < 0.5 ? (-GRID_SIZE_XZ) : (GRID_SIZE_XZ)) / 2f
        //     );
        // } else {
        //     this.goalPosition = new Vec3(
        //         roomLocation.getX() + (Math.random() < 0.5 ? (-GRID_SIZE_XZ) : (GRID_SIZE_XZ)) / 2f,
        //         (roomLocation.getY() + 1),
        //         roomLocation.getZ() + (Math.random() - 0.5) * GRID_SIZE_XZ
        //     );
        // }
        double[] randomPoint = getRandomPointInCircle(GRID_SIZE_XZ / 2f, GRID_SIZE_XZ);
        this.goalPosition = new Vec3(
            roomLocation.getX() + 8.5 + randomPoint[0],
            roomLocation.getY() + 1,
            roomLocation.getZ() + 8.5 + randomPoint[1]
        );
        BlockDisplay bblock = new BlockDisplayBuilder(Debugger.WORLD).block(Material.LIME_STAINED_GLASS.createBlockData()).build();
        bblock.setTransformationMatrix(new Matrix4f().scale(0.1f));
        bblock.teleport(new Location(Debugger.WORLD, goalPosition.x, goalPosition.y, goalPosition.z));

        Material randomConcrete = new Material[]{
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

                if (offsetX > roomSize && offsetX < (16 - roomSize) && offsetZ > roomSize && offsetZ < (16 - roomSize)) {
                    roomLocation.getWorld().getBlockAt(worldX, 0, worldZ).setType(randomConcrete);
                }

            }
        }

        roomLocation.getWorld().getBlockAt(roomLocation.getBlockX() + 8, 0, roomLocation.getBlockZ() + 8).setType(Material.GOLD_BLOCK);
        roomLocation.getWorld().getBlockAt((int) goalPosition.x, (int) roomLocation.getBlockY(), (int) goalPosition.z).setType(Material.EMERALD_BLOCK);

        Observation observation = getObservation();
        Info info = getInfo();
        previousDistanceToGoal = info.distanceToGoal();
        return new ResetResult(observation, info);
    }

    public StepResult step(Tensor actionTensor) {
        LOGGER.info("Stepping in MinecraftEnvironment with actions");
        Action action = new Action(actionTensor);

        this.currentStep++;

        LOGGER.info("Updating agent action pack with action before: {}", agent.actionPack);
        agent.actionPack.setSprinting(action.sprinting() == 1);
        agent.actionPack.setSneaking(action.sneaking() == 1);
        if (action.jumping() == 1) {
            agent.actionPack.start(EntityPlayerActionPack.ActionType.JUMP, EntityPlayerActionPack.Action.once());
        }

        Vec2 rotation = action.lookChange();
        LOGGER.info("Setting agent rotation: [{}, {}]", rotation.x, rotation.y);
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
        LOGGER.info("Movement keys: {}", movementKeys);
        int moveForward = (movementKeys.forward() == 1 ? 1 : 0) - (movementKeys.backward() == 1 ? 1 : 0);
        int moveRight = (movementKeys.right() == 1 ? 1 : 0) - (movementKeys.left() == 1 ? 1 : 0);

        agent.actionPack.setForward(moveForward);
        agent.actionPack.setStrafing(moveRight);

        LOGGER.info("Updating agent action pack with action after: {}", agent.actionPack);

        // --- Calculate rewards
        Info info = getInfo();

        LOGGER.info("Calculating reward based on distance to goal: {}", info.distanceToGoal());

        double reward = (this.previousDistanceToGoal - info.distanceToGoal()) * 10.0; // Reward based on distance to goal
        previousDistanceToGoal = info.distanceToGoal();

        reward -= 0.5; // Small penalty for each step taken

        LOGGER.info("Current step: {}, Reward: {}", this.currentStep, reward);

        boolean terminated = false;
        if (info.distanceToGoal() < GOAL_THRESHOLD) {
            LOGGER.info("Goal reached! Distance to goal: {}", info.distanceToGoal());
            reward += 200.0; // Large reward for reaching the goal
            terminated = true;
        }

        boolean truncated = this.currentStep > this.args.numSteps;

        LOGGER.info("Truncated step: {}", truncated);

        return new StepResult(getObservation(), reward, terminated, truncated, info);
    }

    public boolean isReady() {
        return agent != null && agent.isReady();
    }
}
