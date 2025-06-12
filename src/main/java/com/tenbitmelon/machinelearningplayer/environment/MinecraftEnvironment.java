package com.tenbitmelon.machinelearningplayer.environment;

import com.tenbitmelon.machinelearningplayer.agent.Agent;
import com.tenbitmelon.machinelearningplayer.agent.EntityPlayerActionPack;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.phys.Vec3;
import org.bukkit.*;
import org.bukkit.craftbukkit.CraftServer;
import org.bukkit.util.BoundingBox;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorIndex;
import org.bytedeco.pytorch.TensorIndexVector;
import org.joml.Vector3f;

import java.util.concurrent.CompletableFuture;

public class MinecraftEnvironment {

    public static final int GRID_SIZE_XZ = 5;
    public static final int GRID_SIZE_Y = 5;
    public static final int GRID_VOLUME = GRID_SIZE_XZ * GRID_SIZE_XZ * GRID_SIZE_Y;

    public static final int MAX_STEPS_PER_EPISODE = 20;
    public static final double GOAL_THRESHOLD = 0.25;
    private static int nextEnvironmentId = 0;

    private Agent agent;
    private Location roomLocation;
    private BoundingBox boundingBox;
    private int currentStep = 0;
    private Vec3 goalPosition;
    private double previousDistanceToGoal;

    public MinecraftEnvironment() {
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
        int x = (int) (position.x() - Math.floor(position.x()));
        int y = (int) (position.y() - Math.floor(position.y()));
        int z = (int) (position.z() - Math.floor(position.z()));
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

        return observation;
    }

    public Info getInfo() {
        double distanceToGoal = agent.position().distanceTo(goalPosition);
        return new Info(distanceToGoal);
    }

    public ResetResult reset(int seed) {
        this.currentStep = 0;
        this.agent.reset(roomLocation.clone().add(8.5, 1.5, 8.5));

        // this.goalPosition = new Location(
        //     roomLocation.getWorld(),
        //     roomLocation.getX() + 8.5 + (Math.random() - 0.5) * GRID_SIZE_XZ,
        //     roomLocation.getY() + 1.5 + (Math.random() - 0.5) * GRID_SIZE_Y,
        //     roomLocation.getZ() + 8.5 + (Math.random() - 0.5) * GRID_SIZE_XZ
        // );
        this.goalPosition = new Vec3(
            (int) (roomLocation.getX() + 8.5 + (Math.random() - 0.5) * GRID_SIZE_XZ),
            (int) (roomLocation.getY()),
            (int) (roomLocation.getZ() + 8.5 + (Math.random() - 0.5) * GRID_SIZE_XZ)
        );

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

        for (int offsetX = 0; offsetX < 16; offsetX++) {
            for (int offsetZ = 0; offsetZ < 16; offsetZ++) {

                int worldX = roomLocation.getBlockX() + offsetX;
                int worldZ = roomLocation.getBlockZ() + offsetZ;

                roomLocation.getWorld().getBlockAt(worldX, 0, worldZ).setType(Material.BARRIER);

                if (offsetX > 4 && offsetX < 12 && offsetZ > 4 && offsetZ < 12) {
                    roomLocation.getWorld().getBlockAt(worldX, 0, worldZ).setType(randomConcrete);
                }

            }
        }

        roomLocation.getWorld().getBlockAt(roomLocation.getBlockX() + 8, 0, roomLocation.getBlockZ() + 8).setType(Material.GOLD_BLOCK);
        roomLocation.getWorld().getBlockAt((int) goalPosition.x, (int) goalPosition.y, (int) goalPosition.z).setType(Material.EMERALD_BLOCK);

        Observation observation = getObservation();
        Info info = getInfo();
        previousDistanceToGoal = info.distanceToGoal();
        return new ResetResult(observation, info);
    }

    public StepResult step(Tensor actionTensor) {
        Action action = new Action(actionTensor);

        this.currentStep++;

        agent.actionPack.setSprinting(action.sprinting() == 1);
        agent.actionPack.setSneaking(action.sneaking() == 1);
        agent.actionPack.start(EntityPlayerActionPack.ActionType.JUMP, action.jumping() == 1 ? EntityPlayerActionPack.Action.continuous() : EntityPlayerActionPack.Action.once());

        agent.actionPack.turn(action.lookChange());

        /*
        table:
        forward | forwardPressed | backwardPressed
        --------|----------------|----------------
        0       | false          | false
        1       | true           | false
        -1      | false          | true
        0       | true           | true
         */
        int moveForward = (action.moveKeys().forward() ? 1 : 0) - (action.moveKeys().backward() ? 1 : 0);
        int moveRight = (action.moveKeys().right() ? 1 : 0) - (action.moveKeys().left() ? 1 : 0);

        agent.actionPack.setForward(moveForward);
        agent.actionPack.setStrafing(moveRight);


        // --- Calculate rewards
        Info info = getInfo();

        double reward = (this.previousDistanceToGoal - info.distanceToGoal()) * 10.0; // Reward based on distance to goal
        previousDistanceToGoal = info.distanceToGoal();

        reward -= 0.5; // Small penalty for each step taken

        boolean terminated = false;
        if (info.distanceToGoal() < GOAL_THRESHOLD) {
            reward += 200.0; // Large reward for reaching the goal
            terminated = true;
        }

        boolean truncated = this.currentStep > MAX_STEPS_PER_EPISODE;

        return new StepResult(getObservation(), reward, terminated, truncated, info);
    }

    public boolean isReady() {
        return agent != null && agent.isReady();
    }
}
