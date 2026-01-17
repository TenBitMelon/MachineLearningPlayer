package com.tenbitmelon.machinelearningplayer.environment;

import com.tenbitmelon.machinelearningplayer.agent.Agent;
import com.tenbitmelon.machinelearningplayer.agent.EntityPlayerActionPack;
import com.tenbitmelon.machinelearningplayer.debugger.ui.TextWindow;
import com.tenbitmelon.machinelearningplayer.models.ExperimentConfig;
import com.tenbitmelon.machinelearningplayer.util.BlockDisplayBuilder;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.phys.Vec2;
import net.minecraft.world.phys.Vec3;
import org.bukkit.*;
import org.bukkit.craftbukkit.CraftServer;
import org.bukkit.entity.Display;
import org.bukkit.entity.TextDisplay;
import org.bytedeco.pytorch.Tensor;
import org.joml.Vector3d;

import java.util.concurrent.CompletableFuture;

import static com.tenbitmelon.machinelearningplayer.util.Utils.szudzikUnpairing;

public class MinecraftEnvironment {

    public static final int CIRCLE_RADIUS = 4;
    private static int nextEnvironmentId = 0;
    public final Location roomLocation;
    public final Vec3 centerPosition;
    public final int environmentId;
    private final ExperimentConfig args;
    private final Material randomConcrete;
    public Agent agent;
    public Entity targetEntity;
    TextWindow environmentLog;
    private int currentStep = 0;

    public MinecraftEnvironment(ExperimentConfig args) {
        this.args = args;
        this.environmentId = nextEnvironmentId++;
        World world = Bukkit.getWorlds().getFirst();

        int[] coords = szudzikUnpairing(this.environmentId / 2);
        roomLocation = new Location(world, coords[0] * 16 + 16, 0, coords[1] * 16 + 16);

        environmentLog = new TextWindow(Display.Billboard.VERTICAL, TextDisplay.TextAlignment.LEFT);
        Vector3d logpos = roomLocation.toVector().toVector3d().mul(1, 0, 1);
        if (this.environmentId % 2 == 0) logpos.add(5.0f, 4.0f, 5.0f);
        else logpos.add(11.0f, 4.0f, 11.0f);
        environmentLog.setPosition(logpos);

        Chunk chunk = roomLocation.getChunk();
        chunk.load();

        int startX = chunk.getX() * 16;
        int startZ = chunk.getZ() * 16;

        double centerX = startX + 8.5;
        double centerZ = startZ + 8.5;
        this.centerPosition = new Vec3(centerX, 1.0, centerZ);

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

        for (int offsetX = 0; offsetX < 16; offsetX++) {
            for (int offsetZ = 0; offsetZ < 16; offsetZ++) {

                int worldX = roomLocation.getBlockX() + offsetX;
                int worldZ = roomLocation.getBlockZ() + offsetZ;

                roomLocation.getWorld().getBlockAt(worldX, 0, worldZ).setType(Material.BARRIER);

                // Circle check
                if (Math.sqrt(Math.pow(offsetX - 8, 2) + Math.pow(offsetZ - 8, 2)) <= CIRCLE_RADIUS) {
                    roomLocation.getWorld().getBlockAt(worldX, 0, worldZ).setType(randomConcrete);
                }
            }
        }

        if (this.environmentId == 0) {
            for (int angle = 0; angle < 360; angle += (360 / 12)) {
                double rad = Math.toRadians(angle);
                double x = centerX + CIRCLE_RADIUS * Math.cos(rad);
                double z = centerZ + CIRCLE_RADIUS * Math.sin(rad);
                new BlockDisplayBuilder(world).block(Material.CRIMSON_NYLIUM.createBlockData()).teleport(x, 1.0 - 0.1 + 0.001, z).setScale(0.1f).brightness(new Display.Brightness(15, 15)).build();
            }
        }

        roomLocation.getWorld().getBlockAt(roomLocation.getBlockX() + 8, 0, roomLocation.getBlockZ() + 8).setType(Material.GOLD_BLOCK);

        Location agentLocation;
        if (this.environmentId % 2 == 0) {
            agentLocation = new Location(world, startX + 8.5, 1.5, startZ + 8.5 + CIRCLE_RADIUS / 2.0);
        } else {
            agentLocation = new Location(world, startX + 8.5, 1.5, startZ + 8.5 - CIRCLE_RADIUS / 2.0);
        }

        MinecraftServer server = ((CraftServer) Bukkit.getServer()).getServer();
        CompletableFuture<Agent> completableFuture = Agent.spawn(server, agentLocation);
        completableFuture.whenComplete((agent, throwable) -> {
            if (throwable == null) {
                this.agent = agent;
            }
        });
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

    public void setTarget(Entity target) {
        this.targetEntity = target;
    }

    public Observation getObservation() {
        // LOGGER.info("Getting observation for environment " + this.environmentId + " at step " + this.currentStep + ", agent: " + this.agent.getName());
        float pitchScaled = (agent.getXRot() / 90.0f); // Normalize pitch to [-1, 1] where 1 is looking straight up and -1 is looking straight down

        // Distance to center
        float centerDistance = (float) this.getDistanceToCenter(this.agent);

        // getYRot is in Degrees
        // yRot expects Radians
        float yawRadians = (float) Math.toRadians(agent.getYRot());
        // yRot rotates clockwise around the Y axis, which is the opposite of what I expected,
        // so we don't need to negate the angle because its already doing that
        Vec3 opponentDirectionWorldSpace = targetEntity.position().subtract(agent.position());
        float opponentDistance = (float) opponentDirectionWorldSpace.length();
        opponentDirectionWorldSpace = opponentDirectionWorldSpace.normalize();
        Vec3 opponentDirectionLocalSpace = opponentDirectionWorldSpace.yRot(yawRadians);


        Observation observation = new Observation(
            pitchScaled,
            agent.actionPack.sprinting,
            agent.actionPack.sneaking,
            agent.onGround,
            centerDistance,
            opponentDirectionLocalSpace,
            opponentDistance
        );

        agent.displayObservation(observation);

        return observation;
    }


    public ResetResult reset() {
        environmentLog.clearLines();
        this.currentStep = 0;

        if (this.environmentId % 2 == 0) {
            this.agent.reset(roomLocation.clone().add(8.5, 1.0, 8.5 + CIRCLE_RADIUS / 2.0));
        } else {
            this.agent.reset(roomLocation.clone().add(8.5, 1.0, 8.5 - CIRCLE_RADIUS / 2.0));
        }

        return new ResetResult(getObservation());
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

        int attackUse = action.attackUseItem();
        if (attackUse == 1) {
            agent.actionPack.start(EntityPlayerActionPack.ActionType.ATTACK, EntityPlayerActionPack.Action.once());
        } else if (attackUse == 2) {
            agent.actionPack.start(EntityPlayerActionPack.ActionType.USE, EntityPlayerActionPack.Action.once());
        }
    }

    public StepResult postTickStep() {

        double myDist = getDistanceToCenter(this.agent);
        double oppDist = getDistanceToCenter(targetEntity);

        boolean iAmIn = myDist <= CIRCLE_RADIUS;
        boolean oppIsIn = oppDist <= CIRCLE_RADIUS;

        boolean terminated = false;
        float reward = 0.0f;

        if (iAmIn && oppIsIn) {
            // BATTLE CONTINUES
            reward = 0.01f;
            terminated = false;
        } else if (!iAmIn && oppIsIn) {
            // I LOST (I fell out, other is still in)
            reward = -100.0f;
            terminated = true;
        } else if (iAmIn && !oppIsIn) {
            // I WON (I am in, other fell out)
            reward = 100.0f;
            terminated = true;
        } else if (!iAmIn && !oppIsIn) {
            // DRAW / DOUBLE KO (Both fell out same tick)
            reward = -50.0f; // Penalty for falling, but not as bad as losing outright
            terminated = true;
        }

        boolean truncated = this.currentStep > this.args.numSteps;

        Observation observation = getObservation();

        return new StepResult(observation, reward, terminated, truncated);
    }

    private double getDistanceToCenter(Entity targetAgent) {
        Vec3 pos = targetAgent.position();
        return Math.sqrt(
            Math.pow(pos.x - centerPosition.x, 2) +
                Math.pow(pos.z - centerPosition.z, 2)
        );
    }

    public boolean isReady() {
        return agent != null && agent.isReady();
    }
}
