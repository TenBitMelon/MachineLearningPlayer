package com.tenbitmelon.machinelearningplayer.environment;

import com.tenbitmelon.machinelearningplayer.MachineLearningPlayer;
import com.tenbitmelon.machinelearningplayer.agent.Agent;
import com.tenbitmelon.machinelearningplayer.agent.EntityPlayerActionPack;
import com.tenbitmelon.machinelearningplayer.debugger.ui.TextWindow;
import com.tenbitmelon.machinelearningplayer.models.ExperimentConfig;
import com.tenbitmelon.machinelearningplayer.models.TrainingManager;
import com.tenbitmelon.machinelearningplayer.util.BlockDisplayBuilder;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.phys.Vec2;
import net.minecraft.world.phys.Vec3;
import org.bukkit.*;
import org.bukkit.craftbukkit.CraftServer;
import org.bukkit.craftbukkit.inventory.CraftItemStack;
import org.bukkit.entity.Display;
import org.bukkit.entity.TextDisplay;
import org.bukkit.inventory.ItemStack;
import org.bytedeco.pytorch.Tensor;
import org.joml.Vector3d;

import java.util.concurrent.CompletableFuture;

import static com.tenbitmelon.machinelearningplayer.MachineLearningPlayer.CURRENT_MODE;
import static com.tenbitmelon.machinelearningplayer.MachineLearningPlayer.WORLD;
import static com.tenbitmelon.machinelearningplayer.util.Utils.szudzikUnpairing;

public class MinecraftEnvironment {

    public static final int NORMALIZATION_DISTANCE = 15;
    private static int nextEnvironmentId = 0;
    public final Vec3 centerPosition;
    public final int environmentId;
    private final ExperimentConfig args;
    public Agent agent;
    public LivingEntity targetEntity;
    // TextWindow environmentLog;
    private int currentStep = 0;
    private float lastKnownTargetHealth = 0.0f;
    private float lastKnownMyHealth = 0.0f;
    private float lastKnownDistanceToTarget = 0.0f;

    public MinecraftEnvironment(ExperimentConfig args) {
        this.args = args;
        this.environmentId = nextEnvironmentId++;

        int[] coords = szudzikUnpairing(this.environmentId / 2);
        // Location roomLocation = new Location(WORLD, coords[0] * 16 + 16, 0, coords[1] * 16 + 16);

        // environmentLog = new TextWindow(Display.Billboard.VERTICAL, TextDisplay.TextAlignment.LEFT);
        // Vector3d logpos = roomLocation.toVector().toVector3d().mul(1, 0, 1);
        // if (this.environmentId % 2 == 0) logpos.add(5.0f, 4.0f, 5.0f);
        // else logpos.add(11.0f, 4.0f, 11.0f);
        // environmentLog.setPosition(logpos);

        Chunk chunk = WORLD.getChunkAt(coords[0] + 1, coords[1] + 1);
        chunk.load();

        int startX = chunk.getX() * 16;
        int startZ = chunk.getZ() * 16;

        double centerX = startX + 8.0;
        double centerZ = startZ + 8.0;
        this.centerPosition = new Vec3(centerX, 0.0, centerZ);

        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                for (int y = -16; y < 64; y++) {
                    WORLD.getBlockAt(startX + x, y, startZ + z).setType(Material.AIR);
                }
                WORLD.getBlockAt(startX + x, -1, startZ + z).setType(Material.BARRIER);
            }
        }

        int i = (int) (Math.random() * 15);
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
        }[i];
        Material randomWool = new Material[]{
            Material.WHITE_WOOL,
            Material.ORANGE_WOOL,
            Material.MAGENTA_WOOL,
            Material.LIGHT_BLUE_WOOL,
            Material.YELLOW_WOOL,
            Material.LIME_WOOL,
            Material.PINK_WOOL,
            Material.GRAY_WOOL,
            Material.LIGHT_GRAY_WOOL,
            Material.CYAN_WOOL,
            Material.PURPLE_WOOL,
            Material.BLUE_WOOL,
            Material.BROWN_WOOL,
            Material.GREEN_WOOL,
            Material.RED_WOOL
        }[i];

        for (int offsetX = 0; offsetX < 16; offsetX++) {
            for (int offsetZ = 0; offsetZ < 16; offsetZ++) {

                int worldX = startX + offsetX;
                int worldZ = startZ + offsetZ;

                // Checker grid pattern (2x2 squares)
                if (offsetX >= 1 && offsetX <= 14 && offsetZ >= 1 && offsetZ <= 14) {
                    int gridX = (offsetX - 1) / 2;
                    int gridZ = (offsetZ - 1) / 2;
                    boolean isConcreteSquare = (gridX + gridZ) % 2 == 0;

                    if (isConcreteSquare) {
                        WORLD.getBlockAt(worldX, -1, worldZ).setType(randomConcrete);
                    } else {
                        WORLD.getBlockAt(worldX, -1, worldZ).setType(randomWool);
                    }
                }
            }
        }

        /*
        // Generate random slope parameters
        double slopeAngle = Math.random() * Math.PI * 2; // Random direction (0 to 2π)
        double slopeGradient = 0.1 + Math.random() * 0.3; // Random steepness (0.1 to 0.4 blocks per block)

        // Calculate slope direction vector
        double slopeDx = Math.cos(slopeAngle);
        double slopeDz = Math.sin(slopeAngle);

        // Then modify your floor-building loop:
        for (int offsetX = 0; offsetX < 16; offsetX++) {
            for (int offsetZ = 0; offsetZ < 16; offsetZ++) {
                int worldX = startX + offsetX;
                int worldZ = startZ + offsetZ;

                // Calculate distance along slope direction from center
                double relativeX = offsetX - 8.0;
                double relativeZ = offsetZ - 8.0;
                double distanceAlongSlope = (relativeX * slopeDx + relativeZ * slopeDz);

                // Calculate height based on slope
                int baseY = (int) Math.round(distanceAlongSlope * slopeGradient);

                // Clear above and fill below the sloped surface
                for (int y = -16; y < 64; y++) {
                    if (y < baseY) {
                        // Fill below with barrier or solid block
                        WORLD.getBlockAt(worldX, y, worldZ).setType(Material.BARRIER);
                    } else if (y == baseY) {
                        // This is the surface - apply your checkerboard pattern
                        if (offsetX >= 1 && offsetX <= 14 && offsetZ >= 1 && offsetZ <= 14) {
                            int gridX = (offsetX - 1) / 2;
                            int gridZ = (offsetZ - 1) / 2;
                            boolean isConcreteSquare = (gridX + gridZ) % 2 == 0;

                            WORLD.getBlockAt(worldX, y, worldZ).setType(
                                isConcreteSquare ? randomConcrete : randomWool
                            );
                        } else {
                            WORLD.getBlockAt(worldX, y, worldZ).setType(Material.BARRIER);
                        }
                    } else {
                        // Clear air above
                        WORLD.getBlockAt(worldX, y, worldZ).setType(Material.AIR);
                    }
                }
            }
        }
         */

        double[] randomPointInCircle = getRandomPointInCircle(2, 8);
        Location agentLocation = new Location(WORLD, centerPosition.x + randomPointInCircle[0], 0, centerPosition.z + randomPointInCircle[1]);
        if (this.environmentId % 2 == 0) {
            agentLocation = new Location(WORLD, centerPosition.x - 3.0, 0, centerPosition.z - 7.0);
        } else {
            agentLocation = new Location(WORLD, centerPosition.x + 3.0, 0, centerPosition.z - 7.0);
        }

        MinecraftServer server = ((CraftServer) Bukkit.getServer()).getServer();
        CompletableFuture<Agent> completableFuture = Agent.spawn(server, agentLocation);
        completableFuture.whenComplete((agent, throwable) -> {
            if (throwable == null) {
                this.agent = agent;
                this.reset();
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

    public void setTarget(LivingEntity target) {
        this.targetEntity = target;
    }

    public Observation getObservation() {
        // LOGGER.info("Getting observation for environment " + this.environmentId + " at step " + this.currentStep + ", agent: " + this.agent.getName());
        float pitchScaled = (agent.getXRot() / 90.0f); // Normalize pitch to [-1, 1] where 1 is looking straight up and -1 is looking straight down

        // getYRot is in Degrees
        // yRot expects Radians
        float yawRadians = (float) Math.toRadians(agent.getYRot());
        // yRot rotates clockwise around the Y axis, which is the opposite of what I expected,
        // so we don't need to negate the angle because its already doing that
        Vec3 opponentDirectionWorldSpace = targetEntity.position().subtract(agent.position());
        float opponentDistance = (float) opponentDirectionWorldSpace.length() / NORMALIZATION_DISTANCE;
        opponentDirectionWorldSpace = opponentDirectionWorldSpace.normalize();
        Vec3 opponentDirectionLocalSpace = opponentDirectionWorldSpace.yRot(yawRadians);

        // Opponent velocity
        Vec3 opponentVelocity = targetEntity.getDeltaMovement();
        opponentVelocity = opponentVelocity.yRot(yawRadians);

        // My Velocity
        Vec3 agentVelocity = agent.getDeltaMovement();
        // TODO: Normalize based on max expected velocity
        agentVelocity.multiply(1.0 / 5.0, 1.0 / 20.0, 1.0 / 5.0); // falling from 15 blocks is 26.41 m/s (8 blocks is 20.95 m/s)

        // Attack cooldown
        float attackStrengthTicker = agent.getAttackStrengthScale(0.0f); // not actually attack cooldown, but it's the damage scaling that I assume is 0-1

        // My health
        float myHealth = agent.getHealth() / agent.getMaxHealth();

        Observation observation = new Observation(
            pitchScaled,
            agent.actionPack.sprinting,
            agent.actionPack.sneaking,
            agent.onGround,
            attackStrengthTicker,
            myHealth,
            agentVelocity,
            opponentDirectionLocalSpace,
            opponentDistance,
            opponentVelocity
        );

        agent.displayObservation(observation);

        return observation;
    }

    public ResetResult reset() {
        // environmentLog.clearLines();
        this.currentStep = 0;

        double minRadius = 1.0;
        double maxRadius = 1.0;

        if (CURRENT_MODE == MachineLearningPlayer.Mode.TRAINING) {
            minRadius += 1.0 / 3000.0 * TrainingManager.iteration;
            maxRadius += 6.0 / 3000.0 * TrainingManager.iteration;
        } else {
            minRadius = 3.0;
            maxRadius = 8.0;
        }


        double[] randomPointInCircle = getRandomPointInCircle(minRadius, maxRadius);
        Vec3 agentLocation = centerPosition.add(randomPointInCircle[0], 0, randomPointInCircle[1]);

        this.agent.reset(agentLocation);

        agent.getInventory().setSelectedSlot(0);
        ItemStack itemStack = Material.WOODEN_SWORD.asItemType().createItemStack();
        agent.setItemInHand(InteractionHand.MAIN_HAND, ((CraftItemStack) itemStack).handle);

        // Assuming that when I get reset, the target is also reset and at full health
        lastKnownTargetHealth = targetEntity.getMaxHealth();
        lastKnownMyHealth = agent.getMaxHealth();
        lastKnownDistanceToTarget = 0.0f;

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

        action.close();
    }

    public StepResult postTickStep() {
        float myHealth = agent.getHealth();
        float targetHealth = targetEntity.getHealth();

        float damageTaken = lastKnownMyHealth - myHealth;
        lastKnownMyHealth = myHealth;

        float damageDealt = lastKnownTargetHealth - targetHealth;
        lastKnownTargetHealth = targetHealth;

        double distanceTo = agent.position().distanceTo(targetEntity.position());
        if (lastKnownDistanceToTarget == 0.0f) {
            lastKnownDistanceToTarget = (float) distanceTo;
        }
        float deltaDistance = lastKnownDistanceToTarget - (float) distanceTo;
        lastKnownDistanceToTarget = (float) distanceTo;

        boolean terminated = false;
        float reward = 0.0f;

        reward += 0.05f * damageDealt;
        reward += -0.02f * damageTaken;
        // reward += -0.001f; // timestep cost

        if (myHealth <= 0 && targetHealth > 0) {
            // I LOST (I died, other is still up)
            reward += -10.0f;
            terminated = true;
        } else if (myHealth > 0 && targetHealth <= 0) {
            // I WON (I am alive, other died)
            reward += 10.0f;
            terminated = true;
        } else if (myHealth <= 0 && targetHealth <= 0) {
            // DRAW / DOUBLE KO (Both died in the same tick)
            reward += -5.0f; // Penalty for dying, but not as bad as losing outright
            terminated = true;
        }

        if (distanceTo < 3.0f) {
            reward += 0.001f; // small reward for being close to the target
        }
        if (distanceTo > 1.5f) {
            reward += deltaDistance * 0.001f;
        }

        boolean truncated = this.currentStep > this.args.numSteps;

        if (truncated) {
            if (myHealth > targetHealth) {
                // I had more health when time ran out
                reward += 0.5f;
            } else if (myHealth < targetHealth) {
                // Opponent had more health when time ran out
                reward += -0.5f;
            }
        }

        Observation observation = getObservation();

        return new StepResult(observation, reward, terminated, truncated);
    }


    public boolean isReady() {
        return agent != null && agent.isReady();
    }
}
