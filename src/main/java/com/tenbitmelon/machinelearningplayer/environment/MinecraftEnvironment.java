package com.tenbitmelon.machinelearningplayer.environment;

import com.tenbitmelon.machinelearningplayer.MachineLearningPlayer;
import com.tenbitmelon.machinelearningplayer.agent.Agent;
import com.tenbitmelon.machinelearningplayer.agent.EntityPlayerActionPack;
import com.tenbitmelon.machinelearningplayer.models.ExperimentConfig;
import com.tenbitmelon.machinelearningplayer.models.TrainingManager;
import net.kyori.adventure.text.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.phys.Vec2;
import net.minecraft.world.phys.Vec3;
import org.bukkit.*;
import org.bukkit.craftbukkit.CraftServer;
import org.bukkit.craftbukkit.inventory.CraftItemStack;
import org.bukkit.enchantments.Enchantment;
import org.bukkit.entity.Player;
import org.bukkit.inventory.ItemStack;
import org.bytedeco.pytorch.Tensor;

import java.util.concurrent.CompletableFuture;

import static com.tenbitmelon.machinelearningplayer.MachineLearningPlayer.CURRENT_MODE;
import static com.tenbitmelon.machinelearningplayer.MachineLearningPlayer.WORLD;
import static com.tenbitmelon.machinelearningplayer.util.Utils.*;

public class MinecraftEnvironment {

    public static final int NORMALIZATION_DISTANCE = 15;
    private static int nextEnvironmentId = 0;
    public final Vec3 centerPosition;
    public final int posX;
    public final int posZ;
    public final int environmentId;
    private final ExperimentConfig args;
    public Agent agent;
    public LivingEntity targetEntity;
    // TextWindow environmentLog;
    private int currentStep = 0;
    private float lastKnownTargetHealth = 0.0f;
    private float lastKnownMyHealth = 0.0f;
    private float lastKnownDistanceToTarget = 0.0f;
    private int ticksActionUse = 0;
    private int lastSlotSelected = 0;

    private int[][] heightMap = new int[32][32];

    public MinecraftEnvironment(ExperimentConfig args) {
        this.args = args;
        this.environmentId = nextEnvironmentId++;

        int[] coords = szudzikUnpairing(this.environmentId / 2);

        generateFloorForChunk(coords[0] * 2, 1, coords[1] * 2, 1);
        generateFloorForChunk(coords[0] * 2, 2, coords[1] * 2, 1);
        generateFloorForChunk(coords[0] * 2, 1, coords[1] * 2, 2);
        generateFloorForChunk(coords[0] * 2, 2, coords[1] * 2, 2);

        this.posX = (coords[0] * 2 + 1) * 16;
        this.posZ = (coords[1] * 2 + 1) * 16;
        double centerX = (coords[0] * 2 + 2) * 16.0;
        double centerZ = (coords[1] * 2 + 2) * 16.0;

        this.centerPosition = new Vec3(centerX, 0.0, centerZ);

        Location agentLocation;
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

    public void generateFloorForChunk(int chunkX, int xChunkOffset, int chunkZ, int zChunkOffset) {
        Chunk chunk = WORLD.getChunkAt(chunkX + xChunkOffset, chunkZ + zChunkOffset);
        chunk.load();
        chunk.setForceLoaded(true);

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

        int startX = chunk.getX() * 16;
        int startZ = chunk.getZ() * 16;

        int currChunkId = szudzikPairing(chunk.getX(), chunk.getZ());
        int posXChunkId = szudzikPairing(chunk.getX() + 1, chunk.getZ());
        int posZChunkId = szudzikPairing(chunk.getX(), chunk.getZ() + 1);
        int posXZChunkId = szudzikPairing(chunk.getX() + 1, chunk.getZ() + 1);
        int salt = Bukkit.getCurrentTick();

        double heightA = hashInt(currChunkId + salt) * 8;
        double heightB = hashInt(posXChunkId + salt) * 8;
        double heightC = hashInt(posZChunkId + salt) * 8;
        double heightD = hashInt(posXZChunkId + salt) * 8;

        double area = 128.0;

        // Then modify your floor-building loop:
        for (int offsetX = 0; offsetX < 16; offsetX++) {
            for (int offsetZ = 0; offsetZ < 16; offsetZ++) {
                int worldX = startX + offsetX;
                int worldZ = startZ + offsetZ;

                int baseY;

                if (offsetX + offsetZ >= 16) {
                    // BDC side
                    double areaBPD = 16.0 * (16 - offsetX) / 2.0;
                    double areaPCD = 16.0 * (16 - offsetZ) / 2.0;
                    double areaBCP = area - areaBPD - areaPCD;

                    double barycentricB = areaPCD / area;
                    double barycentricD = areaBCP / area;
                    double barycentricC = areaBPD / area;

                    double heightAtP = barycentricB * heightB + barycentricD * heightD + barycentricC * heightC;
                    baseY = (int) heightAtP;
                } else {
                    // ABC side
                    double areaAPB = offsetZ * 16.0 / 2.0;
                    double areaACP = offsetX * 16.0 / 2.0;
                    double areaPCB = area - areaAPB - areaACP;

                    double barycentricA = areaPCB / area;
                    double barycentricB = areaACP / area;
                    double barycentricC = areaAPB / area;


                    double heightAtP = barycentricA * heightA + barycentricB * heightB + barycentricC * heightC;
                    baseY = (int) heightAtP;
                }

                // Set in the height map
                heightMap[(xChunkOffset - 1) * 16 + offsetX][(zChunkOffset - 1) * 16 + offsetZ] = baseY;

                // Clear above and fill below the sloped surface
                for (int y = -16; y < 64; y++) {
                    if (y < baseY) {
                        // Fill below with barrier or solid block
                        WORLD.getBlockAt(worldX, y, worldZ).setType(Material.BARRIER);
                    } else if (y == baseY) {
                        // This is the surface - apply your checkerboard pattern
                        int gridX = offsetX / 2;
                        int gridZ = offsetZ / 2;
                        boolean isConcreteSquare = (gridX + gridZ) % 2 == 0;

                        WORLD.getBlockAt(worldX, y, worldZ).setType(
                            isConcreteSquare ? randomConcrete : randomWool
                        );
                    } else {
                        // Clear air above
                        WORLD.getBlockAt(worldX, y, worldZ).setType(Material.AIR);
                    }
                }
            }
        }
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

        // Local height map rotated relative to the agent's orientation, centered on the agent, and normalized by an expected max relative height difference (e.g. 10 blocks)
        float[] localHeightMap = new float[Observation.SIZE_LOCAL_HEIGHT_MAP];
        for (int x = 0; x < 7; x++) {
            for (int z = 0; z < 7; z++) {
                int localX = x - 3;
                int localZ = z - 3;

                Vec3 vec = new Vec3(localX, 0.0, localZ).yRot(-yawRadians);

                int worldX = (int) (agent.getX() + vec.x);
                int worldZ = (int) (agent.getZ() + vec.z);

                int envX = worldX - posX;
                int envZ = worldZ - posZ;

                if (envX < 0 || envX >= 32 || envZ < 0 || envZ >= 32) {
                    localHeightMap[x * 7 + z] = 1.0f; // Default value for out-of-bounds
                    continue;
                }

                // Get the height of the block at this world coordinate
                int blockY = heightMap[worldX - posX][worldZ - posZ]; // Use modulo to wrap around the height map
                float relativeHeight = (float) (blockY - agent.getY());
                localHeightMap[x * 7 + z] = relativeHeight / 10.0f; // Normalize by expected max relative height
            }
        }

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
            opponentVelocity,
            localHeightMap
        );

        agent.displayObservation(observation);

        return observation;
    }

    public ResetResult reset() {
        this.currentStep = 0;

        // int[] coords = szudzikUnpairing(this.environmentId / 2);
        // generateFloorForChunk(coords[0] * 2, 1, coords[1] * 2, 1);
        // generateFloorForChunk(coords[0] * 2, 2, coords[1] * 2, 1);
        // generateFloorForChunk(coords[0] * 2, 1, coords[1] * 2, 2);
        // generateFloorForChunk(coords[0] * 2, 2, coords[1] * 2, 2);

        double minRadius = 3.0;
        double maxRadius = 8.0;

        // if (CURRENT_MODE == MachineLearningPlayer.Mode.TRAINING && TrainingManager.iteration < 5000) {
        //     minRadius += 1.0 / 3000.0 * TrainingManager.iteration;
        //     maxRadius += 6.0 / 3000.0 * TrainingManager.iteration;
        // } else {
        //     minRadius = 3.0;
        //     maxRadius = 8.0;
        // }


        double[] randomPointInCircle = getRandomPointInCircle(minRadius, maxRadius);
        int[] randomPointInCircleInt = {(int) randomPointInCircle[0], (int) randomPointInCircle[1]};
        int height;
        if ((randomPointInCircleInt[0] + 16) < 0 || (randomPointInCircleInt[0] + 16) >= 32 || (randomPointInCircleInt[1] + 16) < 0 || (randomPointInCircleInt[1] + 16) >= 32) {
            height = 8;
        } else {
            height = heightMap[16 + randomPointInCircleInt[0]][16 + randomPointInCircleInt[1]];
        }
        Vec3 agentLocation = centerPosition.add(randomPointInCircle[0], height + 1.0, randomPointInCircle[1]);

        this.agent.reset(agentLocation);

        agent.getInventory().setSelectedSlot(0);
        ItemStack itemStack = Material.WOODEN_SWORD.asItemType().createItemStack();
        agent.setItemInHand(InteractionHand.MAIN_HAND, ((CraftItemStack) itemStack).handle);
        itemStack = Material.SHIELD.asItemType().createItemStack();
        agent.setItemInHand(InteractionHand.OFF_HAND, ((CraftItemStack) itemStack).handle);
        itemStack = Material.BOW.asItemType().createItemStack();
        itemStack.addEnchantment(Enchantment.INFINITY, 1);
        agent.getInventory().add(((CraftItemStack) itemStack).handle);
        itemStack = Material.ARROW.asItemType().createItemStack();
        itemStack.setAmount(1);
        agent.getInventory().add(((CraftItemStack) itemStack).handle);

        // Assuming that when I get reset, the target is also reset and at full health
        lastKnownTargetHealth = targetEntity.getMaxHealth();
        lastKnownMyHealth = agent.getMaxHealth();
        lastKnownDistanceToTarget = 0.0f;
        ticksActionUse = 0;
        lastSlotSelected = 0;

        return new ResetResult(getObservation());
    }

    public void preTickStep(Tensor actionTensor) {
        Action action = new Action(actionTensor);

        this.currentStep++;
        agent.actionPack.stopMovement();
        agent.actionPack.stopAllButUse();

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

        int slotChange = action.slotChange();
        agent.getInventory().setSelectedSlot(slotChange);

        int attackUse = action.attackUseItem();
        if (attackUse == 2) {
            if (ticksActionUse == 0 || lastSlotSelected != slotChange) {
                agent.actionPack.start(EntityPlayerActionPack.ActionType.USE, EntityPlayerActionPack.Action.continuous());
            }
            ticksActionUse++;
        } else {
            agent.actionPack.stop(EntityPlayerActionPack.ActionType.USE);
            ticksActionUse = 0;
            if (attackUse == 1) {
                agent.actionPack.start(EntityPlayerActionPack.ActionType.ATTACK, EntityPlayerActionPack.Action.once());
            }
        }

        lastSlotSelected = slotChange;
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
        reward += -0.001f; // timestep cost

        // Holding use hint reward
        reward += 0.00001f * ticksActionUse;

        // Reward shooting a bow
        if (lastSlotSelected == 1 && ticksActionUse == 20) {
            // The bow is fully drawn at 20 ticks, so give a reward for that
            reward += 0.1f;
        }

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
        // {
        //     Player target = Bukkit.getPlayer("melonboy10");
        //     if (target.getX() > posX && target.getZ() > posZ && target.getX() < posX + 32 && target.getZ() < posZ + 32) {
        //         // Target is within this environment's chunk, so we can visualize the local height map
        //
        //         float yawRadians = (float) Math.toRadians(target.getYaw());
        //
        //         // // For loop over the local height map in the observation and spawn particles at the corresponding world coordinates with a height based on the value in the height map for debugging
        //         // Tensor localHeightMap = observation.localHeightMap();
        //         for (int x = 0; x < 7; x++) {
        //             for (int z = 0; z < 7; z++) {
        //                 float heightValue = localHeightMap[(x * 7 + z)];
        //                 Vec3 vec = new Vec3(x - 3, 0.0, z - 3).yRot(-yawRadians);
        //
        //                 double worldX = target.getX() + vec.x;
        //                 double worldY = target.getY() + heightValue * 10.0f + 1.0f; // scale back up to world coordinates
        //                 double worldZ = target.getZ() + vec.z;
        //                 WORLD.spawnParticle(Particle.BUBBLE, worldX, worldY, worldZ, 1, 0, 0, 0, 0, null);
        //             }
        //         }
        //     }
        // }
        // Bubble the whole height map for debugging
        // for (int x = 0; x < 32; x++) {
        //     for (int z = 0; z < 32; z++) {
        //         int blockY = heightMap[x][z];
        //         double worldX = centerPosition.x + x - 16 + 0.5;
        //         double worldY = blockY + 1;
        //         double worldZ = centerPosition.z + z - 16 + 0.5;
        //         WORLD.spawnParticle(Particle.BUBBLE, worldX, worldY, worldZ, 1, 0, 0, 0, 0, null);
        //     }
        // }

        return new StepResult(
            observation,
            reward,
            terminated,
            truncated,
            myHealth,
            targetHealth,
            damageTaken,
            damageDealt,
            distanceTo,
            lastSlotSelected == 1,
            lastSlotSelected == 1 && ticksActionUse > 0,
            lastSlotSelected == 1 && ticksActionUse >= 20,
            lastSlotSelected == 0 && ticksActionUse > 0
        );
    }


    public boolean isReady() {
        return agent != null && agent.isReady();
    }
}
