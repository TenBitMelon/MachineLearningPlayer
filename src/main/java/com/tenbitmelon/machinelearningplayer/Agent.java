package com.tenbitmelon.machinelearningplayer;

import com.mojang.authlib.GameProfile;
import com.mojang.brigadier.context.CommandContext;
import io.papermc.paper.command.brigadier.CommandSourceStack;
import net.kyori.adventure.text.TextComponent;
import net.minecraft.commands.arguments.GameModeArgument;
import net.minecraft.core.BlockPos;
import net.minecraft.core.UUIDUtil;
import net.minecraft.network.DisconnectionDetails;
import net.minecraft.network.chat.Component;
import net.minecraft.network.chat.contents.TranslatableContents;
import net.minecraft.network.protocol.PacketFlow;
import net.minecraft.network.protocol.game.ClientboundEntityPositionSyncPacket;
import net.minecraft.network.protocol.game.ClientboundRotateHeadPacket;
import net.minecraft.network.protocol.game.ServerboundClientCommandPacket;
import net.minecraft.resources.ResourceKey;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.TickTask;
import net.minecraft.server.level.ClientInformation;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.network.CommonListenerCookie;
import net.minecraft.server.players.GameProfileCache;
import net.minecraft.world.damagesource.DamageSource;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.entity.ai.attributes.Attributes;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.food.FoodData;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.entity.SkullBlockEntity;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.portal.TeleportTransition;
import org.bukkit.Location;
import org.bukkit.craftbukkit.CraftServer;
import org.bukkit.event.player.PlayerGameModeChangeEvent;

import java.util.Set;

@SuppressWarnings("UnstableApiUsage")
public class Agent extends ServerPlayer {

    public Runnable fixStartingPosition = () -> {};

    public Agent(MinecraftServer server, ServerLevel level, GameProfile gameProfile, ClientInformation clientInformation) {
        super(server, level, gameProfile, clientInformation);
    }

    public static void spawn(CommandContext<CommandSourceStack> context, Location location) {
        System.out.println("Spawning MLPlayer at " + location);
        GameType gamemode = GameType.CREATIVE;

        CommandSourceStack source = context.getSource();
        MinecraftServer server = ((CraftServer) source.getExecutor().getServer()).getServer();
        String username = "MLPlayer_" + UUIDUtil.createOfflinePlayerUUID("MLPlayer").toString().substring(0, 8);


        // -- Stolen from Carpet Mod: https://github.com/gnembon/fabric-carpet --

        // prolly half of that crap is not necessary, but it works
        ServerLevel worldIn = server.overworld();
        GameProfileCache.setUsesAuthentication(false);
        GameProfile gameprofile;
        try {
            gameprofile = server.getProfileCache().get(username).orElse(null); // findByName  .orElse(null)
        } finally {
            GameProfileCache.setUsesAuthentication(server.isDedicatedServer() && server.usesAuthentication());
        }
        if (gameprofile == null) {
            gameprofile = new GameProfile(UUIDUtil.createOfflinePlayerUUID(username), username);
        }
        GameProfile finalGP = gameprofile;

        // We need to mark this player as spawning so that we do not
        // try to spawn another player with the name while the profile
        // is being fetched - preventing multiple players spawning
        String name = gameprofile.getName();

        System.out.println("Starting callback");

        SkullBlockEntity.fetchGameProfile(name).whenCompleteAsync((p, t) -> {
            try {
                System.out.println("Callback completed");
                // Always remove the name, even if exception occurs
                if (t != null) {
                    return;
                }

                GameProfile current = finalGP;
                if (p.isPresent()) {
                    current = p.get();
                }

                System.out.println("Creating Agent instance");

                Agent instance = new Agent(server, worldIn, current, ClientInformation.createDefault());
                System.out.println("1");
                instance.fixStartingPosition = () -> instance.snapTo(location.getX(), location.getY(), location.getZ(), 0.0f, 0.0f);
                System.out.println("2");
                server.getPlayerList().placeNewPlayer(new FakeClientConnection(PacketFlow.SERVERBOUND), instance, new CommonListenerCookie(current, 0, instance.clientInformation(), false));
                System.out.println("3");
                instance.teleportTo(worldIn, location.getX(), location.getY(), location.getZ(), Set.of(), 0.0f, 0.0f, true);
                System.out.println("4");
                instance.setHealth(20.0F);
                System.out.println("5");
                instance.unsetRemoved();
                System.out.println("6");
                instance.getAttribute(Attributes.STEP_HEIGHT).setBaseValue(0.6F);
                System.out.println("7");
                instance.gameMode.changeGameModeForPlayer(gamemode, PlayerGameModeChangeEvent.Cause.DEFAULT_GAMEMODE, null);
                System.out.println("8");
                server.getPlayerList().broadcastAll(new ClientboundRotateHeadPacket(instance, (byte) (instance.yHeadRot * 256 / 360)));// instance.dimension);
                System.out.println("9");
                server.getPlayerList().broadcastAll(ClientboundEntityPositionSyncPacket.of(instance));// instance.dimension);
                System.out.println("10");
                // instance.world.getChunkManager(). updatePosition(instance);
                System.out.println("11");
                instance.entityData.set(DATA_PLAYER_MODE_CUSTOMISATION, (byte) 0x7f); // show all model layers (incl. capes)
                System.out.println("12");

                System.out.println("Done!");
            } catch (Exception e) {
                e.printStackTrace();
                System.out.println("Failed to spawn MLPlayer: " + e.getMessage());
            }
        }, server);
    }

    @Override
    public void onEquipItem(final EquipmentSlot slot, final ItemStack previous, final ItemStack stack) {
        if (!isUsingItem()) super.onEquipItem(slot, previous, stack);
    }

    public void kill(Component reason) {
        shakeOff();

        this.getServer().schedule(new TickTask(this.getServer().getTickCount(), () -> {
            this.connection.onDisconnect(new DisconnectionDetails(reason));
        }));
    }

    @Override
    public void tick() {
        if (this.getServer().getTickCount() % 10 == 0) {
            this.connection.resetPosition();
            // this.level().getChunkSource().move(this);
        }
        try {
            super.tick();
            this.doTick();
        } catch (NullPointerException ignored) {
            // happens with that paper port thingy - not sure what that would fix, but hey
            // the game not gonna crash violently.
        }


    }

    private void shakeOff() {
        if (getVehicle() instanceof Player) stopRiding();
        for (Entity passenger : getIndirectPassengers()) {
            if (passenger instanceof Player) passenger.stopRiding();
        }
    }

    @Override
    public void die(DamageSource cause) {
        shakeOff();
        super.die(cause);
        setHealth(20);
        this.foodData = new FoodData();
        kill(this.getCombatTracker().getDeathMessage());
    }

    @Override
    public String getIpAddress() {
        return "127.0.0.1";
    }

    @Override
    public boolean allowsListing() {
        return true;
    }

    @Override
    protected void checkFallDamage(double y, boolean onGround, BlockState state, BlockPos pos) {
        doCheckFallDamage(0.0, y, 0.0, onGround);
    }

    @Override
    public ServerPlayer teleport(TeleportTransition serverLevel) {
        super.teleport(serverLevel);
        if (wonGame) {
            ServerboundClientCommandPacket p = new ServerboundClientCommandPacket(ServerboundClientCommandPacket.Action.PERFORM_RESPAWN);
            connection.handleClientCommand(p);
        }

        // If above branch was taken, *this* has been removed and replaced, the new instance has been set
        // on 'our' connection (which is now theirs, but we still have a ref).
        if (connection.player.isChangingDimension()) {
            connection.player.hasChangedDimension();
        }
        return connection.player;
    }

    public static Agent respawnFake(MinecraftServer server, ServerLevel level, GameProfile profile, ClientInformation cli) {
        return new Agent(server, level, profile, cli);
    }
}
