package com.tenbitmelon.machinelearningplayer.agent;

import com.mojang.authlib.GameProfile;
import com.tenbitmelon.machinelearningplayer.debugger.ui.ControlsWindow;
import com.tenbitmelon.machinelearningplayer.debugger.ui.UIElement;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.Control;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.TextControl;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.VariableControl;
import com.tenbitmelon.machinelearningplayer.environment.Info;
import com.tenbitmelon.machinelearningplayer.environment.Observation;
import net.kyori.adventure.text.Component;
import net.minecraft.core.BlockPos;
import net.minecraft.core.UUIDUtil;
import net.minecraft.network.DisconnectionDetails;
import net.minecraft.network.protocol.PacketFlow;
import net.minecraft.network.protocol.game.ClientboundEntityPositionSyncPacket;
import net.minecraft.network.protocol.game.ClientboundRotateHeadPacket;
import net.minecraft.network.protocol.game.ServerboundClientCommandPacket;
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
import net.minecraft.world.level.block.entity.SkullBlockEntity;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.portal.TeleportTransition;
import org.bukkit.Location;
import org.bukkit.craftbukkit.CraftWorld;
import org.bukkit.event.player.PlayerGameModeChangeEvent;
import org.joml.Vector3d;

import java.util.ArrayList;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

import static com.tenbitmelon.machinelearningplayer.util.Utils.tensorString;

public class Agent extends ServerPlayer {

    public static int agentCount = 0;
    public final EntityPlayerActionPack actionPack = new EntityPlayerActionPack(this);
    final ControlsWindow debugWindow = new ControlsWindow();
    final TextControl infoDisplayControl = new TextControl("Latest Info: ");
    private final ArrayList<Control> observationSectionControls = new ArrayList<>();
    private boolean ready = false;

    public Agent(MinecraftServer server, ServerLevel level, GameProfile gameProfile, ClientInformation clientInformation) {
        super(server, level, gameProfile, clientInformation);

        debugWindow.addText("Agent: " + this.getName().getString());
        debugWindow.addControl(new VariableControl(Component.text("Ready"), () -> ready));
        debugWindow.addControl(new VariableControl(Component.text("Sneaking"), () -> actionPack.sneaking ? "Yes" : "No"));
        debugWindow.addControl(new VariableControl(Component.text("Sprinting"), () -> actionPack.sprinting ? "Yes" : "No"));
        debugWindow.addControl(new VariableControl(Component.text("Forward"), () -> actionPack.forward));
        debugWindow.addControl(new VariableControl(Component.text("Strafing"), () -> actionPack.strafing));
        debugWindow.addControl(new VariableControl(Component.text("Jumping"), () -> actionPack.hasAction(EntityPlayerActionPack.ActionType.JUMP) ? "Yes" : "No"));
        debugWindow.addControl(new VariableControl(Component.text("P Jump"), () -> this.jumping ? "Yes" : "No"));
        debugWindow.addControl(new TextControl(""));
        debugWindow.addControl(new TextControl("Info Display:"));
        debugWindow.addControl(infoDisplayControl);


    }

    public static CompletableFuture<Agent> spawn(MinecraftServer server, Location location) {
        CompletableFuture<Agent> agentFuture = new CompletableFuture<>();

        String username = UUID.randomUUID().toString().substring(0, 3) + agentCount++;
        // String username = String.valueOf(agentCount++);
        // -- Stolen from Carpet Mod: https://github.com/gnembon/fabric-carpet --

        ServerLevel worldIn = server.overworld();
        GameProfileCache.setUsesAuthentication(false);
        GameProfile gameprofile = null;
        try {
            // gameprofile = server.getProfileCache().get(username).orElse(null); // findByName  .orElse(null)
        } finally {
            GameProfileCache.setUsesAuthentication(server.isDedicatedServer() && server.usesAuthentication());
        }
        if (gameprofile == null) {
            gameprofile = new GameProfile(UUIDUtil.createOfflinePlayerUUID(username), username);
        }
        GameProfile finalGP = gameprofile;
        String name = gameprofile.getName();

        // SkullBlockEntity.fetchGameProfile(name).whenCompleteAsync((p, t) -> {
        //     if (t != null) {
        //         agentFuture.completeExceptionally(t);
        //         return;
        //     }

        GameProfile current = finalGP;
        // if (p.isPresent()) {
        //     current = p.get();
        // }

        Agent instance = new Agent(server, worldIn, current, ClientInformation.createDefault());
        instance.snapTo(location.getX(), location.getY(), location.getZ(), 0.0f, 0.0f);
        server.getPlayerList().placeNewPlayer(new FakeClientConnection(PacketFlow.SERVERBOUND), instance, new CommonListenerCookie(current, 0, instance.clientInformation(), false));
        instance.snapTo(location.getX(), location.getY(), location.getZ(), 0.0f, 0.0f);
        instance.teleportTo(worldIn, location.getX(), location.getY(), location.getZ(), Set.of(), 0.0f, 0.0f, true);
        instance.setHealth(20.0F);
        instance.unsetRemoved();
        instance.getAttribute(Attributes.STEP_HEIGHT).setBaseValue(0.6F);
        instance.gameMode.changeGameModeForPlayer(GameType.SURVIVAL, PlayerGameModeChangeEvent.Cause.PLUGIN, null);
        server.getPlayerList().broadcastAll(new ClientboundRotateHeadPacket(instance, (byte) (instance.yHeadRot * 256 / 360)));
        server.getPlayerList().broadcastAll(ClientboundEntityPositionSyncPacket.of(instance));
        instance.entityData.set(DATA_PLAYER_MODE_CUSTOMISATION, (byte) 0x7f); // show all model layers (incl. capes)
        // instance.debugWindow.setAnchor(Bukkit.getEntity(instance.getUUID()));
        instance.debugWindow.setPosition(new Vector3d(instance.getX(), instance.getY() + 3, instance.getZ()));

        instance.ready = true;

        agentFuture.complete(instance);
        // }, server);

        return agentFuture;
    }

    public void displayObservation(Observation observation) {
        if (!UIElement.ALLOW_UPDATES) return;
        observationSectionControls.forEach(debugWindow::removeControl);
        observationSectionControls.clear();

        observationSectionControls.add(new TextControl(""));
        observationSectionControls.add(new TextControl("Latest Observation:"));
        observationSectionControls.add(new TextControl("Position in Block", tensorString(observation.positionInBlock())));
        observationSectionControls.add(new TextControl("Velocity", tensorString(observation.velocity())));
        observationSectionControls.add(new TextControl("Look Direction", tensorString(observation.lookDirection())));
        observationSectionControls.add(new TextControl("Jumping", tensorString(observation.jumping())));
        observationSectionControls.add(new TextControl("Sprinting", tensorString(observation.sprinting())));
        observationSectionControls.add(new TextControl("Sneaking", tensorString(observation.sneaking())));
        observationSectionControls.add(new TextControl("On Ground", tensorString(observation.onGround())));
        observationSectionControls.add(new TextControl("Goal Direction", tensorString(observation.goalDirection())));

        for (Control control : observationSectionControls) {
            debugWindow.addControl(control);
        }
    }

    public void displayInfo(Info info) {
        if (!UIElement.ALLOW_UPDATES) return;
        infoDisplayControl.setValue(String.valueOf(info.distanceToGoal()));
    }

    public boolean isReady() {
        return ready;
    }


    @Override
    public void onEquipItem(final EquipmentSlot slot, final ItemStack previous, final ItemStack stack) {
        if (!isUsingItem()) super.onEquipItem(slot, previous, stack);
    }

    public void kill(net.minecraft.network.chat.Component reason) {
        shakeOff();

        this.getServer().schedule(new TickTask(this.getServer().getTickCount(), () -> {
            this.connection.onDisconnect(new DisconnectionDetails(reason));
        }));
    }

    @Override
    public void tick() {
        // LOGGER.info("Agent {} tick", this.getName().getString()); // AGENTS GET TICKED BEFORE the loop tick
        actionPack.onUpdate();
        if (this.getServer().getTickCount() % 10 == 0) {
            this.connection.resetPosition();
            this.serverLevel().getChunkSource().move(this);
        }
        try {
            super.tick();
            this.doTick();
        } catch (NullPointerException ignored) {
            // happens with that paper port thingy - not sure what that would fix, but hey
            // the game not gonna crash violently.
        }

        if (UIElement.ALLOW_UPDATES) debugWindow.refresh();
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

    public EntityPlayerActionPack getActionPack() {
        return this.actionPack;
    }

    public void reset(Location location) {
        this.reset();
        this.actionPack.stopAll();

        float yRot = (float) (Math.random() * 180.0f - 90.0f);
        float xRot = (float) (Math.random() * 360.0f - 180.0f);
        this.snapTo(location.getX(), location.getY(), location.getZ(), yRot, xRot);
        this.teleportTo(((CraftWorld) location.getWorld()).getHandle(), location.getX(), location.getY(), location.getZ(), Set.of(), yRot, xRot, true);
        this.setHealth(20.0F);
    }
}
