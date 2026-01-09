package com.tenbitmelon.machinelearningplayer.agent;

import io.netty.channel.embedded.EmbeddedChannel;
import net.minecraft.network.Connection;
import net.minecraft.network.PacketListener;
import net.minecraft.network.PacketSendListener;
import net.minecraft.network.ProtocolInfo;
import net.minecraft.network.protocol.Packet;
import net.minecraft.network.protocol.PacketFlow;
import net.minecraft.network.protocol.game.ClientboundMoveEntityPacket;
import net.minecraft.network.protocol.game.ClientboundSetEntityMotionPacket;
import org.jetbrains.annotations.Nullable;

public class FakeClientConnection extends Connection {

    private final Agent agent;

    public FakeClientConnection(PacketFlow p, Agent agent) {
        super(p);
        // compat with adventure-platform-fabric. This does NOT trigger other vanilla handlers for establishing a channel
        // also makes #isOpen return true, allowing enderpearls to teleport fake players
        // ((ClientConnectionInterface)this).setChannel(new EmbeddedChannel());
        this.channel = new EmbeddedChannel();

        this.agent = agent;
    }

    @Override
    public void send(Packet<?> packet, @Nullable PacketSendListener listener, boolean flush) {
        if (packet instanceof ClientboundSetEntityMotionPacket cbsemp) {
            if (cbsemp.getId() == agent.getId()) {
                agent.actionPack.setVelocity(cbsemp.getXa(), cbsemp.getYa(), cbsemp.getZa());
                agent.hurtMarked = true; // force position update next tick
            }
        }
    }

    @Override
    public void setReadOnly() {
    }

    @Override
    public void handleDisconnection() {
    }

    @Override
    public void setListenerForServerboundHandshake(PacketListener packetListener) {
    }

    @Override
    public <T extends PacketListener> void setupInboundProtocol(ProtocolInfo<T> protocolInfo, T packetListener) {
    }
}