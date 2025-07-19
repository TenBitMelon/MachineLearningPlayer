package com.tenbitmelon.machinelearningplayer.debugger.ui.controls;

import com.tenbitmelon.machinelearningplayer.debugger.Debugger;
import com.tenbitmelon.machinelearningplayer.util.InteractionBuilder;
import com.tenbitmelon.machinelearningplayer.util.Utils;
import net.kyori.adventure.text.Component;
import net.kyori.adventure.text.TextComponent;
import net.kyori.adventure.text.event.ClickEvent;
import net.kyori.adventure.text.format.TextDecoration;
import net.kyori.adventure.text.serializer.plain.PlainTextComponentSerializer;
import org.apache.commons.lang3.tuple.MutableTriple;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.entity.Interaction;
import org.bukkit.event.player.PlayerInteractEntityEvent;
import org.jetbrains.annotations.NotNull;
import org.joml.Vector3d;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.Supplier;

public abstract class Control {
    public Component label;
    private HashMap<UUID, ClickEvent> interactionEntityToClickEvent = new HashMap<>();
    private HashMap<UUID, Interaction> uuidToInteraction = new HashMap<>();

    public Control(Component label) {
        this.label = label;
    }

    public @NotNull Component renderLabel() {
        return label.append(Component.text("  "));
    }

    public int getLabelWidth() {
        return Utils.stringWidth(PlainTextComponentSerializer.plainText().serialize(this.renderLabel()));
    }

    public abstract @NotNull Component renderValue();

    public int getValueWidth() {
        return Utils.stringWidth(PlainTextComponentSerializer.plainText().serialize(this.renderValue()));
    }

    public @NotNull Component placeAndRender(int longestLabel, int longestValue, int line, Vector3d position, double direction) {
        TextComponent.Builder builder = Component.text();

        int lineWidth = longestLabel + longestValue + 1; // +1 for the left pad in text displays
        int gapWidth = longestLabel - getLabelWidth();

        builder.append(this.renderLabel());

        int numSpaces = 0;
        int remainingPixels = 0;
        char fillerChar = ' ';

        if (gapWidth >= 12) {
            //     Combination of 4 & 5 pixel spacers
            numSpaces = gapWidth / 4;
            remainingPixels = gapWidth % 4;
        } else if (gapWidth >= 2) {
            //     Combination of 2 & 3 pixel spacers
            fillerChar = '.';
            numSpaces = gapWidth / 2;
            remainingPixels = gapWidth % 2;
        } else {
            //     Single pixel spacer
            fillerChar = '.';
            numSpaces = 1;
        }

        numSpaces -= remainingPixels;

        builder
            .append(Component.text(String.valueOf(fillerChar).repeat(numSpaces)))
            .append(Component.text(String.valueOf(fillerChar).repeat(remainingPixels)).decoration(TextDecoration.BOLD, true));

        Component component = this.renderValue();
        List<MutableTriple<Integer, Integer, ClickEvent>> clickEvents = Utils.findClickEvents(component);

        if (clickEvents != null && !clickEvents.isEmpty()) {
            updateInteractionEntities(clickEvents, longestLabel, lineWidth, position, direction, line);
        } else {
            clearAllInteractions();
        }

        builder.append(component);
        return builder.build();
    }

    private void updateInteractionEntities(List<MutableTriple<Integer, Integer, ClickEvent>> clickEvents, int longestLabel, int lineWidth, Vector3d position, double direction, int line) {
        int neededInteractions = clickEvents.size();
        int currentInteractions = interactionEntityToClickEvent.size();
        ArrayList<UUID> uuids = new ArrayList<>(interactionEntityToClickEvent.keySet());

        // Reuse existing interactions and update their click events and positions
        for (int i = 0; i < Math.min(neededInteractions, currentInteractions); i++) {
            MutableTriple<Integer, Integer, ClickEvent> clickEvent = clickEvents.get(i);
            Interaction interaction = uuidToInteraction.get(uuids.get(i));
            interactionEntityToClickEvent.put(interaction.getUniqueId(), clickEvent.getRight());

            float interactionWidth = (clickEvent.getMiddle() - clickEvent.getLeft()) * (1f / 40f);


            updateInteractionPosition(interaction, clickEvent, longestLabel, lineWidth, position, direction, line, interactionWidth);
        }

        // Create new interactions if we need more
        if (neededInteractions > currentInteractions) {
            for (int i = currentInteractions; i < neededInteractions; i++) {
                MutableTriple<Integer, Integer, ClickEvent> clickEvent = clickEvents.get(i);
                float interactionWidth = (clickEvent.getMiddle() - clickEvent.getLeft()) * (1f / 40f);

                Interaction interaction = new InteractionBuilder(Debugger.WORLD)
                    .width(interactionWidth)
                    .height(0.25f)
                    .build();
                interaction.setResponsive(true);

                updateInteractionPosition(interaction, clickEvent, longestLabel, lineWidth, position, direction, line, interactionWidth);

                UUID interactionId = interaction.getUniqueId();
                uuidToInteraction.put(interactionId, interaction);
                interactionEntityToClickEvent.put(interactionId, clickEvent.getRight());
            }
        }
        // Remove excess interactions if we have too many
        else if (neededInteractions < currentInteractions) {
            for (int i = currentInteractions - 1; i >= neededInteractions; i--) {
                Interaction interaction = uuidToInteraction.remove(uuids.get(i));
                if (interaction != null) {
                    interaction.remove();
                }
            }
        }
    }

    private void updateInteractionPosition(Interaction interaction, MutableTriple<Integer, Integer, ClickEvent> clickEvent, int longestLabel, int lineWidth, Vector3d position, double direction, int line, float interactionWidth) {
        Vector3d planarPosition = new Vector3d(
            -(lineWidth / 2.0 * (1f / 40f)) + ((longestLabel + clickEvent.getLeft()) * (1f / 40f)) + (interactionWidth / 2f),
            line * 0.25f,
            -(interactionWidth / 2f)
        );

        double angle = Math.toRadians(-direction + 180);
        planarPosition.rotateAxis(angle, 0, 1, 0);
        planarPosition.add(position);

        interaction.teleport(new Location(Debugger.WORLD, planarPosition.x, planarPosition.y, planarPosition.z));
    }

    private void clearAllInteractions() {
        for (Interaction interaction : uuidToInteraction.values()) {
            interaction.remove();
        }
        uuidToInteraction.clear();
        interactionEntityToClickEvent.clear();
    }

    public boolean onInteract(PlayerInteractEntityEvent event) {
        if (event.getRightClicked() instanceof Interaction interaction) {
            UUID interactionId = interaction.getUniqueId();
            if (interactionEntityToClickEvent.containsKey(interactionId)) {
                ClickEvent clickEvent = interactionEntityToClickEvent.get(interactionId);
                if (clickEvent != null) {
                    Bukkit.dispatchCommand(event.getPlayer(), clickEvent.value().substring(1));
                    return true;
                }
            }
        }
        return false;
    }
}
