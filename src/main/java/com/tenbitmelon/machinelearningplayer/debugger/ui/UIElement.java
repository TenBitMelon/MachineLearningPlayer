package com.tenbitmelon.machinelearningplayer.debugger.ui;

import com.tenbitmelon.machinelearningplayer.debugger.Debugger;
import com.tenbitmelon.machinelearningplayer.util.InteractionBuilder;
import com.tenbitmelon.machinelearningplayer.util.TextDisplayBuilder;
import org.bukkit.Color;
import org.bukkit.entity.Display;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Interaction;
import org.bukkit.entity.TextDisplay;
import org.bukkit.event.player.PlayerInteractEntityEvent;
import org.bukkit.util.Vector;
import org.joml.Matrix4f;
import org.joml.Vector3d;

public abstract class UIElement {

    public static boolean ALLOW_UPDATES = true; // Set to false to disable updates for all UIElements, useful for performance during training
    final TextDisplay anchorText;
    final Interaction anchorInteraction;
    public boolean dirty = true;
    Entity anchorEntity;
    Vector3d position = new Vector3d(0, 0, 0);
    double direction = 180;
    private boolean visible = true;

    public UIElement() {
        anchorText = new TextDisplayBuilder(Debugger.WORLD).text("≡≡≡").billboard(Display.Billboard.VERTICAL).teleportDuration(1).build(); // Braille Pattern "⠿" (U+283F) or Full Square "⬛" (U+2B1B) or Identical to "≡" (U+2261)
        anchorInteraction = new InteractionBuilder(Debugger.WORLD).width(0.5).height(0.3).responsive(true).build();
        Debugger.addElement(this);
    }

    public UIElement(Entity anchor) {
        this();
        this.anchorEntity = anchor;
    }

    public void setPosition(Vector3d position) {
        setPosition(position, direction);
    }

    public void setPosition(Vector3d position, double direction) {
        this.direction = direction;
        this.position = position;
        anchorText.teleport(anchorText.getLocation().set(position.x, position.y - 0.3, position.z));
        anchorInteraction.teleport(anchorInteraction.getLocation().set(position.x, position.y - 0.25, position.z));
    }

    public void destroy() {
        anchorText.remove();
        anchorInteraction.remove();
    }

    public boolean onInteract(PlayerInteractEntityEvent event) {
        if (Debugger.AWAITING_ANCHOR.containsKey(event.getPlayer().getUniqueId()) && Debugger.AWAITING_ANCHOR.get(event.getPlayer().getUniqueId()).equals(this)) {
            if (!event.getPlayer().isSneaking()) {
                if (event.getRightClicked() instanceof Display || event.getRightClicked() instanceof Interaction) {
                    return false; // Prevents anchoring to another TextDisplay
                }
                this.anchorEntity = event.getRightClicked();
            }

            //     Attach this display to the anchor
            Debugger.AWAITING_ANCHOR.remove(event.getPlayer().getUniqueId());

            this.anchorText.setBackgroundColor(null);
            this.update();
            return true;
        }


        if (!event.getRightClicked().equals(anchorInteraction)) return false;

        if (event.getPlayer().isSneaking()) {
            //     Prepare to anchor the display
            if (this.anchorEntity != null) {
                this.anchorEntity = null;
            } else {
                Debugger.AWAITING_ANCHOR.put(event.getPlayer().getUniqueId(), this);
                this.anchorText.setBackgroundColor(Color.AQUA);
            }
            this.update();
            return true;
        }

        // Just move the display
        if (Debugger.DRAGGED_ELEMENTS.containsKey(event.getPlayer().getUniqueId())) {
            Debugger.DRAGGED_ELEMENTS.remove(event.getPlayer().getUniqueId());
            this.anchorText.setBackgroundColor(null);
        } else {
            Debugger.DRAGGED_ELEMENTS.put(event.getPlayer().getUniqueId(), this);
            this.anchorText.setBackgroundColor(Color.GREEN);
        }
        this.update();
        return true;
    }

    public void update() {
        if (!ALLOW_UPDATES) return;
        if (anchorEntity == null) return;

        Vector center = anchorEntity.getBoundingBox().getCenter();
        center.setY(anchorEntity.getBoundingBox().getMaxY() + 1);
        setPosition(center.toVector3d());
    }

    public boolean isVisible() {
        return this.visible;
    }

    public void setVisible(boolean b) {
        this.visible = b;
        if (!b) {
            anchorText.setTransformationMatrix(new Matrix4f().scale(0));
            anchorInteraction.setResponsive(false);
        } else {
            anchorText.setTransformationMatrix(new Matrix4f());
            anchorInteraction.setResponsive(true);
        }
    }

    public void setAnchor(Entity entity) {
        this.anchorEntity = entity;
    }
}
