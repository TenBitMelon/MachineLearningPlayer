package com.tenbitmelon.machinelearningplayer.debugger;

import com.tenbitmelon.machinelearningplayer.debugger.ui.ControlsWindow;
import com.tenbitmelon.machinelearningplayer.debugger.ui.TextWindow;
import com.tenbitmelon.machinelearningplayer.debugger.ui.UIElement;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.BooleanControl;
import com.tenbitmelon.machinelearningplayer.debugger.ui.controls.ButtonControl;
import com.tenbitmelon.machinelearningplayer.models.TrainingManager;
import com.tenbitmelon.machinelearningplayer.util.Utils;
import net.kyori.adventure.text.Component;
import org.bukkit.Bukkit;
import org.bukkit.World;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerInteractEntityEvent;
import org.bukkit.event.player.PlayerMoveEvent;
import org.joml.Vector3d;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.UUID;

public class Debugger {

    public static final HashMap<UUID, UIElement> AWAITING_ANCHOR = new HashMap<>();
    public static World WORLD = Bukkit.getWorlds().getFirst();
    public static HashMap<UUID, UIElement> DRAGGED_ELEMENTS = new HashMap<>();
    static ArrayList<UIElement> elements = new ArrayList<>();

    static {
        ControlsWindow mainDebugWindow = new ControlsWindow();
        mainDebugWindow.addText("ML Player Debugger");
        mainDebugWindow.addText("Author: TenBitMelon");
        mainDebugWindow.addText(" ");
        mainDebugWindow.addControl(new BooleanControl(Component.text("Run Training"), () -> TrainingManager.runTraining, (value) -> TrainingManager.runTraining = value));
        mainDebugWindow.addControl(new ButtonControl(Component.text("Single Step"), () -> {
            TrainingManager.runTraining = true;
            TrainingManager.trainingStep();
            TrainingManager.runTraining = false;
        }));
        mainDebugWindow.setPosition(new Vector3d(16, 3, 16));
        addElement(mainDebugWindow);
    }

    private Debugger() {
        throw new IllegalStateException("Utility class");
    }

    public static void organizeElementsAround(Vector3d position) {
        Vector3d offset = new Vector3d(position);
        for (UIElement element : elements) {
            element.setPosition(offset);
            //      offset.add(0, element.height + 4, 0);
        }
    }

    public static void addElement(UIElement element) {
        elements.add(element);
    }

    public static void start() {
        //    Later add a restore from file
    }

    public static void stop() {
        for (UIElement element : elements) {
            element.destroy();
        }
    }

    public static void update() {
        for (UIElement element : elements) {
            element.update();
        }
    }

    public static class DebugListener implements Listener {
        public static final DebugListener INSTANCE = new DebugListener();

        private DebugListener() {}

        @EventHandler(ignoreCancelled = true)
        public void onPlayerMove(PlayerMoveEvent event) {
            if (!DRAGGED_ELEMENTS.containsKey(event.getPlayer().getUniqueId())) {
                return;
            }
            UIElement uiElement = DRAGGED_ELEMENTS.get(event.getPlayer().getUniqueId());
            double facing = Utils.roundRotation(event.getPlayer().getLocation().getYaw(), 8);
            uiElement.setPosition(event.getPlayer().getLocation().getDirection().toVector3d().mul(2).add(event.getPlayer().getEyeLocation().toVector().toVector3d()), facing);

        }

        @EventHandler(ignoreCancelled = true)
        public void onPlayerInteractEntity(PlayerInteractEntityEvent event) {
            for (UIElement element : elements) {
                if (element.onInteract(event)) {
                    return;
                }
            }
        }
    }
}
