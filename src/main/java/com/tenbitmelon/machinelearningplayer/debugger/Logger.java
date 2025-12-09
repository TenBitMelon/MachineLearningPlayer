package com.tenbitmelon.machinelearningplayer.debugger;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.cuda.DeviceStats;
import org.bytedeco.pytorch.global.torch_cuda;
import org.slf4j.event.Level;

import java.util.Arrays;
import java.util.HashMap;

import static com.tenbitmelon.machinelearningplayer.MachineLearningPlayer.LOGGER;
import static com.tenbitmelon.machinelearningplayer.models.TrainingManager.device;

public class Logger {

    public final HashMap<Level, Boolean> enabledLevels = new HashMap<>();

    public Logger() {
        enabledLevels.put(Level.TRACE, true);
        enabledLevels.put(Level.DEBUG, true);
        enabledLevels.put(Level.INFO, true);
        enabledLevels.put(Level.WARN, true);
        enabledLevels.put(Level.ERROR, true);
    }

    private static String format(String template, Object... args) {
        for (Object arg : args) {
            String value = "";
            if (arg == null) {
                value = "null";
            } else if (arg.getClass().isArray()) {
                value = Arrays.deepToString(new Object[]{arg});
                value = value.substring(1, value.length() - 1); // Remove brackets
            } else {
                value = arg.toString();
            }
            template = template.replaceFirst("\\{}", value);
        }
        return template;
    }

    private void log(Level level, String message, Object... args) {
        if (enabledLevels.getOrDefault(level, false)) {
            System.out.printf("[%s] %s%n", level.name(), format(message, args));
        }
    }

    public void memory() {
        // StackTraceElement stackTraceElement = Thread.currentThread().getStackTrace()[2];
        // DeviceStats deviceStats = torch_cuda.getAllocator().getDeviceStats(device.index());
        // long nativeUsed = Pointer.physicalBytes(); // in bytes
        // log(Level.INFO, "T:{},\tC:{}\tL:{},\tA:{},\tN:{}",
        //     System.currentTimeMillis(),
        //     stackTraceElement.getClassName(),
        //     stackTraceElement.getLineNumber(),
        //     deviceStats.allocated_bytes().current(), // in bytes
        //     nativeUsed
        // );
    }

    public void debug(String message, Object... args) {
        log(Level.DEBUG, message, args);
    }

    public void info(String message, Object... args) {
        log(Level.INFO, message, args);
    }

    public void warn(String message, Object... args) {
        log(Level.WARN, message, args);
    }

    public void error(String message, Object... args) {
        log(Level.ERROR, message, args);
    }


    public void trace(String message, Object... args) {
        log(Level.TRACE, message, args);
    }

    public boolean isEnabled(Level level) {
        return enabledLevels.getOrDefault(level, false);
    }

    public void setEnabled(Level level, boolean enabled) {
        enabledLevels.put(level, enabled);
    }

}
