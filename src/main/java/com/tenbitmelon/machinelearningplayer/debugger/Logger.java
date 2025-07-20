package com.tenbitmelon.machinelearningplayer.debugger;

import net.kyori.adventure.text.logger.slf4j.ComponentLogger;
import org.slf4j.event.Level;

import java.util.HashMap;

public class Logger {

    public HashMap<Level, Boolean> enabledLevels = new HashMap<>();

    public Logger() {
        enabledLevels.put(Level.TRACE, true);
        enabledLevels.put(Level.DEBUG, true);
        enabledLevels.put(Level.INFO, true);
        enabledLevels.put(Level.WARN, true);
        enabledLevels.put(Level.ERROR, true);
    }

    public void log(Level level, String message) {
        if (enabledLevels.getOrDefault(level, false)) {
            System.out.println("[" + level.name() + "] " + message);
        }
    }

    public void debug(String message) {
        log(Level.DEBUG, message);
    }

    public void debug(String message, Object... args) {
        log(Level.DEBUG, String.format(message, args));
    }

    public void info(String message) {
        log(Level.INFO, message);
    }

    public void info(String message, Object... args) {
        log(Level.INFO, String.format(message, args));
    }

    public void warn(String message) {
        log(Level.WARN, message);
    }

    public void warn(String message, Object... args) {
        log(Level.WARN, String.format(message, args));
    }

    public void error(String message) {
        log(Level.ERROR, message);
    }

    public void error(String message, Object... args) {
        log(Level.ERROR, String.format(message, args));
    }

    public void trace(String message) {
        log(Level.TRACE, message);
    }

    public void trace(String message, Object... args) {
        log(Level.TRACE, String.format(message, args));
    }

    public boolean isEnabled(Level level) {
        return enabledLevels.getOrDefault(level, false);
    }

    public void setEnabled(Level level, boolean enabled) {
        enabledLevels.put(level, enabled);
    }

}
