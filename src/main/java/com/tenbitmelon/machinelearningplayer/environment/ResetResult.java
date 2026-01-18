package com.tenbitmelon.machinelearningplayer.environment;

public record ResetResult(Observation observation) implements AutoCloseable {
    @Override
    public void close() {
        observation.close();
    }
}
