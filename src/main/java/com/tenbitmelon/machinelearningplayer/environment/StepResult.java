package com.tenbitmelon.machinelearningplayer.environment;

public record StepResult(Observation observation, double reward, boolean terminated,
                         boolean truncated) implements AutoCloseable {
    public int logicalOrTerminationAndTruncation() {
        return (terminated || truncated) ? 1 : 0;
    }

    @Override
    public void close() {
        observation.close();
    }
}
