package com.tenbitmelon.machinelearningplayer.environment;

public record StepResult(Observation observation, double reward, boolean terminated, boolean truncated) {
    public int logicalOrTerminationAndTruncation() {
        return (terminated || truncated) ? 1 : 0;
    }
}
