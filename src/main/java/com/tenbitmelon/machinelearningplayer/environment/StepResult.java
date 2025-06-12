package com.tenbitmelon.machinelearningplayer.environment;

public record StepResult(Observation observation, double reward, boolean terminated, boolean truncated,
                         Info info) {}
