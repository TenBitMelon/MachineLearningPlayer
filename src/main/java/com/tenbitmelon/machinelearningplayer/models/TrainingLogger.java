package com.tenbitmelon.machinelearningplayer.models;

import java.io.*;
import java.nio.file.*;
import java.time.*;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class TrainingLogger {
    private final BufferedWriter writer;

    public TrainingLogger(ExperimentConfig args, String logDir) {
        String experimentId = UUID.randomUUID().toString();
        String startTime = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));

        try {
            Files.createDirectories(Paths.get(logDir));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        String metaFilePath = logDir + "/" + experimentId + "_meta.txt";
        try (BufferedWriter metaWriter = new BufferedWriter(new FileWriter(metaFilePath))) {
            metaWriter.write("experiment_id=" + experimentId + "\n");
            metaWriter.write("start_time=" + startTime + "\n");
            metaWriter.write("args=" + args.toString() + "\n");
        } catch (IOException e) {
            throw new RuntimeException("Failed to write metadata file", e);
        }

        try {
            String filePath = logDir + "/" + experimentId + ".csv";
            this.writer = new BufferedWriter(new FileWriter(filePath, true));

            writer.write("timestamp,iteration,learning_rate,value_loss,policy_loss,entropy,old_approx_kl,approx_kl,clipfrac,explained_variance,iteration_time,SPS,num_truncations,average_rewards\n");
            writer.flush();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    public void logStep(long iteration, double learningRate, double valueLoss, double policyLoss, double entropy,
                        Double oldApproxKl, Double approxKl, double clipfrac, double explainedVariance, double iterationTime, double sps, int numTerminations, double averageRewards) throws IOException {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        writer.write(String.format("%s,%d,%.6f,%.6f,%.6f,%.6f,%s,%s,%.6f,%.6f,%.6f,%.6f,%d,%.6f\n",
            timestamp, iteration, learningRate, valueLoss, policyLoss, entropy,
            oldApproxKl != null ? String.format("%.6f", oldApproxKl) : "",
            approxKl != null ? String.format("%.6f", approxKl) : "",
            clipfrac, explainedVariance, iterationTime, sps, numTerminations, averageRewards));
        writer.flush();
    }

    public void close() {
        try {
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}

