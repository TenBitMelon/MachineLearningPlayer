package com.tenbitmelon.machinelearningplayer.models;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.UUID;

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

            writer.write("timestamp,iteration,learning_rate,value_loss,policy_loss,entropy,old_approx_kl,approx_kl,clipfrac,explained_variance,iteration_time,SPS,num_terminations,num_truncations,average_rewards,total_rewards,gpu_mem_used,java_native_used\n");
            writer.flush();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    public void logStep(long iteration, double learningRate, double valueLoss, double policyLoss, double entropy,
                        Double oldApproxKl, Double approxKl, double clipfrac, double explainedVariance, double iterationTime, double sps, int numTerminations, int numTruncations, double averageRewards, double totalRewards, long gpuMemUsed, long javaNativeUsed) throws IOException {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        writer.write(String.format("%s,%d,%.6f,%.6f,%.6f,%.6f,%s,%s,%.6f,%.6f,%.6f,%.6f,%d,%d,%.6f,%.6f,%d,%d\n",
            timestamp, iteration, learningRate, valueLoss, policyLoss, entropy,
            oldApproxKl != null ? String.format("%.6f", oldApproxKl) : "",
            approxKl != null ? String.format("%.6f", approxKl) : "",
            clipfrac, explainedVariance, iterationTime, sps, numTerminations, numTruncations, averageRewards, totalRewards,
            gpuMemUsed, javaNativeUsed));
        writer.flush();
    }

    public void close() {
        try {
            if (writer != null)
                writer.close();
        } catch (IOException e) {
            System.err.println("Failed to close the writer");
        }
    }
}

