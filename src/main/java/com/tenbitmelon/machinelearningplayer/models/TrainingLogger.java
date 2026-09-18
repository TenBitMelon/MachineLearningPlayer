package com.tenbitmelon.machinelearningplayer.models;

import com.tenbitmelon.machinelearningplayer.ExperimentConfig;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Locale;
import java.util.UUID;

public class TrainingLogger {
    private final BufferedWriter writer;

    public TrainingLogger(ExperimentConfig args) {
        String logDir = "training/" + args.experimentId;
        String experimentId = UUID.randomUUID().toString();
        String startTime = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));

        try {
            Files.createDirectories(Paths.get(logDir));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        String metaFilePath = logDir + "/meta.txt";
        try (BufferedWriter metaWriter = new BufferedWriter(new FileWriter(metaFilePath))) {
            metaWriter.write("experiment_id=" + experimentId + "\n");
            metaWriter.write("start_time=" + startTime + "\n");
            metaWriter.write("args=" + args.toString() + "\n");
        } catch (IOException e) {
            throw new RuntimeException("Failed to write metadata file", e);
        }

        try {
            String filePath = logDir + "/log.csv";
            this.writer = new BufferedWriter(new FileWriter(filePath, true));

            writer.write("timestamp,iteration,learning_rate,value_loss,policy_loss,entropy,old_approx_kl,approx_kl,clipfrac,explained_variance,iteration_time,SPS,num_terminations,num_truncations,average_rewards,total_rewards,bow_selected_steps,bow_drawing_steps,bow_fully_drawn_steps,shield_using_steps,gpu_mem_used_nvidia,gpu_mem_total_nvidia,gpu_util_nvidia,gpu_temp_nvidia,torch_allocated_bytes_current,torch_allocated_bytes_peak,torch_reserved_bytes_current,torch_reserved_bytes_peak,torch_active_bytes_current,torch_active_bytes_peak,torch_inactive_split_bytes_current,torch_inactive_split_bytes_peak,torch_requested_bytes_current,torch_requested_bytes_peak,torch_num_alloc_retries,torch_num_ooms,java_native_used,javacpp_registered_bytes,javacpp_registered_count,java_heap_used,os_available_physical_bytes,os_total_physical_bytes,javacpp_deallocator_thread_alive\n");
            writer.flush();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static String formatDouble(double value) {
        return String.format(Locale.ROOT, "%.6f", value);
    }

    private static String formatNullableDouble(Double value) {
        return value != null ? formatDouble(value) : "";
    }

    public void logStep(long iteration, double learningRate, double valueLoss, double policyLoss, double entropy,
                        Double oldApproxKl, Double approxKl, double clipfrac, double explainedVariance, double iterationTime, double sps,
                        int numTerminations, int numTruncations, double averageRewards, double totalRewards,
                        int bowSelectedSteps, int bowDrawingSteps, int bowFullyDrawnSteps, int shieldUsingSteps,
                        long gpuMemUsedNvidia, long gpuMemTotalNvidia, int gpuUtilNvidia, int gpuTempNvidia,
                        long torchAllocatedBytesCurrent, long torchAllocatedBytesPeak,
                        long torchReservedBytesCurrent, long torchReservedBytesPeak,
                        long torchActiveBytesCurrent, long torchActiveBytesPeak,
                        long torchInactiveSplitBytesCurrent, long torchInactiveSplitBytesPeak,
                        long torchRequestedBytesCurrent, long torchRequestedBytesPeak,
                        long torchNumAllocRetries, long torchNumOoms,
                        long javaNativeUsed, long javaCppRegisteredBytes, long javaCppRegisteredCount,
                        long javaHeapUsed, long osAvailablePhysicalBytes, long osTotalPhysicalBytes,
                        boolean javaCppDeallocatorThreadAlive) throws IOException {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        String[] values = new String[]{
            timestamp,
            Long.toString(iteration),
            formatDouble(learningRate),
            formatDouble(valueLoss),
            formatDouble(policyLoss),
            formatDouble(entropy),
            formatNullableDouble(oldApproxKl),
            formatNullableDouble(approxKl),
            formatDouble(clipfrac),
            formatDouble(explainedVariance),
            formatDouble(iterationTime),
            formatDouble(sps),
            Integer.toString(numTerminations),
            Integer.toString(numTruncations),
            formatDouble(averageRewards),
            formatDouble(totalRewards),
            Integer.toString(bowSelectedSteps),
            Integer.toString(bowDrawingSteps),
            Integer.toString(bowFullyDrawnSteps),
            Integer.toString(shieldUsingSteps),
            Long.toString(gpuMemUsedNvidia),
            Long.toString(gpuMemTotalNvidia),
            Integer.toString(gpuUtilNvidia),
            Integer.toString(gpuTempNvidia),
            Long.toString(torchAllocatedBytesCurrent),
            Long.toString(torchAllocatedBytesPeak),
            Long.toString(torchReservedBytesCurrent),
            Long.toString(torchReservedBytesPeak),
            Long.toString(torchActiveBytesCurrent),
            Long.toString(torchActiveBytesPeak),
            Long.toString(torchInactiveSplitBytesCurrent),
            Long.toString(torchInactiveSplitBytesPeak),
            Long.toString(torchRequestedBytesCurrent),
            Long.toString(torchRequestedBytesPeak),
            Long.toString(torchNumAllocRetries),
            Long.toString(torchNumOoms),
            Long.toString(javaNativeUsed),
            Long.toString(javaCppRegisteredBytes),
            Long.toString(javaCppRegisteredCount),
            Long.toString(javaHeapUsed),
            Long.toString(osAvailablePhysicalBytes),
            Long.toString(osTotalPhysicalBytes),
            javaCppDeallocatorThreadAlive ? "1" : "0"
        };
        writer.write(String.join(",", values));
        writer.write('\n');
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

