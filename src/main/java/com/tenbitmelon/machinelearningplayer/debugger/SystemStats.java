package com.tenbitmelon.machinelearningplayer.debugger;

import com.sun.management.OperatingSystemMXBean;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.Stat;
import org.bytedeco.pytorch.cuda.DeviceStats;
import org.bytedeco.pytorch.global.torch_cuda;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.management.ManagementFactory;

public class SystemStats {

    public static HardwareMetrics snapshot(byte deviceIndex) {
        // 1. CPU & Java Stats
        OperatingSystemMXBean osBean = (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
        double cpuLoad = osBean.getCpuLoad() * 100.0;
        long heapUsed = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

        // 2. JavaCPP / Native Stats
        // "physicalBytes" tracks memory allocated via JavaCPP pointers (C++ heap)
        JavaCppDiagnostics.Snapshot javaCppSnapshot = JavaCppDiagnostics.snapshot();
        long nativeUsed = javaCppSnapshot.physicalBytes();

        // 3. GPU Stats via PyTorch Allocator (Fast, accurate for Torch tensors)
        // Note: This only sees memory managed by PyTorch.
        DeviceStats torchStats = torch_cuda.getAllocator().getDeviceStats(deviceIndex);
        Stat allocatedBytes = torchStats.allocated_bytes();
        Stat reservedBytes = torchStats.reserved_bytes();
        Stat activeBytes = torchStats.active_bytes();
        Stat inactiveSplitBytes = torchStats.inactive_split_bytes();
        Stat requestedBytes = torchStats.requested_bytes();

        long torchAllocatedCurrent = allocatedBytes.current();
        long torchAllocatedPeak = allocatedBytes.peak();
        long torchReservedCurrent = reservedBytes.current();
        long torchReservedPeak = reservedBytes.peak();
        long torchActiveCurrent = activeBytes.current();
        long torchActivePeak = activeBytes.peak();
        long torchInactiveSplitCurrent = inactiveSplitBytes.current();
        long torchInactiveSplitPeak = inactiveSplitBytes.peak();
        long torchRequestedCurrent = requestedBytes.current();
        long torchRequestedPeak = requestedBytes.peak();
        long torchNumAllocRetries = torchStats.num_alloc_retries();
        long torchNumOoms = torchStats.num_ooms();

        // 4. GPU Stats via nvidia-smi (Slower, but gets Utilization & Temp)
        long gpuTotal = 0;
        long gpuUsed = 0;
        int gpuUtil = -1;
        int gpuTemp = -1;

        try {
            // Queries: memory.used, memory.total, utilization.gpu, temperature.gpu
            ProcessBuilder pb = new ProcessBuilder("nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
                "--id=" + deviceIndex
            );
            Process p = pb.start();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                String line = reader.readLine();
                if (line != null) {
                    String[] parts = line.split(",");
                    gpuUsed = Long.parseLong(parts[0].trim()) * 1024 * 1024; // MB to Bytes
                    gpuTotal = Long.parseLong(parts[1].trim()) * 1024 * 1024;
                    gpuUtil = Integer.parseInt(parts[2].trim());
                    gpuTemp = Integer.parseInt(parts[3].trim());
                }
            }
        } catch (Exception e) {
            // Fail silently or log if nvidia-smi is missing
        }

        // Fallback: If nvidia-smi failed, use torch stats for 'used' (though it will be lower than actual VRAM usage)
        if (gpuUsed == 0) gpuUsed = torchReservedCurrent != 0 ? torchReservedCurrent : torchAllocatedCurrent;

        allocatedBytes.close();
        reservedBytes.close();
        activeBytes.close();
        inactiveSplitBytes.close();
        requestedBytes.close();
        torchStats.close();

        return new HardwareMetrics(
            cpuLoad,
            heapUsed,
            nativeUsed,
            javaCppSnapshot.registeredBytes(),
            javaCppSnapshot.registeredCount(),
            javaCppSnapshot.availablePhysicalBytes(),
            javaCppSnapshot.totalPhysicalBytes(),
            javaCppSnapshot.deallocatorThreadAlive(),
            gpuUsed,
            gpuTotal,
            gpuUtil,
            gpuTemp,
            torchAllocatedCurrent,
            torchAllocatedPeak,
            torchReservedCurrent,
            torchReservedPeak,
            torchActiveCurrent,
            torchActivePeak,
            torchInactiveSplitCurrent,
            torchInactiveSplitPeak,
            torchRequestedCurrent,
            torchRequestedPeak,
            torchNumAllocRetries,
            torchNumOoms
        );
    }

    public static String formatBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        int exp = (int) (Math.log(bytes) / Math.log(1024));
        char pre = "KMGTPE".charAt(exp - 1);
        return String.format("%.1f %sB", bytes / Math.pow(1024, exp), pre);
    }

    public static String formatBytes(long bytes, int precision) {
        if (bytes < 1024) return bytes + " B";
        int exp = (int) (Math.log(bytes) / Math.log(1024));
        char pre = "KMGTPE".charAt(exp - 1);
        return String.format("%." + precision + "f %sB", bytes / Math.pow(1024, exp), pre);
    }

    public record HardwareMetrics(
        double cpuLoad,
        long javaHeapUsed,
        long javaNativeUsed, // JavaCPP physical bytes
        long javaCppRegisteredBytes,
        long javaCppRegisteredCount,
        long osAvailablePhysicalBytes,
        long osTotalPhysicalBytes,
        boolean javaCppDeallocatorThreadAlive,
        long gpuMemUsed,
        long gpuMemTotal,
        int gpuUtil,
        int gpuTemp,
        long torchAllocatedBytesCurrent,
        long torchAllocatedBytesPeak,
        long torchReservedBytesCurrent,
        long torchReservedBytesPeak,
        long torchActiveBytesCurrent,
        long torchActiveBytesPeak,
        long torchInactiveSplitBytesCurrent,
        long torchInactiveSplitBytesPeak,
        long torchRequestedBytesCurrent,
        long torchRequestedBytesPeak,
        long torchNumAllocRetries,
        long torchNumOoms
    ) {}
}
