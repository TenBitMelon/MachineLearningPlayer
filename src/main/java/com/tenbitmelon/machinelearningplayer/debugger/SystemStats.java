package com.tenbitmelon.machinelearningplayer.debugger;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.cuda.DeviceStats;
import org.bytedeco.pytorch.global.torch_cuda;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.management.ManagementFactory;

import com.sun.management.OperatingSystemMXBean; // Note: Requires jdk.management module

public class SystemStats {

    public static HardwareMetrics snapshot(byte deviceIndex) {
        // 1. CPU & Java Stats
        OperatingSystemMXBean osBean = (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
        double cpuLoad = osBean.getCpuLoad() * 100.0;
        long heapUsed = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();

        // 2. JavaCPP / Native Stats
        // "physicalBytes" tracks memory allocated via JavaCPP pointers (C++ heap)
        long nativeUsed = Pointer.physicalBytes();

        // 3. GPU Stats via PyTorch Allocator (Fast, accurate for Torch tensors)
        // Note: This only sees memory managed by PyTorch.
        DeviceStats torchStats = torch_cuda.getAllocator().getDeviceStats(deviceIndex);
        long torchAllocated = torchStats.allocated_bytes().current();

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
        if (gpuUsed == 0) gpuUsed = torchAllocated;

        return new HardwareMetrics(cpuLoad, heapUsed, nativeUsed, gpuUsed, gpuTotal, gpuUtil, gpuTemp);
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
        long gpuMemUsed,
        long gpuMemTotal,
        int gpuUtil,
        int gpuTemp
    ) {}
}
