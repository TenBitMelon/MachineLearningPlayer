package com.tenbitmelon.machinelearningplayer.debugger;

import com.tenbitmelon.machinelearningplayer.ExperimentConfig;
import com.tenbitmelon.machinelearningplayer.models.TrainingManager;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.javacpp.tools.NativeAllocationTracer;
import org.bytedeco.pytorch.Stat;
import org.bytedeco.pytorch.cuda.BlockInfo;
import org.bytedeco.pytorch.cuda.DeviceStats;
import org.bytedeco.pytorch.cuda.SegmentInfo;
import org.bytedeco.pytorch.cuda.SnapshotInfo;
import org.bytedeco.pytorch.global.torch_cuda;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public final class LeakLogger {

    private final ExperimentConfig args;
    private BufferedWriter writer;
    private long iterStartMs = -1L;

    public LeakLogger(ExperimentConfig args) {
        this.args = args;

        String logDir = "training/" + args.experimentId;

        try {
            Files.createDirectories(Paths.get(logDir));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        try {
            this.writer = new BufferedWriter(new FileWriter(logDir + "/leak_log.csv", false));
            String[] columns = new String[]{
                "timestamp",
                "iteration",
                "phase",
                "ms_since_iter_start",
                "torch_allocated_bytes_current",
                "torch_reserved_bytes_current",
                "torch_active_bytes_current",
                "torch_inactive_split_bytes_current",
                "torch_requested_bytes_current",
                "torch_allocated_bytes_after_emptycache",
                "torch_reserved_bytes_after_emptycache",
                "java_heap_used",
                "gpu_mem_used_nvidia",
                "gpu_mem_total_nvidia",
                "gpu_util_nvidia",
                "gpu_temp_nvidia"
            };

            writer.write(String.join(",", columns));
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static String formatBytes(long bytes) {
        return formatBytes(bytes, 1);
    }

    public static String formatBytes(long bytes, int precision) {
        if (bytes < 1024) return bytes + " B";
        int exp = (int) (Math.log(bytes) / Math.log(1024));
        char pre = "KMGTPE".charAt(exp - 1);
        return String.format("%." + precision + "f %sB", bytes / Math.pow(1024, exp), pre);
    }

    public void iterStart(int iteration) {
        iterStartMs = System.currentTimeMillis();
        probe(iteration, "iter_start");
    }

    public void stepStart(int iteration) {
        probe(iteration, "step");
    }

    public void collectionEnd(int iteration) {
        probe(iteration, "collection_end");
    }

    public void afterGae(int iteration) {
        probe(iteration, "after_gae");
    }

    public void afterEpochs(int iteration) {
        probe(iteration, "after_epochs");
    }

    public void iterEnd(int iteration) {
        probe(iteration, "iter_end", true, true);
    }

    public void snapshotDump(int iteration) {
        if (!args.featureFlags.contains(ExperimentConfig.FeatureFlag.ALLOCATOR_SNAPSHOT))
            return;
        if (iteration % 10 != 0)
            return;


        SnapshotInfo snapshot = null;
        try {
            snapshot = torch_cuda.getAllocator().snapshot();
            StringBuilder sb = new StringBuilder();
            sb.append("== iter ").append(iteration).append('\n');
            long liveBlocks = 0;
            long liveBytes = 0;
            SegmentInfo segments = snapshot.segments();
            long segCount = segments.limit();
            sb.append("seg_total=").append(segCount).append('\n');
            for (long i = 0; i < segCount; i++) {
                SegmentInfo segment = segments.getPointer(i);
                sb.append("SEG size=").append(segment.total_size())
                    .append(" requested=").append(segment.requested_size())
                    .append(" allocated=").append(segment.allocated_size())
                    .append(" active=").append(segment.active_size())
                    .append(" large=").append(segment.is_large())
                    .append(" expandable=").append(segment.is_expandable())
                    .append('\n');
                BlockInfo blocks = segment.blocks();
                long blockCount = blocks.limit();
                for (long b = 0; b < blockCount; b++) {
                    BlockInfo block = blocks.getPointer(b);
                    boolean isAllocated = block.allocated();
                    long blockSize = block.size();
                    sb.append("  BLOCK size=").append(blockSize)
                        .append(" requested=").append(block.requested_size())
                        .append(" allocated=").append(isAllocated)
                        .append(" active=").append(block.active())
                        .append('\n');
                    if (isAllocated) {
                        liveBlocks++;
                        liveBytes += blockSize;
                    }
                }
            }
            sb.append("live_blocks=").append(liveBlocks)
                .append(" live_bytes=").append(liveBytes)
                .append('\n');
            Path dir = Paths.get("training", TrainingManager.args.experimentId, "allocator");
            Files.createDirectories(dir);
            Files.writeString(dir.resolve(iteration + "_allocator.txt"), sb.toString());
        } catch (Throwable e) {
            System.err.println("[leakprobe] snapshot dump failed: " + e.getMessage());
            try {
                Path dir = Paths.get("training", TrainingManager.args.experimentId, "allocator");
                Files.createDirectories(dir);
                Files.writeString(dir.resolve(iteration + "_allocator_FAILED.txt"),
                    "exception: " + e + "\n" + java.util.Arrays.toString(e.getStackTrace()));
            } catch (IOException ignored) {
            }
        } finally {
            if (snapshot != null) {
                try {
                    snapshot.close();
                } catch (Throwable ignored) {
                }
            }
        }
    }

    public void nativeAllocationSnapshot(int iteration) {
        if (TrainingManager.args == null)
            return;
        if (!TrainingManager.args.featureFlags.contains(ExperimentConfig.FeatureFlag.LEAK_PROBE))
            return;

        File logFile = new File("training/" + TrainingManager.args.experimentId + "/sites/");
        logFile.mkdirs();
        String contents = "Native Allocation Tracer Sites for Iteration " + iteration + "\n";
        for (NativeAllocationTracer.Site site : NativeAllocationTracer.getSites()) {
            contents += site.toString() + "\n";
        }
        try {
            Files.writeString(logFile.toPath().resolve(iteration + "_sites.txt"), contents);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public long[] nvidiaGPUSMISnapshot() {
        long gpuTotal = 0;
        long gpuUsed = 0;
        int gpuUtil = -1;
        int gpuTemp = -1;

        byte deviceIndex = TrainingManager.device.index();

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
        }

        return new long[]{gpuUsed, gpuTotal, gpuUtil, gpuTemp};
    }

    private void probe(int iteration, String phase) {
        probe(iteration, phase, false, false);
    }

    private void probe(int iteration, String phase, boolean emptyCacheTest, boolean takeGPUSnapshot) {
        if (!args.featureFlags.contains(ExperimentConfig.FeatureFlag.LEAK_PROBE))
            return;

        try (PointerScope scope = new PointerScope()) {

            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));

            DeviceStats torchStats = torch_cuda.getAllocator().getDeviceStats(TrainingManager.device.index());
            Stat allocatedBytes = torchStats.allocated_bytes();
            Stat reservedBytes = torchStats.reserved_bytes();
            Stat activeBytes = torchStats.active_bytes();
            Stat inactiveSplitBytes = torchStats.inactive_split_bytes();
            Stat requestedBytes = torchStats.requested_bytes();

            long torchAllocatedCurrent = allocatedBytes.current();
            long torchReservedCurrent = reservedBytes.current();
            long torchActiveCurrent = activeBytes.current();
            long inactiveSplit = inactiveSplitBytes.current();
            long torchRequestedCurrent = requestedBytes.current();


            long allocatedAfter = torchAllocatedCurrent;
            long reservedAfter = torchReservedCurrent;
            if (emptyCacheTest) {
                torch_cuda.device_synchronize();
                torch_cuda.getAllocator().emptyCache();

                DeviceStats torchStats2 = torch_cuda.getAllocator().getDeviceStats(TrainingManager.device.index());
                Stat allocatedBytesAfter = torchStats2.allocated_bytes();
                Stat reservedBytesAfter = torchStats2.reserved_bytes();
                allocatedAfter = allocatedBytesAfter.current();
                reservedAfter = reservedBytesAfter.current();
            }

            long[] gpuSnapshot = takeGPUSnapshot ? nvidiaGPUSMISnapshot() : new long[]{-1L, -1L, -1L, -1L};
            long heapUsed = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
            long ms = iterStartMs >= 0 ? System.currentTimeMillis() - iterStartMs : -1L;

            String[] values = new String[]{
                timestamp,
                Integer.toString(iteration),
                phase,
                Long.toString(ms),
                Long.toString(torchAllocatedCurrent),
                Long.toString(torchReservedCurrent),
                Long.toString(torchActiveCurrent),
                Long.toString(inactiveSplit),
                Long.toString(torchRequestedCurrent),
                Long.toString(allocatedAfter),
                Long.toString(reservedAfter),
                Long.toString(heapUsed),
                Long.toString(gpuSnapshot[0]), // gpuUsed
                Long.toString(gpuSnapshot[1]), // gpuTotal
                Long.toString(gpuSnapshot[2]), // gpuUtil
                Long.toString(gpuSnapshot[3]), // gpuTemp
            };

            try {
                writer.write(String.join(",", values));
                writer.newLine();
                writer.flush();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

        }
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