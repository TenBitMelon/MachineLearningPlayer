package com.tenbitmelon.machinelearningplayer.debugger;

import com.tenbitmelon.machinelearningplayer.ExperimentConfig;
import com.tenbitmelon.machinelearningplayer.models.TrainingManager;
import org.bytedeco.javacpp.tools.NativeAllocationTracer;
import org.bytedeco.pytorch.Stat;
import org.bytedeco.pytorch.cuda.BlockInfo;
import org.bytedeco.pytorch.cuda.DeviceStats;
import org.bytedeco.pytorch.cuda.SegmentInfo;
import org.bytedeco.pytorch.cuda.SnapshotInfo;
import org.bytedeco.pytorch.global.torch_cuda;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public final class LeakProbe {

    private static final String[] COLUMNS = {
        "iter", "phase", "ms_since_iter_start", "allocated", "reserved", "active",
        "inactive_split", "requested", "alloc_after_emptycache", "reserved_after_emptycache",
        "tracer_registered_bytes", "tracer_registered_count", "javacpp_physical", "java_heap_used"
    };

    private static BufferedWriter writer;
    private static long iterStartMs = -1L;

    private LeakProbe() {}

    public static void iterStart(int iteration) {
        iterStartMs = System.currentTimeMillis();
        probe(iteration, "iter_start", false);
    }

    public static void stepStart(int iteration) {
        probe(iteration, "step", false);
    }

    public static void collectionEnd(int iteration) {
        probe(iteration, "collection_end", false);
    }

    public static void afterGae(int iteration) {
        probe(iteration, "after_gae", false);
    }

    public static void afterEpochs(int iteration) {
        probe(iteration, "after_epochs", false);
    }

    public static void iterEnd(int iteration) {
        probe(iteration, "iter_end", true);
    }

    public static void snapshotDump(int iteration) {
        if (TrainingManager.args == null)
            return;
        if (!TrainingManager.args.featureFlags.contains(ExperimentConfig.FeatureFlag.ALLOCATOR_SNAPSHOT))
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

    public static void close() {
        if (writer != null) {
            try {
                writer.close();
            } catch (IOException ignored) {
            }
            writer = null;
        }
    }

    public static void nativeAllocationSnapshot(int iteration) {
        if (TrainingManager.args == null)
            return;
        if (!TrainingManager.args.featureFlags.contains(ExperimentConfig.FeatureFlag.LEAK_PROBE))
            return;

        // Write all sites to file for debugging memory leaks
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

    private static void probe(int iteration, String phase, boolean emptyCacheTest) {
        if (TrainingManager.args == null)
            return;
        if (!TrainingManager.args.featureFlags.contains(ExperimentConfig.FeatureFlag.LEAK_PROBE))
            return;

        long allocated = -1L;
        long reserved = -1L;
        long active = -1L;
        long inactiveSplit = -1L;
        long requested = -1L;
        DeviceStats stats = null;
        try {
            stats = torch_cuda.getAllocator().getDeviceStats(TrainingManager.device.index());
            Stat stat = stats.allocated_bytes();
            allocated = stat.current();
            stat.close();
            stat = stats.reserved_bytes();
            reserved = stat.current();
            stat.close();
            stat = stats.active_bytes();
            active = stat.current();
            stat.close();
            stat = stats.inactive_split_bytes();
            inactiveSplit = stat.current();
            stat.close();
            stat = stats.requested_bytes();
            requested = stat.current();
            stat.close();
        } finally {
            if (stats != null) {
                stats.close();
            }
        }

        long allocatedAfter = allocated;
        long reservedAfter = reserved;
        if (emptyCacheTest) {
            torch_cuda.device_synchronize();
            torch_cuda.getAllocator().emptyCache();
            DeviceStats stats2 = null;
            try {
                stats2 = torch_cuda.getAllocator().getDeviceStats(TrainingManager.device.index());
                Stat stat = stats2.allocated_bytes();
                allocatedAfter = stat.current();
                stat.close();
                stat = stats2.reserved_bytes();
                reservedAfter = stat.current();
                stat.close();
            } finally {
                if (stats2 != null) {
                    stats2.close();
                }
            }
        }

        JavaCppDiagnostics.Snapshot diagnostics = JavaCppDiagnostics.snapshot();
        long heapUsed = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        long ms = iterStartMs >= 0 ? System.currentTimeMillis() - iterStartMs : -1L;

        writeRow(
            iteration,
            phase,
            ms,
            allocated,
            reserved,
            active,
            inactiveSplit,
            requested,
            allocatedAfter,
            reservedAfter,
            diagnostics.registeredBytes(),
            diagnostics.registeredCount(),
            diagnostics.physicalBytes(),
            heapUsed
        );
    }

    private static synchronized void writeRow(long iteration, String phase, long... values) {
        if (writer == null) {
            writer = openWriter();
            if (writer == null) {
                return;
            }
        }
        StringBuilder sb = new StringBuilder(128);
        sb.append(iteration).append(',').append(phase);
        for (long value : values) {
            sb.append(',').append(value);
        }
        try {
            writer.write(sb.toString());
            writer.newLine();
            writer.flush();
        } catch (IOException e) {
            System.err.println("[leakprobe] write failed: " + e.getMessage());
        }
    }

    private static synchronized BufferedWriter openWriter() {
        String experimentId = TrainingManager.args != null ? TrainingManager.args.experimentId : "unknown";
        Path dir = Paths.get("training", experimentId);
        try {
            Files.createDirectories(dir);
            BufferedWriter out = new BufferedWriter(new FileWriter(dir.resolve("leak_probe.csv").toFile(), false));
            out.write(String.join(",", COLUMNS));
            out.newLine();
            return out;
        } catch (IOException e) {
            System.err.println("[leakprobe] failed to open " + dir + ": " + e.getMessage());
            return null;
        }
    }

}