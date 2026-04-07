package com.tenbitmelon.machinelearningplayer.debugger;

import org.bytedeco.javacpp.Pointer;

import java.lang.reflect.Field;
import java.util.Locale;

public final class JavaCppDiagnostics {

    private static final Class<?> DEALLOCATOR_REFERENCE_CLASS = resolveClass("org.bytedeco.javacpp.Pointer$DeallocatorReference");
    private static final Field DEALLOCATOR_TOTAL_BYTES_FIELD = resolveField(DEALLOCATOR_REFERENCE_CLASS, "totalBytes");
    private static final Field DEALLOCATOR_TOTAL_COUNT_FIELD = resolveField(DEALLOCATOR_REFERENCE_CLASS, "totalCount");
    private static final Field MAX_BYTES_FIELD = resolveField(Pointer.class, "maxBytes");
    private static final Field MAX_PHYSICAL_BYTES_FIELD = resolveField(Pointer.class, "maxPhysicalBytes");
    private static final Field MAX_RETRIES_FIELD = resolveField(Pointer.class, "maxRetries");
    private static final Field DEALLOCATOR_THREAD_FIELD = resolveField(Pointer.class, "deallocatorThread");

    private JavaCppDiagnostics() {}

    public static Snapshot snapshot() {
        long registeredBytes = readLong(DEALLOCATOR_TOTAL_BYTES_FIELD);
        long registeredCount = readLong(DEALLOCATOR_TOTAL_COUNT_FIELD);
        long physicalBytes = safePointerCall(Pointer::physicalBytes);
        long totalPhysicalBytes = safePointerCall(Pointer::totalPhysicalBytes);
        long availablePhysicalBytes = safePointerCall(Pointer::availablePhysicalBytes);
        Thread deallocatorThread = readThread(DEALLOCATOR_THREAD_FIELD);

        return new Snapshot(
            registeredBytes,
            registeredCount,
            physicalBytes,
            totalPhysicalBytes,
            availablePhysicalBytes,
            deallocatorThread != null && deallocatorThread.isAlive()
        );
    }

    public static String describeConfiguration() {
        boolean pointerGcEnabled = !isPropertyTrue(
            System.getProperty("org.bytedeco.javacpp.nopointergc"),
            System.getProperty("org.bytedeco.javacpp.noPointerGC")
        );
        long maxBytes = readLong(MAX_BYTES_FIELD);
        long maxPhysicalBytes = readLong(MAX_PHYSICAL_BYTES_FIELD);
        long maxRetries = readLong(MAX_RETRIES_FIELD);

        return String.format(
            Locale.ROOT,
            "pointerGc=%s, deallocatorThread=%s, maxBytes=%s, maxPhysicalBytes=%s, maxRetries=%d, pathsFirst=%s, nativeAllocationTracer=%s",
            pointerGcEnabled,
            snapshot().deallocatorThreadAlive() ? "alive" : "disabled",
            formatMaybeBytes(maxBytes),
            formatMaybeBytes(maxPhysicalBytes),
            maxRetries,
            propertyOrDefault("org.bytedeco.javacpp.pathsFirst", "false"),
            propertyOrDefault("org.bytedeco.javacpp.nativeAllocationTracer", "unsupported in 1.5.12")
        );
    }

    public static String describeSnapshot() {
        Snapshot snapshot = snapshot();
        return String.format(
            Locale.ROOT,
            "registered=%s, refs=%d, physical=%s, availablePhysical=%s/%s, deallocatorThread=%s",
            formatMaybeBytes(snapshot.registeredBytes()),
            snapshot.registeredCount(),
            formatMaybeBytes(snapshot.physicalBytes()),
            formatMaybeBytes(snapshot.availablePhysicalBytes()),
            formatMaybeBytes(snapshot.totalPhysicalBytes()),
            snapshot.deallocatorThreadAlive() ? "alive" : "disabled"
        );
    }

    private static String propertyOrDefault(String key, String fallback) {
        String value = System.getProperty(key);
        return value != null ? value : fallback;
    }

    private static boolean isPropertyTrue(String... values) {
        for (String value : values) {
            if (value == null) {
                continue;
            }
            String normalized = value.trim().toLowerCase(Locale.ROOT);
            if (normalized.equals("true") || normalized.equals("t") || normalized.isEmpty()) {
                return true;
            }
        }
        return false;
    }

    private static long safePointerCall(LongSupplier supplier) {
        try {
            return supplier.getAsLong();
        } catch (Throwable ignored) {
            return -1L;
        }
    }

    private static long readLong(Field field) {
        if (field == null) {
            return -1L;
        }
        try {
            Object value = field.get(null);
            if (value instanceof Number number) {
                return number.longValue();
            }
        } catch (IllegalAccessException ignored) {
        }
        return -1L;
    }

    private static Thread readThread(Field field) {
        if (field == null) {
            return null;
        }
        try {
            Object value = field.get(null);
            if (value instanceof Thread thread) {
                return thread;
            }
        } catch (IllegalAccessException ignored) {
        }
        return null;
    }

    private static Class<?> resolveClass(String className) {
        try {
            return Class.forName(className);
        } catch (ClassNotFoundException ignored) {
            return null;
        }
    }

    private static Field resolveField(Class<?> owner, String fieldName) {
        if (owner == null) {
            return null;
        }
        try {
            Field field = owner.getDeclaredField(fieldName);
            field.setAccessible(true);
            return field;
        } catch (ReflectiveOperationException ignored) {
            return null;
        }
    }

    private static String formatMaybeBytes(long bytes) {
        if (bytes < 0) {
            return "n/a";
        }
        return SystemStats.formatBytes(bytes);
    }

    @FunctionalInterface
    private interface LongSupplier {
        long getAsLong() throws Throwable;
    }

    public record Snapshot(
        long registeredBytes,
        long registeredCount,
        long physicalBytes,
        long totalPhysicalBytes,
        long availablePhysicalBytes,
        boolean deallocatorThreadAlive
    ) {}
}
