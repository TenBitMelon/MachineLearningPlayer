package com.tenbitmelon.machinelearningplayer.util;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Convention;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.pytorch.GatheredContext;

@Convention("C++")
public class GatherFunction extends FunctionPointer {
    static {
        Loader.load();
    }

    public GatherFunction() {allocate();}

    private native void allocate();

    public @Name("gather") GatheredContext call() {
        return new GatheredContext();
    }
}