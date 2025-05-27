package com.tenbitmelon.machinelearningplayer;

import org.bytedeco.pytorch.Conv3dImpl;
import org.bytedeco.pytorch.Module;

public class ParkourNet extends Module {

    public static final int VOXEL_AREA_SIZE = 11;


    Conv3dImpl conv1, conv2;
}
