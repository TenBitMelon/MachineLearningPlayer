# Machine Learning Player (Java + LibTorch)

A **Work-In-Progress**, native Java implementation of Proximal Policy Optimization (PPO) running directly inside a
Minecraft Paper server.

![Status](https://img.shields.io/badge/Status-Experimental%20%2F%20WIP-red)
![Java](https://img.shields.io/badge/Java-21-orange)
![PyTorch](https://img.shields.io/badge/LibTorch-C%2B%2B-firebrick)
![Platform](https://img.shields.io/badge/Platform-Windows%20(x86__64)-blue)


> **🚧 VERY WIP AND UNSTABLE 🚧**
>
> This project is in the very early stages of development. It is **experimental**.
> * The build setup is currently complex due to specific LibTorch/JavaCPP snapshot dependencies.
> * It is currently hardcoded for **Windows (x86_64) with CUDA**.

## Overview

This project attempts to port the logic from [CleanRL's PPO implementation](https://docs.cleanrl.dev/rl-algorithms/ppo/)
entirely into Java, running as a Spigot/Paper plugin.

Instead of bridging to Python (like MineRL), this project uses **JavaCPP** to access **LibTorch (PyTorch C++)** directly
from Java. This allows the Network training loop to run synchronously with the Minecraft
Server tick loop, sharing memory and data structures without network overhead.

### Key Features

* **In-Engine Training:** No external Python scripts. The server *is* the training environment.
* **Vectorized Environments:** Simulates multiple "Agents" ($N$ environments) on a single server thread, batched into
  single Tensor operations.
* **PPO + LSTM:** Implements Proximal Policy Optimization with Recurrent Neural Networks (LSTM) to handle partial
  observability.
* **Visual Debugger:** A custom, in-game 3D UI using `TextDisplay` entities to visualize tensors, probabilities, and
  logs in real-time.
* **Mixed Action Space:** Agents output discrete movement keys (WASD) and continuous rotation (Yaw/Pitch)
  simultaneously.

## Running & Building

To run or build this, you need a specific environment:

1. **OS:** Windows 10 (idk about 11) (Current build scripts are hardcoded for `windows-x86_64`).
2. **Java:** JDK 21.
3. **GPU:** NVIDIA GPU, CUDA support is required.
4. **Minecraft:** Paper 1.21.5.

## The Convoluted Build Setup

> **Read this carefully.** This project relies on snapshot builds of JavaCPP presets for PyTorch, CUDA, and OpenBLAS.
> These versions change frequently or may be removed from repositories.

The `build.gradle.kts` is currently configured to look for specific versions (e.g., `pytorch-2.7.1-1.5.12-SNAPSHOT`)
which do not exist anymore.

1. **JavaCPP Dependencies:** If the snapshot versions listed in `build.gradle.kts` are no longer available on Sonatype
   Snapshots, you need to:
    * Update the versions to the latest available snapshots.
    * Verify that all the required native libraries (e.g., CUDA, OpenBLAS) are compatible with each other.

2. **Shadow Jar:** The build uses the Shadow plugin to bundle dependencies.
   ```bash
   ./gradlew shadowJar
   ```

3. **Dependency Extraction:**
   Due to the massive size of PyTorch/CUDA libraries, the `build.gradle.kts` contains a task `copyDependencies` which
   extracts JARs to a specific local path. You will likely need to edit the `into` path in `build.gradle.kts` to match
   your local server setup:

   ```kotlin
   // Inside build.gradle.kts
   tasks.register<Copy>("copyDependencies") {
       // ...
       into("C:\\Your\\Path\\To\\Server\\plugins\\lib\\") 
   }
   ```

## Running the Project

1. **Server Configuration:**
   Ensure your server start script allocates enough RAM (LibTorch + Minecraft requires it).

   This is my current `run.bat` script:
   ```bat
   @echo off
   
   REM Adding native library paths for JavaCPP (adjust paths as necessary)
   set "PATH=C:\Users\Aidan\.javacpp\cache\cuda-12.9-9.10-1.5.12-20250612.145546-3-windows-x86_64.jar\org\bytedeco\cuda\windows-x86_64;C:\Users\Aidan\.javacpp\cache\MachineLearningPlayer-1.0.0-SNAPSHOT.jar\org\bytedeco\pytorch\windows-x86_64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\extras\CUPTI\lib64;%PATH%"
   
   REM Configure the JVM options for memory management and logging
   java -Dorg.bytedeco.javacpp.maxbytes=5G -Dorg.bytedeco.javacpp.maxphysicalbytes=10G -Xmx2G -DIReallyKnowWhatIAmDoingISwear -XX:ErrorFile=./hs_err_pid%p.log -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=./dumps -XX:+UnlockDiagnosticVMOptions -XX:+LogVMOutput -XX:LogFile=./jvmlog.%p.log -jar paper-1.21.5-101.jar nogui
   
   REM Log the exit time and error level for debugging purposes
   echo %date% %time%: Exited with %errorlevel% >> run_history.log
   
   pause
   ```

2. **Native Libraries:**
   JavaCPP usually handles extracting DLLs to `~/.javacpp/cache`. If you encounter `UnsatisfiedLinkError`, ensure your
   `PATH` includes CUDA binaries or that the cache is accessible.

3. **In-Game Commands:**

   If you get this far, congratulations! 🎉

   The plugin registers the `/ml` command tree:

    * `/ml runTraining`: Toggles the training loop on/off.
    * `/ml sprint`: Toggles whether agents are forced to sprint.
    * `/ml checkpoint`: Manually saves the model to disk.
    * `/ml loglevel <debug|info|warn>`: Changes console logging verbosity.
    * `/ml uiupdates`: Pauses/Resumes the in-game debug UI (saves performance).
    * `/ml args`: Displays current hyperparameters from `ExperimentConfig`.
    * `/ml productionRun`: Disables UI and starts high-speed training.

## Trainings

This project is iterative. The `main` branch always contains the latest version of the agent and environment.

For historical context, here are the key releases:

| Release                                                                                | Description                                                                |
|:---------------------------------------------------------------------------------------|:---------------------------------------------------------------------------|
| [**v0.1.0**](https://github.com/TenBitMelon/MachineLearningPlayer/releases/tag/v0.1.0) | First working version. The agent can walk to goal block and turn its head. |
| [**v0.2.0**](https://github.com/TenBitMelon/MachineLearningPlayer/releases/tag/v0.2.0) | Sumo battles. Only trained for a small bit                                 |

# Credits

* TenBitMelon

## Acknowledgements

* [CleanRL](https://docs.cleanrl.dev/) (PPO Implementation).
* [JavaCPP](https://github.com/bytedeco/javacpp)
