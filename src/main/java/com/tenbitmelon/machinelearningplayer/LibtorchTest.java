package com.tenbitmelon.machinelearningplayer;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;

import static org.bytedeco.pytorch.global.torch.*;

public class LibtorchTest {

    // Define a new Module.
    static class Net extends Module {
        Net() {
            // Construct and register two Linear submodules.
            fc1 = register_module("fc1", new LinearImpl(784, 64));
            fc2 = register_module("fc2", new LinearImpl(64, 32));
            fc3 = register_module("fc3", new LinearImpl(32, 10));
        }

        // Implement the Net's algorithm.
        Tensor forward(Tensor x) {
            // Use one of many tensor manipulation functions.
            x = relu(fc1.forward(x.reshape(x.size(0), 784)));
            x = dropout(x, /*p=*/0.5, /*train=*/is_training());
            x = relu(fc2.forward(x));
            x = log_softmax(fc3.forward(x), /*dim=*/1);
            return x;
        }

        // Use one of many "standard library" modules.
        final LinearImpl fc1, fc2, fc3;
    }
}
