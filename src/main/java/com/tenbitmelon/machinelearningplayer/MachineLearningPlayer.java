package com.tenbitmelon.machinelearningplayer;

import com.tenbitmelon.machinelearningplayer.debugger.Debugger;
import com.tenbitmelon.machinelearningplayer.debugger.Logger;
import com.tenbitmelon.machinelearningplayer.events.EntityInteractEvent;
import com.tenbitmelon.machinelearningplayer.models.TrainingManager;
import io.papermc.paper.command.brigadier.Commands;
import io.papermc.paper.plugin.lifecycle.event.types.LifecycleEvents;
import net.kyori.adventure.text.Component;
import org.bukkit.*;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.player.PlayerMoveEvent;
import org.bukkit.plugin.PluginManager;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.scheduler.BukkitRunnable;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.presets.torch_cuda;
import org.slf4j.event.Level;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Method;
import java.net.URL;

@SuppressWarnings("UnstableApiUsage")
public final class MachineLearningPlayer extends JavaPlugin implements Listener {

    public static Logger LOGGER = null;

    static {
        Loader.load(org.bytedeco.javacpp.presets.javacpp.class);
        Loader.load(org.bytedeco.openblas.presets.openblas.class);
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
        // Loader.load(org.bytedeco.pytorch.presets.torch_cuda.class);
    }

    @Override
    public void onEnable() {
        System.setProperty("org.bytedeco.openblas.load", "mkl");
        // loadExternalDependencies();


        // System.setProperty("org.bytedeco.javacpp.logger.debug", "true");
        // System.setProperty("org.bukkit.plugin.java.LibraryLoader.centralURL", "https://oss.sonatype.org/content/groups/public/");
        // System.load("C:/Users/Aidan/.javacpp/cache/MachineLearningPlayer-1.0.0-SNAPSHOT.jar/org/bytedeco/pytorch/windows-x86_64/jnitorch.dll");
        // Loader.load(org.bytedeco.javacpp.presets.javacpp.class);
        // Loader.load(org.bytedeco.openblas.presets.openblas.class);
        // Loader.load(org.bytedeco.pytorch.presets.torch.class);
        // Loader.load(org.bytedeco.pytorch.presets.torch_cuda.class);
        // Loader.load(torch.class);


        // try {
        //     Loader.load(org.bytedeco.pytorch.presets.torch.class);
        // } catch (UnsatisfiedLinkError e) {
        //     String path = null;
        //     try {
        //         path = Loader.cacheResource(org.bytedeco.pytorch.presets.torch.class, "windows-x86_64/jnitorch.dll").getPath();
        //         new ProcessBuilder("C:/Users/Aidan/Downloads/Dependencies_x64_Release/DependenciesGui.exe", path).start().waitFor();
        //     } catch (InterruptedException | IOException ex) {
        //         throw new RuntimeException(ex);
        //     }
        // }
        // Device device = args.cuda ? torch.device(torch.kCUDA) : torch.device(torch.kCPU);
        System.out.println("torch.cuda_is_available() = " + org.bytedeco.pytorch.global.torch.cuda_is_available());
        System.out.println("torch.cuda_device_count() = " + org.bytedeco.pytorch.global.torch.cuda_device_count());
        System.out.println("torch.hasCUDA() = " + torch.hasCUDA());
        //
        Device device = new Device("cuda:0");
        System.out.println("device = " + device);
        System.out.println("is_cuda = " + device.is_cuda());
        System.out.println("is_cpu = " + device.is_cpu());
        System.out.println("type = " + device.type().toString());


        LOGGER = new Logger();
        LOGGER.setEnabled(Level.DEBUG, false);

        PluginManager pluginManager = getServer().getPluginManager();
        pluginManager.registerEvents(this, this);
        pluginManager.registerEvents(new EntityInteractEvent(), this);
        pluginManager.registerEvents(Debugger.DebugListener.INSTANCE, this);

        this.getLifecycleManager().registerEventHandler(LifecycleEvents.COMMANDS, event -> {
            final Commands commands = event.registrar();
            commands.register(MachineLearningCommand.register(), "Run the machine learning command");
        });

        World world = Bukkit.getWorlds().getFirst();

        world.setTime(0);
        world.setGameRule(GameRule.DO_DAYLIGHT_CYCLE, false);
        world.setWeatherDuration(0);
        world.setStorm(false);
        world.setGameRule(GameRule.DO_WEATHER_CYCLE, false);
        world.setGameRule(GameRule.DO_MOB_SPAWNING, false);
        world.setGameRule(GameRule.RANDOM_TICK_SPEED, 0);
        world.setGameRule(GameRule.DO_ENTITY_DROPS, false);
        world.setGameRule(GameRule.DO_TILE_DROPS, false);
        world.setGameRule(GameRule.DO_INSOMNIA, false);
        world.setGameRule(GameRule.DO_IMMEDIATE_RESPAWN, true);


        // DedicatedServer server = ((CraftServer) Bukkit.getServer()).getServer();
        // ServerTickRateManager serverTickRateManager = server.tickRateManager();
        // if (serverTickRateManager.isSprinting()) {
        //     serverTickRateManager.stopSprinting();
        // }
        // if (serverTickRateManager.isSteppingForward()) {
        //     serverTickRateManager.stopStepping();
        // }
        // serverTickRateManager.setFrozen(true);

        for (Entity entity : world.getEntities()) {
            if (entity instanceof Player player) {
                if (player.getGameMode() == GameMode.CREATIVE) {
                    player.teleport(new Location(world, 16, 5, 20, 180, 0));
                } else {
                    player.kick();
                }
            } else {
                entity.remove();
            }
        }

        TrainingManager.setup();

        LOGGER.info("Hello, Machine Learning Player is starting!");

        Debugger.start();
        new BukkitRunnable() {
            @Override
            public void run() {
                TrainingManager.trainingStep();
                Debugger.update();
            }
        }.runTaskTimer(this, 0, 1);


        // Net net = new Net();
        //
        // // Create a multi-threaded data loader for the MNIST dataset.
        // MNISTMapDataset data_set = new MNIST("./data").map(new ExampleStack());
        // MNISTRandomDataLoader data_loader = new MNISTRandomDataLoader(
        //     data_set, new RandomSampler(data_set.size().get()),
        //     new DataLoaderOptions(/*batch_size=*/64));
        //
        // // Instantiate an SGD optimization algorithm to update our Net's parameters.
        // SGD optimizer = new SGD(net.parameters(), new SGDOptions(/*lr=*/0.01));
        //
        // for (int epoch = 1; epoch <= 10; ++epoch) {
        //     int batch_index = 0;
        //     // Iterate the data loader to yield batches from the dataset.
        //     for (ExampleIterator it = data_loader.begin(); !it.equals(data_loader.end()); it = it.increment()) {
        //         Example batch = it.access();
        //         // Reset gradients.
        //         optimizer.zero_grad();
        //         // Execute the model on the input data.
        //         Tensor prediction = net.forward(batch.data());
        //         // Compute a loss value to judge the prediction of our model.
        //         Tensor loss = nll_loss(prediction, batch.target());
        //         // Compute gradients of the loss w.r.t. the parameters of our model.
        //         loss.backward();
        //         // Update the parameters based on the calculated gradients.
        //         optimizer.step();
        //         // Output the loss and checkpoint every 100 batches.
        //         if (++batch_index % 100 == 0) {
        //             System.out.println("Epoch: " + epoch + " | Batch: " + batch_index
        //                 + " | Loss: " + loss.item_float());
        //             // Serialize your model periodically as a checkpoint.
        //             OutputArchive archive = new OutputArchive();
        //             net.save(archive);
        //             archive.save_to("net.pt");
        //         }
        //     }
        // }
    }

    @Override
    public void onDisable() {
        Debugger.stop();
        TrainingManager.shutdown();
    }

    @EventHandler
    public void onPlayerJoin(PlayerJoinEvent event) {
        event.getPlayer().sendMessage(Component.text("Hello, " + event.getPlayer().getName() + "!"));
    }

    // @EventHandler
    // public void onMove(PlayerMoveEvent event) {
    //     System.out.println(event.getPlayer().getVelocity());
    // }

    // static class Net extends Module {
    //     // Use one of many "standard library" modules.
    //     final LinearImpl fc1, fc2, fc3;
    //
    //     Net() {
    //         // Construct and register two Linear submodules.
    //         fc1 = register_module("fc1", new LinearImpl(784, 64));
    //         fc2 = register_module("fc2", new LinearImpl(64, 32));
    //         fc3 = register_module("fc3", new LinearImpl(32, 10));
    //     }
    //
    //     // Implement the Net's algorithm.
    //     Tensor forward(Tensor x) {
    //         // Use one of many tensor manipulation functions.
    //         x = relu(fc1.forward(x.reshape(x.size(0), 784)));
    //         x = dropout(x, /*p=*/0.5, /*train=*/is_training());
    //         x = relu(fc2.forward(x));
    //         x = log_softmax(fc3.forward(x), /*dim=*/1);
    //         return x;
    //     }
    // }

}
