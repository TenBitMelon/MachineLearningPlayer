package com.tenbitmelon.machinelearningplayer;

import io.papermc.paper.command.brigadier.Commands;
import io.papermc.paper.plugin.lifecycle.event.types.LifecycleEvents;
import net.kyori.adventure.text.Component;
import org.bukkit.Bukkit;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.plugin.java.JavaPlugin;

import static org.bytedeco.pytorch.global.torch.nll_loss;

@SuppressWarnings("UnstableApiUsage")
public final class MachineLearningPlayer extends JavaPlugin implements Listener {

    @Override
    public void onEnable() {
        Bukkit.getPluginManager().registerEvents(this, this);

        this.getLifecycleManager().registerEventHandler(LifecycleEvents.COMMANDS, event -> {
            final Commands commands = event.registrar();
            commands.register(MachineLearningCommand.register(), "Run the machine learning command");
        });

        // /* try to use MKL when available */
        // System.setProperty("org.bytedeco.openblas.load", "mkl");
        //
        // // Create a new Net.
        // LibtorchTest.Net net = new LibtorchTest.Net();
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
        //   int batch_index = 0;
        //   // Iterate the data loader to yield batches from the dataset.
        //   for (ExampleIterator it = data_loader.begin(); !it.equals(data_loader.end()); it = it.increment()) {
        //     Example batch = it.access();
        //     // Reset gradients.
        //     optimizer.zero_grad();
        //     // Execute the model on the input data.
        //     Tensor prediction = net.forward(batch.data());
        //     // Compute a loss value to judge the prediction of our model.
        //     Tensor loss = nll_loss(prediction, batch.target());
        //     // Compute gradients of the loss w.r.t. the parameters of our model.
        //     loss.backward();
        //     // Update the parameters based on the calculated gradients.
        //     optimizer.step();
        //     // Output the loss and checkpoint every 100 batches.
        //     if (++batch_index % 100 == 0) {
        //       String message = ("Epoch: " + epoch + " | Batch: " + batch_index
        //           + " | Loss: " + loss.item_float());
        //       Bukkit.getOnlinePlayers().forEach(player -> player.sendMessage(Component.text(message)));
        //       // Serialize your model periodically as a checkpoint.
        //       OutputArchive archive = new OutputArchive();
        //       net.save(archive);
        //       archive.save_to("net.pt");
        //     }
        //   }
        // }
    }

    @Override
    public void onDisable() {
        // Plugin shutdown logic
    }

    @EventHandler
    public void onPlayerJoin(PlayerJoinEvent event) {
        event.getPlayer().sendMessage(Component.text("Hello, " + event.getPlayer().getName() + "!"));
    }
}
