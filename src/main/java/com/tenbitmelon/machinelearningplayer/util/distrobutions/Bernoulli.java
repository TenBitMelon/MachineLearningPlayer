package com.tenbitmelon.machinelearningplayer.util.distrobutions;

import org.bytedeco.pytorch.BCEWithLogitsLossOptions;
import org.bytedeco.pytorch.LossReduction;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.kNone;

public class Bernoulli {

    private final Tensor logits;
    private final Tensor probs;
    private final long[] batchShape;

    public Bernoulli(Tensor logits) {
        this.logits = logits;
        this.probs = torch.sigmoid(logits);
        this.batchShape = logits.shape();
    }

    // def sample(self, sample_shape=torch.Size()):
    //     shape = self._extended_shape(sample_shape)
    //     with torch.no_grad():
    //         return torch.bernoulli(self.probs.expand(shape))

    public Tensor sample() {
        Tensor expandedProbs = probs.expand(batchShape);
        return torch.bernoulli(expandedProbs);
    }


    // def log_prob(self, value):
    //     if self._validate_args:
    //         self._validate_sample(value)
    //     logits, value = broadcast_all(self.logits, value)
    //     return -binary_cross_entropy_with_logits(logits, value, reduction="none")
    public Tensor logProb(Tensor value) {
        BCEWithLogitsLossOptions bceWithLogitsLossOptions = new BCEWithLogitsLossOptions();
        bceWithLogitsLossOptions.reduction().put(new kNone());
        return torch.binary_cross_entropy_with_logits(probs, value, bceWithLogitsLossOptions);
    }

    // def entropy(self):
    //     return binary_cross_entropy_with_logits(
    //         self.logits, self.probs, reduction="none"
    //     )
    public Tensor entropy() {
        BCEWithLogitsLossOptions bceWithLogitsLossOptions = new BCEWithLogitsLossOptions();
        bceWithLogitsLossOptions.reduction().put(new kNone());
        return torch.binary_cross_entropy_with_logits(logits, probs, bceWithLogitsLossOptions);
    }
}
