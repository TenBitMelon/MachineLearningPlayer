package com.tenbitmelon.machinelearningplayer.util.distrobutions;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

public class Categorical {

    private final Tensor logits;
    private final Tensor probs;
    private final Tensor logProbs;
    // private final int numEvents;
    // private final long[] batchShape;

    public Categorical(Tensor logits) {
        // this.logits = logits.sub(logits.logsumexp(new long[]{-1}, true));
        // this.probs = torch.exp(this.logits);
        // this.numEvents = (int) logits.size(-1);
        // this.batchShape = logits.shape();
        this.logits = logits;
        this.probs = torch.softmax(logits, -1);
        this.logProbs = torch.log_softmax(logits, -1);
    }

    // def sample(self, sample_shape=torch.Size()):
    //     if not isinstance(sample_shape, torch.Size):
    //         sample_shape = torch.Size(sample_shape)
    //     probs_2d = self.probs.reshape(-1, self._num_events)
    //     samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
    //     return samples_2d.reshape(self._extended_shape(sample_shape))


    // def _extended_shape(self, sample_shape: _size = torch.Size()) -> torch.Size:
    //     """
    //     Returns the size of the sample returned by the distribution, given
    //     a `sample_shape`. Note, that the batch and event shapes of a distribution
    //     instance are fixed at the time of construction. If this is empty, the
    //     returned shape is upcast to (1,).
    //
    //     Args:
    //         sample_shape (torch.Size): the size of the sample to be drawn.
    //     """
    //     if not isinstance(sample_shape, torch.Size):
    //         sample_shape = torch.Size(sample_shape)
    //     return torch.Size(sample_shape + self._batch_shape + self._event_shape)

    public Tensor sample() {
        // Tensor probs = logits.exp();
        // Tensor probs2d = probs.reshape(-1, numEvents);
        // Tensor samples2d = torch.multinomial(probs2d, 1, true, null).t();
        // return samples2d.reshape(batchShape);
        return torch.multinomial(probs, 1, true, null);
    }

    // def log_prob(self, value):
    //     if self._validate_args:
    //         self._validate_sample(value)
    //     value = value.long().unsqueeze(-1)
    //     value, log_pmf = torch.broadcast_tensors(value, self.logits)
    //     value = value[..., :1]
    //     return log_pmf.gather(-1, value).squeeze(-1)

    public Tensor logProb(Tensor value) {
        // value = value.unsqueeze(-1);
        // TensorVector vector = new TensorVector(value, logits);
        // TensorVector broadcasted = torch.broadcast_tensors(vector);
        // value = broadcasted.get(0).slice(-1, new LongOptional(0), new LongOptional(1), 1);
        // Tensor logPmf = broadcasted.get(1);
        // return logPmf.gather(-1, value).squeeze(-1);
        Tensor val = value.to(torch.ScalarType.Long);
        return logProbs.gather(1, val).squeeze(-1);
    }


    // def entropy(self):
    //     min_real = torch.finfo(self.logits.dtype).min
    //         logits = torch.clamp(self.logits, min=min_real)
    //     p_log_p = logits * self.probs
    //     return -p_log_p.sum(-1)

    public Tensor entropy() {
        // f32 min value: -3.4028235E38
        // double minReal = Float.MAX_VALUE;
        // Tensor clampedLogits = torch.clamp(logits, new ScalarOptional(new Scalar(minReal)));
        Tensor pLogP = logits.mul(probs);
        return pLogP.sum(-1).neg();
    }
}

