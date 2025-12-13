package com.tenbitmelon.machinelearningplayer.util.distributions;

import org.bytedeco.pytorch.LongArrayRef;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

public class Categorical {

    private final Tensor logits;
    private final Tensor probs;
    private final long numEvents;
    private final LongArrayRef batchSize;

    public Categorical(Tensor logits) {
        this.logits = logits.sub(logits.logsumexp(new long[]{-1}, true));

        this.probs = torch.softmax(this.logits, -1);
        this.numEvents = this.logits.size(-1);

        // self._param.size()[:-1] if self._param.ndimension() > 1 else torch.Size()
        if (this.logits.ndimension() > 1) {
            LongArrayRef sizes = this.logits.sizes();
            this.batchSize = sizes.slice(0, sizes.size() - 1);
        } else {
            this.batchSize = new LongArrayRef(0);
        }
    }

    /*
    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
        return samples_2d.reshape(self._extended_shape(sample_shape))
     */

    public Tensor sample() {
        Tensor probs2d = this.probs.reshape(-1, this.numEvents);
        Tensor samples2d = torch.multinomial(probs2d, 1, true, null).t();
        return samples2d.reshape(batchSize);
    }

    /*
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)
     */

    public Tensor logProb(Tensor value) {
        value = value.to(torch.ScalarType.Long).unsqueeze(-1);
        TensorVector vector = new TensorVector(value, this.logits);
        TensorVector broadcasted = torch.broadcast_tensors(vector);
        value = broadcasted.get(0);
        Tensor log_pmf = broadcasted.get(1);
        value = value.slice(-1, new LongOptional(0), new LongOptional(1), 1);
        return log_pmf.gather(-1, value).squeeze(-1);
    }


    /*
    def entropy(self):
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)
     */

    public Tensor entropy() {
        Tensor pLogP = this.logits.mul(this.probs);
        return pLogP.sum(-1).neg();
    }
}

