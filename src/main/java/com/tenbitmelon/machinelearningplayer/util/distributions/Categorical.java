package com.tenbitmelon.machinelearningplayer.util.distributions;

import org.bytedeco.pytorch.LongArrayRef;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

public class Categorical implements AutoCloseable {

    private final Tensor logits;
    private final Tensor probs;
    private final long numEvents;
    private final LongArrayRef batchSize;

    public Categorical(Tensor logits) { // TODO: Make into or a version that works in batches so all of the actions can be sampled at once
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
        Tensor probs2d = this.probs.reshape(-1, this.numEvents); // (batch..., numEvents) -> (prod(batch), numEvents)
        Tensor samples2d = torch.multinomial(probs2d, 1, true, null).t(); // (batch, 1) -> (1, batch)
        Tensor reshaped = samples2d.reshape(batchSize); // (1, batch) -> (batch,)
        probs2d.close();
        samples2d.close();
        return reshaped;
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
        Tensor longValue = value.to(torch.ScalarType.Long);
        Tensor unsqueezed = longValue.unsqueeze(-1); // (batch,) -> (batch, 1)
        TensorVector vector = new TensorVector(unsqueezed, this.logits);
        TensorVector broadcasted = torch.broadcast_tensors(vector);
        Tensor broadFirst = broadcasted.get(0);
        Tensor log_pmf = broadcasted.get(1);
        Tensor sliced = broadFirst.slice(-1, new LongOptional(0), new LongOptional(1), 1);
        Tensor gathered = log_pmf.gather(-1, sliced);
        Tensor squeezed = gathered.squeeze(-1); // (batch, 1) -> (batch,)

        longValue.close();
        unsqueezed.close();
        vector.close();
        broadcasted.close();
        broadFirst.close();
        log_pmf.close();
        sliced.close();
        gathered.close();

        return squeezed;
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
        Tensor summed = pLogP.sum(-1);
        Tensor entropy = summed.neg();
        pLogP.close();
        summed.close();
        return entropy;
    }

    public void close() {
        this.logits.close();
        this.probs.close();
        this.batchSize.close();
    }
}

