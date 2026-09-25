package com.tenbitmelon.machinelearningplayer.util.distributions;

import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

public class Normal implements AutoCloseable {
    //     Args:
    //         loc (float or Tensor): mean of the distribution (often referred to as mu)
    //         scale (float or Tensor): standard deviation of the distribution
    //             (often referred to as sigma)
    private final Tensor loc;
    private final Tensor scale;
    private final long[] batch_shape;

    /*
    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, _Number) and isinstance(scale, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)
     */
    public Normal(Tensor loc, Tensor scale) {
        this.loc = loc;
        this.scale = scale;
        this.batch_shape = loc.shape();
    }

    /*
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape) // torch.Size(sample_shape + self._batch_shape + self._event_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))
     */
    public Tensor sample() {
        try (NoGradGuard noGrad = new NoGradGuard()) {
            Tensor expandedLoc = this.loc.expand(batch_shape);
            Tensor expandedScale = this.scale.expand(batch_shape);
            Tensor sample = torch.normal(expandedLoc, expandedScale);
            expandedLoc.close();
            expandedScale.close();
            return sample;
        }
    }

    /*
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = self.scale**2
        log_scale = (
            math.log(self.scale)
            if isinstance(self.scale, _Number)
            else self.scale.log()
        )
        return (
            -((value - self.loc) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )
     */
    public Tensor logProb(Tensor value) {
        Tensor var = scale.square();
        Tensor logScale = scale.log();
        Tensor diff = value.sub(loc);
        Tensor squared = diff.square();
        Tensor doubledVar = var.mul(new Scalar(2));
        Tensor div = squared.div(doubledVar);
        double log = Math.log(Math.sqrt(2.0 * Math.PI));
        Tensor negDiv = div.neg();
        Tensor shifted = negDiv.sub(logScale);
        Tensor result = shifted.sub(new Scalar(log));

        var.close();
        logScale.close();
        diff.close();
        squared.close();
        doubledVar.close();
        div.close();
        negDiv.close();
        shifted.close();
        return result;
    }

    /*
    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)
     */

    public Tensor entropy() {
        Tensor logScale = torch.log(scale);
        Tensor shifted = logScale.add(new Scalar(0.5));
        Tensor entropy = shifted.add(new Scalar(0.5 * Math.log(2 * Math.PI)));
        logScale.close();
        shifted.close();
        return entropy;
    }

    @Override
    public void close() {
        loc.close();
        scale.close();
    }
}
