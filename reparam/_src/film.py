import haiku as hk
import jax
import jax.numpy as jnp
from haiku.initializers import Constant


class FiLM(hk.Module):
    """
    Feature-wise linear modulation for general conditioning [1].

    This implementation deviates from the original: It uses layer-wise
    normalization [2] rather than batch-wise [3].

    [1]: https://arxiv.org/abs/1709.07871
    [2]: https://arxiv.org/abs/1607.06450
    [3]: https://arxiv.org/abs/1502.03167
    """

    def __init__(self, axis=-1, gate=None, eps=1e-4, name=None):
        super().__init__(name=name)
        self.eps = eps
        self.axis = axis
        self.gate = gate

    def __call__(self, x, z):
        axis = self.axis if self.axis >= 0 else x.ndim + self.axis
        width = x.shape[axis]
        shift = hk.Linear(width, name="shift_generator")(z)
        scale = hk.Linear(width, name="scale_generator")(z)
        if callable(self.gate):
            shift = self.gate(shift)
            scale = self.gate(scale)
        shape = [1] * x.ndim
        shape[axis] = width
        shift = shift.reshape(shape)
        scale = scale.reshape(shape)
        axes = tuple(range(x.ndim))
        axes = axes[:axis] + axes[axis + 1 :]
        scale *= jax.lax.rsqrt(jnp.var(x, axis=axes, keepdims=True) + self.eps)
        return (x - shift) * scale
