import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def standardize(v):
    fanin = np.prod(v.shape[:-1])
    dimen = tuple(range(v.ndim - 1))
    shift = jnp.mean(v, axis=dimen, keepdims=True)
    scale = jnp.var(v, axis=dimen, keepdims=True) * fanin
    shift *= scale
    return (v - shift) * jax.lax.rsqrt(scale + 1e-4)


def weight_standardization(next_getter, value, context):
    """
    Weight-standardization, roughly adapted from Eq. 1 of https://arxiv.org/abs/2102.06171
    """
    if context.name.endswith("_g"):
        return value  # short-circuit recursion
    elif "norm" not in context.full_name.lower():
        scale = hk.get_parameter(
            f"{context.name}_g",
            value.shape[-1:] if value.ndim > 1 else (),
            init=jnp.ones,
        )
        value = standardize(value) * scale
    return next_getter(value)
