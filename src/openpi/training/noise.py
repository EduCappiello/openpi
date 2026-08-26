"""Correlated flow-matching noise for training pi0/pi0.5 on action data with known
temporal/spatial structure, as an alternative to the default iid Gaussian noise sampled in
`Pi0.compute_loss`.

Given a shrinkage-regularized empirical covariance Sigma_reg over `vec(a) in R^{H*D}` (H = action
horizon, D = the number of *real* action channels -- i.e. excluding the zero-padding that
`PadStatesAndActions` adds up to the model's `action_dim`), its Cholesky factor L (Sigma_reg = L
L^T) lets us sample noise with that same covariance: eps = L @ z, z ~ N(0, I).

Padding channels (D..action_dim) always stay iid N(0, I): their target action is always exactly
zero, so there is no expert correlation structure to model there, and mixing them into the
shrinkage-regularized block would only dilute their variance from 1.0 to (1 - beta) as an artifact
of regularization rather than a meaningful noise-structure choice.
"""

import numpy as np

import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at


def load_cholesky(path: str) -> jax.Array:
    """Loads a precomputed Cholesky factor (see scripts/compute_action_covariance.py)."""
    chol = np.load(path)
    if chol.ndim != 2 or chol.shape[0] != chol.shape[1]:
        raise ValueError(f"Expected a square Cholesky factor, got shape {chol.shape} from {path}")
    return jnp.asarray(chol, dtype=jnp.float32)


@at.typecheck
def sample_correlated_noise(
    rng: at.KeyArrayLike,
    shape: tuple[int, int, int],
    chol: at.Float[at.Array, "hd hd"],
    real_action_dim: int,
) -> at.Float[at.Array, "b ah ad"]:
    """Samples flow-matching noise of the given (batch, action_horizon, action_dim) shape.

    The first `real_action_dim` channels of every timestep are drawn jointly from
    N(0, chol @ chol.T) (flattened over time and channel, row-major: index = h * real_action_dim +
    d, matching how scripts/compute_action_covariance.py builds `chol`). Any remaining padding
    channels are iid N(0, I).
    """
    batch, horizon, action_dim = shape
    if action_dim < real_action_dim:
        raise ValueError(f"action_dim ({action_dim}) < real_action_dim ({real_action_dim})")
    expected = horizon * real_action_dim
    if chol.shape != (expected, expected):
        raise ValueError(
            f"chol has shape {chol.shape}, expected ({expected}, {expected}) for "
            f"action_horizon={horizon} x real_action_dim={real_action_dim}"
        )

    real_rng, pad_rng = jax.random.split(rng)
    z = jax.random.normal(real_rng, (batch, expected))
    eps_real = (z @ chol.T).reshape(batch, horizon, real_action_dim)

    if action_dim == real_action_dim:
        return eps_real
    eps_pad = jax.random.normal(pad_rng, (batch, horizon, action_dim - real_action_dim))
    return jnp.concatenate([eps_real, eps_pad], axis=-1)
