"""Computes a shrinkage-regularized empirical action covariance matrix (and its Cholesky
factor) from the B1K task-0 training split, for the `noise_cholesky_path` used by the
pi05_b1k_task0_{lora,full}_corr configs (see `_make_task0_noise_configs` in
`openpi/training/config.py`).

Method (correlated flow-matching noise):
  1. Flatten each normalized action chunk a in R^{H x D} (H = action_horizon, D =
     real_action_dim -- i.e. the real robot DOF, excluding zero-padding) to vec(a) in R^{H*D},
     row-major (index = h * D + d).
  2. Sigma_hat = (1/N) sum_n vec(a^(n)) vec(a^(n))^T over every (frame, action-chunk) sample in
     the train split.
  3. Shrink toward the identity for numerical stability / invertibility: Sigma_reg = beta *
     Sigma_hat + (1 - beta) * I.
  4. Cholesky factor: Sigma_reg = L L^T. Sampling eps = L @ z with z ~ N(0, I) then reproduces
     Sigma_reg-correlated noise (see `openpi.training.noise.sample_correlated_noise`).

Must run AFTER `scripts/compute_norm_stats.py --config-name pi05_b1k_task0_lora_gauss` (any of
the four task0 config names give identical data/norm stats, since they share `AssetsConfig`).
"""

import pathlib

import numpy as np
import tqdm
import tyro

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def main(
    config_name: str = "pi05_b1k_task0_lora_gauss",
    output: str | None = None,
    beta: float = 0.5,
    real_action_dim: int = 23,
    batch_size: int = 64,
    num_workers: int = 8,
    max_samples: int | None = 20_000,
):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise RuntimeError(
            f"No norm stats found for '{config_name}'. Run "
            f"`uv run scripts/compute_norm_stats.py --config-name {config_name}` first -- the "
            "covariance must be computed on the same normalized action space the model trains on."
        )

    action_horizon = config.model.action_horizon
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, config.model)
    dataset = _data_loader.transform_dataset(dataset, data_config)
    n_samples = len(dataset)
    print(f"{n_samples} (frame, action-chunk) samples across the task-0 train split "
          f"({len(data_config.episodes_index)} episodes)")

    hd = action_horizon * real_action_dim
    cov_sum = np.zeros((hd, hd), dtype=np.float64)
    n_seen = 0

    # Video decode (not the covariance math) is the bottleneck -- capping at max_samples (a
    # random subset, like compute_norm_stats.py's --max-frames) keeps this tractable. With
    # shrinkage (beta < 1) toward the identity, the estimate doesn't need every frame to be
    # numerically stable.
    use_subset = max_samples is not None and max_samples < n_samples
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        shuffle=use_subset,
        num_batches=max(1, (max_samples if use_subset else n_samples) // batch_size),
        num_workers=num_workers,
    )
    for batch in tqdm.tqdm(loader, desc="Accumulating covariance"):
        actions = np.asarray(batch["actions"])[..., :real_action_dim]  # (b, action_horizon, real_action_dim)
        if actions.shape[1] != action_horizon:
            raise RuntimeError(
                f"Expected action_horizon={action_horizon} in batch['actions'], got {actions.shape[1]}"
            )
        flat = actions.reshape(actions.shape[0], -1)  # (b, hd), row-major: index = h * D + d
        cov_sum += flat.T.astype(np.float64) @ flat.astype(np.float64)
        n_seen += flat.shape[0]

    if n_seen == 0:
        raise RuntimeError("No samples were seen -- check episodes_index / local_root.")

    sigma_hat = cov_sum / n_seen
    sigma_reg = beta * sigma_hat + (1 - beta) * np.eye(hd)
    chol = np.linalg.cholesky(sigma_reg).astype(np.float32)

    output_path = pathlib.Path(
        output
        or "./outputs/assets/pi05_b1k_task0/action_cholesky.npy"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, chol)
    print(
        f"Saved Cholesky factor {chol.shape} to {output_path} "
        f"(from {n_seen} samples, action_horizon={action_horizon}, real_action_dim={real_action_dim}, beta={beta})"
    )


if __name__ == "__main__":
    tyro.cli(main)
