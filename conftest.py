import os

# Force CPU backend for JAX to avoid CUDA init errors in test envs without GPUs.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

collect_ignore_glob = [
    "XP/*",
]
