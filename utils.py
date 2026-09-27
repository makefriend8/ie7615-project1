"""Reproducibility settings shared by all classification experiments."""
import tensorflow as tf


def set_seed(seed=42):
    """Seed Python, NumPy and TensorFlow and enable deterministic operations.

    Repeated runs use the same random sequences and deterministic TensorFlow
    computations with unchanged inputs, code, hardware and software versions.
    Each model calls this function before data loading and model construction.
    """
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
