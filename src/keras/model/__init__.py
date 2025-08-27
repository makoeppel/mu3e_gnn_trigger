"""
This module provides different types of models for various tasks.
The models include:
- AutoEncoder: A model for unsupervised learning that learns to compress and reconstruct data.
- Variational AutoEncoder: A probabilistic model that learns to encode data into a latent space
  and sample from it.

The submodule `components` contains the necessary components for building these and other models.
"""

from .auto_encoder import *
from .var_auto_encoder import *
