import tensorflow as tf
import keras
from keras import ops
import numpy as np


class EarthMoversDistanceLoss(keras.losses.Loss):
    """
    Earth Mover's Distance (Wasserstein Distance) loss for point sets.

    This loss computes the EMD between two point sets using the Sinkhorn algorithm
    for approximate optimal transport, which is differentiable and GPU-friendly.

    Usage:
        model.compile(
            optimizer='adam',
            loss=EarthMoversDistanceLoss(reg=0.1, max_iter=100)
        )
    """

    def __init__(
        self,
        reg=0.1,
        max_iter=100,
        threshold=1e-3,
        name="earth_movers_distance",
        **kwargs
    ):
        """
        Args:
            reg: Regularization parameter for Sinkhorn algorithm (higher = more regularized)
            max_iter: Maximum number of Sinkhorn iterations
            threshold: Convergence threshold for Sinkhorn iterations (not used in graph mode)
            reduction: Type of reduction to apply to the loss
            name: Name of the loss function
        """
        super().__init__(name=name, **kwargs)
        self.reg = reg
        self.max_iter = max_iter
        self.threshold = threshold

    @tf.function
    def call(self, y_true, y_pred):
        """
        Compute EMD loss between predicted and true point sets.

        Args:
            y_true: Ground truth point set, shape (batch_size, n_points, dim)
            y_pred: Predicted point set, shape (batch_size, n_points, dim)

        Returns:
            EMD loss tensor of shape (batch_size,) before reduction
        """
        # Compute pairwise distance matrix
        cost_matrix = self._compute_cost_matrix(y_pred, y_true)

        # Solve optimal transport using Sinkhorn algorithm
        transport_plan = self._sinkhorn(cost_matrix)

        # Compute EMD as the Frobenius inner product
        emd = ops.sum(transport_plan * cost_matrix, axis=[1, 2])

        return emd

    def _compute_cost_matrix(self, x, y):
        """
        Compute pairwise Euclidean distance matrix between point sets.

        Args:
            x: tensor of shape (batch_size, n_points_x, dim)
            y: tensor of shape (batch_size, n_points_y, dim)

        Returns:
            cost_matrix: tensor of shape (batch_size, n_points_x, n_points_y)
        """
        # Expand dimensions for broadcasting
        x_expanded = ops.expand_dims(x, axis=2)  # (batch, n_x, 1, dim)
        y_expanded = ops.expand_dims(y, axis=1)  # (batch, 1, n_y, dim)

        # Compute squared Euclidean distances
        diff = x_expanded - y_expanded
        distances_squared = ops.sum(diff * diff, axis=-1)

        # Take square root to get Euclidean distances
        distances = ops.sqrt(
            distances_squared + 1e-8
        )  # Add small epsilon for numerical stability

        return distances

    def _sinkhorn(self, cost_matrix):
        """
        Sinkhorn algorithm for approximate optimal transport.

        Args:
            cost_matrix: tensor of shape (batch_size, n_points_x, n_points_y)

        Returns:
            transport_plan: tensor of shape (batch_size, n_points_x, n_points_y)
        """
        batch_size = ops.shape(cost_matrix)[0]
        n_points_x = ops.shape(cost_matrix)[1]
        n_points_y = ops.shape(cost_matrix)[2]

        # Initialize uniform distributions
        mu = ops.ones((batch_size, n_points_x)) / ops.cast(
            n_points_x, dtype=cost_matrix.dtype
        )
        nu = ops.ones((batch_size, n_points_y)) / ops.cast(
            n_points_y, dtype=cost_matrix.dtype
        )

        # Compute kernel matrix
        K = ops.exp(-cost_matrix / self.reg)

        # Initialize dual variables
        u = ops.ones_like(mu)
        v = ops.ones_like(nu)

        # Use tf.while_loop for graph compatibility
        def sinkhorn_step(i, u, v):
            # Update u
            u_new = mu / (ops.sum(K * ops.expand_dims(v, axis=1), axis=2) + 1e-8)

            # Update v
            v_new = nu / (ops.sum(K * ops.expand_dims(u_new, axis=2), axis=1) + 1e-8)

            return i + 1, u_new, v_new

        def condition(i, u, v):
            return i < self.max_iter

        # Run Sinkhorn iterations using while_loop
        _, u_final, v_final = tf.while_loop(
            condition, sinkhorn_step, [0, u, v], maximum_iterations=self.max_iter
        )

        # Compute transport plan
        transport_plan = (
            ops.expand_dims(u_final, axis=2) * K * ops.expand_dims(v_final, axis=1)
        )

        return transport_plan

    def get_config(self):
        config = super().get_config()
        config.update(
            {"reg": self.reg, "max_iter": self.max_iter, "threshold": self.threshold}
        )
        return config


class ChamferDistanceLoss(keras.losses.Loss):
    """
    Chamfer Distance loss for point sets.

    Often faster to compute than EMD and still effective for point set comparisons.
    Good alternative when EMD is too computationally expensive.
    """

    def __init__(self, name="chamfer_distance", **kwargs):
        super().__init__(name=name, **kwargs)

    @tf.function
    def call(self, y_true, y_pred):
        """
        Compute Chamfer distance between predicted and true point sets.

        Args:
            y_true: Ground truth point set, shape (batch_size, n_points, dim)
            y_pred: Predicted point set, shape (batch_size, n_points, dim)

        Returns:
            Chamfer distance tensor of shape (batch_size,) before reduction
        """
        # Compute pairwise distances
        pred_expanded = ops.expand_dims(y_pred, axis=2)  # (batch, n_pred, 1, dim)
        true_expanded = ops.expand_dims(y_true, axis=1)  # (batch, 1, n_true, dim)

        distances = ops.sum(
            (pred_expanded - true_expanded) ** 2, axis=-1
        )  # (batch, n_pred, n_true)

        # Find minimum distances
        min_dist_pred_to_true = ops.min(distances, axis=2)  # (batch, n_pred)
        min_dist_true_to_pred = ops.min(distances, axis=1)  # (batch, n_true)

        # Chamfer distance is sum of both directions
        chamfer_dist = ops.mean(min_dist_pred_to_true, axis=1) + ops.mean(
            min_dist_true_to_pred, axis=1
        )

        return chamfer_dist


class HybridPointSetLoss(keras.losses.Loss):
    """
    Hybrid loss combining EMD and Chamfer distance.

    Provides a good balance between accuracy (EMD) and computational efficiency (Chamfer).
    """

    def __init__(
        self,
        emd_weight=0.7,
        chamfer_weight=0.3,
        reg=0.1,
        max_iter=50,
        name="hybrid_pointset_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.emd_weight = emd_weight
        self.chamfer_weight = chamfer_weight
        self.emd_loss = EarthMoversDistanceLoss(
            reg=reg, max_iter=max_iter, reduction="none"
        )
        self.chamfer_loss = ChamferDistanceLoss(reduction="none")

    @tf.function
    def call(self, y_true, y_pred):
        emd = self.emd_loss(y_true, y_pred)
        chamfer = self.chamfer_loss(y_true, y_pred)

        # Normalize losses to similar scales (optional)
        emd_mean = ops.stop_gradient(ops.mean(emd))
        chamfer_mean = ops.stop_gradient(ops.mean(chamfer))

        emd_normalized = emd / (emd_mean + 1e-8)
        chamfer_normalized = chamfer / (chamfer_mean + 1e-8)

        return (
            self.emd_weight * emd_normalized + self.chamfer_weight * chamfer_normalized
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {"emd_weight": self.emd_weight, "chamfer_weight": self.chamfer_weight}
        )
        return config


# Utility functions for common point set operations
def normalize_point_sets(pointsets):
    """Normalize point sets to unit sphere centered at origin."""
    # Center at origin
    centroid = ops.mean(pointsets, axis=1, keepdims=True)
    centered = pointsets - centroid

    # Scale to unit sphere
    max_dist = ops.max(ops.sqrt(ops.sum(centered**2, axis=-1)), axis=1, keepdims=True)
    normalized = centered / (ops.expand_dims(max_dist, axis=-1) + 1e-8)

    return normalized


def random_point_dropout(pointsets, dropout_rate=0.1, training=True):
    """Randomly drop points during training for data augmentation."""
    if not training:
        return pointsets

    batch_size, n_points, dim = (
        ops.shape(pointsets)[0],
        ops.shape(pointsets)[1],
        ops.shape(pointsets)[2],
    )
    keep_prob = 1.0 - dropout_rate

    # Generate random mask
    random_vals = tf.random.uniform((batch_size, n_points))
    mask = random_vals < keep_prob

    # Apply mask and pad with zeros where needed
    masked_points = pointsets * ops.expand_dims(
        ops.cast(mask, pointsets.dtype), axis=-1
    )

    return masked_points
