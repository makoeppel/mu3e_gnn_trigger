from keras.losses import Loss
from keras.metrics import Metric
import tensorflow as tf
import keras


@keras.utils.register_keras_serializable(package="Custom", name="ChamferDistanceMasked")
class ChamferDistanceMasked(Loss):
    def __init__(
        self, padding_value=-1.0, name="chamfer_distance_masked", mode="cartesian"
    ):
        """
        padding_value: Value used to pad the input tensors.
        mode: Coordinate system for distance calculation.
            - "cartesian": Standard Cartesian coordinates.
            - "cylindrical": Cylindrical coordinates (r, theta, z).
            - "spherical": Spherical coordinates (r, theta, phi).
        """
        super().__init__(name=name)
        self.padding_value = padding_value
        mode = mode.lower()
        if mode == "cartesian":
            self.coordinate_transform = lambda x: x
        elif mode == "cylindrical":
            self.coordinate_transform = lambda x: tf.stack(
                [
                    x[..., 0] * tf.cos(x[..., 1]),
                    x[..., 0] * tf.sin(x[..., 1]),
                    x[..., 2],
                ],
                axis=-1,
            )
        elif mode == "spherical":
            self.coordinate_transform = lambda x: tf.stack(
                [
                    x[..., 0] * tf.sin(x[..., 1]) * tf.cos(x[..., 2]),
                    x[..., 0] * tf.sin(x[..., 1]) * tf.sin(x[..., 2]),
                    x[..., 0] * tf.cos(x[..., 1]),
                ],
                axis=-1,
            )
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Supported modes are 'cartesian', 'cylindrical', and 'spherical'."
            )

    def call(self, y_true, y_pred):
        """
        y_true: (B, N, D)
        y_pred: (B, M, D)
        """
        # Transform coordinates if necessary
        y_true = self.coordinate_transform(y_true)
        y_pred = self.coordinate_transform(y_pred)

        mask_true = tf.reduce_any(tf.not_equal(y_true, self.padding_value), axis=-1)
        mask_pred = tf.reduce_any(tf.not_equal(y_pred, self.padding_value), axis=-1)

        valid_mask = tf.logical_and(
            tf.expand_dims(mask_true, axis=1), tf.expand_dims(mask_pred, axis=0)
        )  # (B, N, M)

        # Compute pairwise distances
        diff = tf.expand_dims(y_true, axis=1) - tf.expand_dims(y_pred, axis=0)

        large_value = 1e9
        dist_sq = tf.where(
            valid_mask,
            tf.reduce_sum(tf.square(diff), axis=-1),
            tf.fill(tf.shape(diff[..., 0]), large_value),
        )  # (B, N, M)
        # Get min distances in both directions
        min_dist_true = tf.reduce_min(dist_sq, axis=1)  # (N,)
        min_dist_pred = tf.reduce_min(dist_sq, axis=0)  # (M,)
        # Apply mask before averaging
        loss_true = tf.reduce_sum(min_dist_true * tf.cast(mask_true, tf.float32)) / (
            tf.reduce_sum(tf.cast(mask_true, tf.float32)) + 1e-6
        )
        loss_pred = tf.reduce_sum(min_dist_pred * tf.cast(mask_pred, tf.float32)) / (
            tf.reduce_sum(tf.cast(mask_pred, tf.float32)) + 1e-6
        )
        chamfer_loss = loss_true + loss_pred
        return chamfer_loss


@keras.utils.register_keras_serializable(package="Custom", name="ChamferDistanceMasked")
class ChamferDistance(Loss):
    def __init__(self, name="chamfer_distance", mode="cartesian"):
        """
        padding_value: Value used to pad the input tensors.
        mode: Coordinate system for distance calculation.
            - "cartesian": Standard Cartesian coordinates.
            - "cylindrical": Cylindrical coordinates (r, theta, z).
            - "spherical": Spherical coordinates (r, theta, phi).
        """
        super().__init__(name=name)
        mode = mode.lower()
        if mode == "cartesian":
            self.coordinate_transform = lambda x: x
        elif mode == "cylindrical":
            self.coordinate_transform = lambda x: tf.stack(
                [
                    x[..., 0] * tf.cos(x[..., 1]),
                    x[..., 0] * tf.sin(x[..., 1]),
                    x[..., 2],
                ],
                axis=-1,
            )
        elif mode == "spherical":
            self.coordinate_transform = lambda x: tf.stack(
                [
                    x[..., 0] * tf.sin(x[..., 1]) * tf.cos(x[..., 2]),
                    x[..., 0] * tf.sin(x[..., 1]) * tf.sin(x[..., 2]),
                    x[..., 0] * tf.cos(x[..., 1]),
                ],
                axis=-1,
            )
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Supported modes are 'cartesian', 'cylindrical', and 'spherical'."
            )

    def call(self, y_true, y_pred):
        """
        y_true: (B, N, D)
        y_pred: (B, M, D)
        """
        # Transform coordinates if necessary
        y_true = self.coordinate_transform(y_true)
        y_pred = self.coordinate_transform(y_pred)

        # Compute pairwise distances
        diff = tf.expand_dims(y_true, axis=1) - tf.expand_dims(y_pred, axis=0)

        dist_sq = tf.reduce_sum(tf.square(diff), axis=-1)  # (B, N, M)

        # Get min distances in both directions
        min_dist_true = tf.reduce_min(dist_sq, axis=1)  # (N,)
        min_dist_pred = tf.reduce_min(dist_sq, axis=0)  # (M,)

        # Compute the loss
        loss_true = tf.reduce_mean(min_dist_true)  # Average over N
        loss_pred = tf.reduce_mean(min_dist_pred)  # Average over M

        chamfer_loss = loss_true + loss_pred
        return chamfer_loss


class ChamferDistanceOutlierPunish(Loss):
    def __init__(
        self,
        padding_value=-1.0,
        outlier_threshold=0.1,
        outlier_weight=1.0,
        mode="cartesian",
        name="chamfer_distance_masked",
    ):
        super(ChamferDistanceOutlierPunish, self).__init__(name=name)
        self.padding_val = padding_value
        self.outlier_threshold = outlier_threshold
        self.outlier_weight = outlier_weight
        mode = mode.lower()
        if mode == "cartesian":
            self.coordinate_transform = lambda x: x
        elif mode == "cylindrical":
            self.coordinate_transform = lambda x: tf.stack(
                [
                    x[..., 0] * tf.cos(x[..., 1]),
                    x[..., 0] * tf.sin(x[..., 1]),
                    x[..., 2],
                ],
                axis=-1,
            )
        elif mode == "spherical":
            self.coordinate_transform = lambda x: tf.stack(
                [
                    x[..., 0] * tf.sin(x[..., 1]) * tf.cos(x[..., 2]),
                    x[..., 0] * tf.sin(x[..., 1]) * tf.sin(x[..., 2]),
                    x[..., 0] * tf.cos(x[..., 1]),
                ],
                axis=-1,
            )
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Supported modes are 'cartesian', 'cylindrical', and 'spherical'."
            )

    def call(self, y_true, y_pred):
        """
        y_true: (B, N, D)
        y_pred: (B, M, D)
        Assumes that padded values are equal to padding_val.
        """
        # Compute pairwise distances
        y_true = self.coordinate_transform(y_true)
        y_pred = self.coordinate_transform(y_pred)

        diff = tf.expand_dims(y_true, axis=2) - tf.expand_dims(
            y_pred, axis=1
        )  # (B, N, M, D)
        dist_sq = tf.reduce_sum(tf.square(diff), axis=-1)  # (B, N, M)

        # Create masks: (B, N), (B, M)
        mask_true = tf.reduce_any(
            tf.not_equal(y_true, self.padding_val), axis=-1
        )  # (B, N)
        mask_pred = tf.reduce_any(
            tf.not_equal(y_pred, self.padding_val), axis=-1
        )  # (B, M)

        mask_true_f = tf.cast(mask_true, tf.float32)
        mask_pred_f = tf.cast(mask_pred, tf.float32)

        # Mask pairwise distances: expand to (B, N, M)
        mask_true_expand = tf.expand_dims(mask_true_f, axis=2)  # (B, N, 1)
        mask_pred_expand = tf.expand_dims(mask_pred_f, axis=1)  # (B, 1, M)
        valid_mask = mask_true_expand * mask_pred_expand  # (B, N, M)

        # masked_dist_sq = tf.where(valid_mask > 0, dist_sq, tf.fill(tf.shape(dist_sq), float('inf')))

        # Get min distances in both directions
        min_dist_true = tf.math.exp(tf.reduce_min(dist_sq, axis=2))  # (B, N)
        min_dist_pred = tf.math.exp(tf.reduce_min(dist_sq, axis=1))  # (B, M)

        # Apply mask before averaging
        loss_true = tf.reduce_sum(min_dist_true * mask_true_f) / (
            tf.reduce_sum(mask_true_f) + 1e-6
        )
        loss_pred = tf.reduce_sum(min_dist_pred * mask_pred_f) / (
            tf.reduce_sum(mask_pred_f) + 1e-6
        )

        chamfer_loss = loss_true + loss_pred

        # Outlier penalty: points with min dist > threshold
        unexplained_true = (
            tf.cast(min_dist_true > self.outlier_threshold**2, tf.float32) * mask_true_f
        )
        unexplained_pred = (
            tf.cast(min_dist_pred > self.outlier_threshold**2, tf.float32) * mask_pred_f
        )

        outlier_loss_true = tf.reduce_sum(unexplained_true) / (
            tf.reduce_sum(mask_true_f) + 1e-6
        )
        outlier_loss_pred = tf.reduce_sum(unexplained_pred) / (
            tf.reduce_sum(mask_pred_f) + 1e-6
        )

        outlier_penalty = outlier_loss_true + outlier_loss_pred

        y_true_std = tf.math.reduce_variance(y_true, axis=0)
        y_pred_std = tf.math.reduce_variance(y_pred, axis=0)

        return chamfer_loss


class MaskedMSE(Loss):
    def __init__(self, padding_value=-1.0, name="masked_mse"):
        super(MaskedMSE, self).__init__(name=name)
        self.padding_value = padding_value

    def call(self, y_true, y_pred):
        mask = tf.not_equal(y_true, self.padding_value)
        squared_diff = tf.square(y_true - y_pred)
        masked_squared_diff = tf.where(mask, squared_diff, tf.zeros_like(squared_diff))
        return tf.reduce_sum(masked_squared_diff) / tf.reduce_sum(
            tf.cast(mask, tf.float32) + 1e-8
        )


class MaskConcatMSE(Loss):
    def __init__(self, padding_value=-1.0, epsilon=1e-5, name="masked_concat_mse"):
        super().__init__(name=name)
        self.padding_value = padding_value
        self.epsilon = epsilon

    def call(self, y_true_unused, y_pred):
        # Split into model output and the target input (used internally by model)
        reconstructed, target = tf.split(y_pred, 2, axis=-1)
        target = tf.stop_gradient(target)

        # Masking for padded values
        mask = tf.abs(target - self.padding_value) > self.epsilon

        squared_error = tf.square(reconstructed - target)
        masked_error = tf.where(mask, squared_error, tf.zeros_like(squared_error))

        loss = tf.reduce_sum(masked_error) / (
            tf.reduce_sum(tf.cast(mask, tf.float32)) + self.epsilon
        )
        return loss
