import tensorflow as tf
import keras


class MMDLoss(keras.losses.Loss):
    """Maximum Mean Discrepancy (MMD) loss for training generative models.
    This loss computes the distance between the latent space distribution and a
    standard Gaussian distribution using the RBF kernel.
    Args:
        latent_dim (int): Dimension of the latent space.
        kernel (str): Kernel type, currently only 'rbf' is supported.
        sigma (float): Bandwidth for the RBF kernel.
        weight (float): Weight for the loss.
    """

    def __init__(self, latent_dim, kernel="rbf", sigma=1.0, weight=1.0, **kwargs):
        """
        Args:
            latent_dim (int): Dimension of the latent space.
            kernel (str): Kernel type, currently only 'rbf' is supported.
            sigma (float): Bandwidth for the RBF kernel.
            weight (float): Weight for the loss.
        """
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.kernel = kernel
        self.sigma = sigma
        self.weight = weight

    def call(self, _, y_pred):

        z = y_pred  # shape: (batch_size, latent_dim)
        batch_size = tf.shape(z)[0]
        prior = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )  # standard Gaussian

        return self.weight * self._mmd(z, prior)

    def _mmd(self, x, y):
        xx = self._compute_kernel(x, x)
        yy = self._compute_kernel(y, y)
        xy = self._compute_kernel(x, y)
        return tf.reduce_mean(xx + yy - 2 * xy)

    def _compute_kernel(self, x, y):
        x = tf.expand_dims(x, 1)  # shape: (batch, 1, dim)
        y = tf.expand_dims(y, 0)  # shape: (1, batch, dim)
        dist = tf.reduce_sum((x - y) ** 2, axis=2)
        return tf.exp(-dist / (2 * self.sigma**2))


class SWDLoss(tf.keras.losses.Loss):
    """Sliced Wasserstein Distance (SWD) loss for training generative models.
    This loss computes the distance between the latent space distribution and a
    standard Gaussian distribution using random projections.
    Args:
        reg_weight (float): Regularization weight for the SWD loss.
        num_projections (int): Number of random projections to use.
        name (str): Name of the loss function.
    """

    def __init__(self, reg_weight=1.0, num_projections=50, name="swd_loss"):
        """Args:
        reg_weight (float): Regularization weight for the SWD loss.
        num_projections (int): Number of random projections to use.
        name (str): Name of the loss function.
        """
        super().__init__(name=name)
        self.reg_weight = reg_weight
        self.num_projections = num_projections

    def call(self, y_true, y_pred):
        z = y_pred

        prior_z = tf.random.normal(tf.shape(z))
        latent_dim = tf.shape(z)[1]

        # Random directions
        projections = tf.random.normal([self.num_projections, latent_dim])
        projections = tf.math.l2_normalize(projections, axis=-1)  # [P, D]

        proj_z = tf.linalg.matmul(z, projections, transpose_b=True)  # [B, P]
        proj_prior = tf.linalg.matmul(prior_z, projections, transpose_b=True)  # [B, P]

        proj_z_sorted = tf.sort(proj_z, axis=0)
        proj_prior_sorted = tf.sort(proj_prior, axis=0)

        swd = tf.reduce_mean(tf.square(proj_z_sorted - proj_prior_sorted))

        return self.reg_weight * swd


class UnitHyperSphereCoverLoss(keras.losses.Loss):
    """Loss function to encourage uniform coverage of the unit hypersphere.
    This loss computes the mean of pairwise squared distances between normalized
    latent vectors, encouraging them to be uniformly distributed on the hypersphere.
    Args:
        latent_dim (int): Dimension of the latent space.
        temperature (float): Temperature parameter for scaling the distances.
    """

    def __init__(self, temperature=10, **kwargs):
        """Args:
        latent_dim (int): Dimension of the latent space.
        temperature (float): Temperature parameter for scaling the distances.
        """
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, y_true, y_pred):
        z = y_pred  # shape: (batch_size, latent_dim)
        batch_size = tf.shape(z)[0]
        z = tf.math.l2_normalize(z, axis=1)

        # Compute pairwise squared distances
        # Using ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T y
        similarity_matrix = tf.matmul(z, z, transpose_b=True)
        sq_dists = 2.0 - 2.0 * similarity_matrix  # since ||z|| = 1

        # Remove diagonal (self-pairs)
        mask = tf.ones_like(sq_dists) - tf.eye(batch_size)
        sq_dists_no_diag = (
            sq_dists * mask
        )  # Compute the loss as the mean of the distances
        exp_dists = tf.exp(-self.temperature * sq_dists_no_diag)
        mean_exp = tf.reduce_sum(exp_dists) / tf.cast(
            batch_size * (batch_size - 1), tf.float32
        )
        return tf.math.log(mean_exp + 1e-6) / self.temperature


class CovarianceLoss(keras.losses.Loss):
    """Loss function to encourage low covariance among latent vectors.
    This loss computes the covariance matrix of the latent vectors and penalizes
    the off-diagonal elements, encouraging them to be close to zero.
    Args:
        **kwargs: Additional keyword arguments for the base class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, _, y_pred):
        z = y_pred  # shape: (batch_size, latent_dim)
        batch_size = tf.cast(tf.shape(z)[0], tf.float32)
        z_size = tf.cast(tf.shape(z)[1], tf.float32)

        # Compute covariance matrix
        z_centered = z - tf.reduce_mean(z, axis=0)
        cov_matrix = tf.matmul(z_centered, z_centered, transpose_a=True) / (
            batch_size - 1.0
        )

        off_diag_mask = 1.0 - tf.eye(z_size)
        loss = tf.reduce_sum(tf.square(cov_matrix * off_diag_mask)) / z_size
        return loss


class VarianceLoss(keras.losses.Loss):
    """Loss function to encourage low variance among latent vectors.
    This loss computes the variance of the latent vectors and penalizes it,
    encouraging the variance to be close to 1.
    Args:
        target_std (float): Target standard deviation for the latent vectors.
        **kwargs: Additional keyword arguments for the base class.
    """

    def __init__(self, target_std, **kwargs):
        super().__init__(**kwargs)
        self.target_std = target_std

    def call(self, _, y_pred):
        z = y_pred  # shape: (batch_size, latent_dim)
        return tf.reduce_mean(
            tf.nn.relu(
                self.target_std - tf.sqrt(tf.math.reduce_variance(z, axis=0) + 1e-8)
            )
        )


class ContrastiveLoss(keras.losses.Loss):
    """Contrastive loss for training models with multiple views.
    This loss computes pairwise contrastive loss between multiple views of the output.
    Assumes that the output is split into multiple views along the last dimension.
    Args:
        contrastive_views (int): Number of views to consider for contrastive loss.
        var_mtpl (float): Multiplier for the variance loss.
        cov_mtpl (float): Multiplier for the covariance loss.
        **kwargs: Additional keyword arguments for the base class.
    """

    def __init__(self, contrastive_views=4, var_mtpl=25, cov_mtpl=1, **kwargs):
        super().__init__(**kwargs)
        self.contrastive_views = contrastive_views
        self.var_mtpl = var_mtpl
        self.cov_mtpl = cov_mtpl
        self.cov_loss = CovarianceLoss()
        self.var_loss = VarianceLoss()
        self.swd_loss = SWDLoss()

    def call(self, y_true, y_pred):
        outputs = tf.split(y_pred, self.contrastive_views, axis=-1)
        loss = 0.0
        for i in range(self.contrastive_views):
            # Compute pairwise contrastive loss
            for j in range(i + 1, self.contrastive_views):
                loss += tf.reduce_mean(tf.square(outputs[i] - outputs[j])) / (
                    self.contrastive_views * (self.contrastive_views - 1) / 2
                )
            loss += (
                self.var_mtpl
                * self.var_loss(outputs[i], outputs[i])
                / self.contrastive_views
            )
            loss += (
                self.cov_mtpl
                * self.cov_loss(outputs[i], outputs[i])
                / self.contrastive_views
            )
            loss += self.swd_loss(outputs[i], outputs[i]) / self.contrastive_views
        return loss


class VarianceCovarianceLoss(keras.losses.Loss):
    """Loss function to encourage low variance and covariance among latent vectors.
    This loss computes the variance and covariance of the latent vectors and penalizes
    them, encouraging the variance to be close to 1 and covariance to be close to 0.
    Args:
        target_std (float): Target standard deviation for the latent vectors.
        **kwargs: Additional keyword arguments for the base class.
    """

    def __init__(self, target_std=1.0, cov_penalty=1, **kwargs):
        super().__init__(**kwargs)
        self.target_std = target_std
        self.var_loss = VarianceLoss(target_std)
        self.cov_loss = CovarianceLoss()
        self.cov_penalty = cov_penalty

    def call(self, _, y_pred):
        return self.var_loss(_, y_pred) + self.cov_penalty * self.cov_loss(_, y_pred)


class EmbeddingSpaceSpreading(keras.losses.Loss):
    """Loss function to encourage low variance and covariance among latent vectors.
    This loss computes the variance and covariance of the latent vectors and penalizes
    them, encouraging the variance to be close to 1 and covariance to be close to 0.
    Args:
        target_std (float): Target standard deviation for the latent vectors.
        **kwargs: Additional keyword arguments for the base class.
    """

    def __init__(
        self, target_std=1.0, var_penalty=25, cov_penalty=1, swd_penalty=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.target_std = target_std
        self.var_loss = VarianceLoss(target_std)
        self.cov_loss = CovarianceLoss()
        self.swd_loss = UnitHyperSphereCoverLoss()
        self.cov_penalty = cov_penalty
        self.var_penalty = var_penalty
        self.swd_penalty = swd_penalty if swd_penalty is not None else var_penalty

    def call(self, _, y_pred):
        return (
            self.var_penalty * self.var_loss(_, y_pred)
            + self.cov_penalty * self.cov_loss(_, y_pred)
            + self.swd_loss(_, y_pred)
        )


class TripletLoss(keras.losses.Loss):
    def __init__(self, margin=1.0, normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.normalize = normalize

    def call(self, _, y_pred):
        anchor, positive, negative = tf.split(y_pred, 3, axis=-1)

        if self.normalize:
            anchor = tf.math.l2_normalize(anchor, axis=-1)
            positive = tf.math.l2_normalize(positive, axis=-1)
            negative = tf.math.l2_normalize(negative, axis=-1)

        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

        loss = tf.nn.relu(pos_dist - neg_dist + self.margin)
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            'margin': self.margin,
            'normalize': self.normalize
        })
        return config


class TripletLossWithRegularization(TripletLoss):
    def __init__(self, regularization_loss : keras.losses.Loss, **kwargs):
        super().__init__(**kwargs)
        self.regularization_loss = regularization_loss

    def call(self, _, y_pred):
        loss = super().call(_, y_pred)
        anchor, positive, negative = tf.split(y_pred, 3, axis=-1)
        anchor_reg = self.regularization_loss(_,anchor)
        positive_reg = self.regularization_loss(_,positive)
        negative_reg = self.regularization_loss(_,negative)
        reg_loss = (anchor_reg + positive_reg + negative_reg) / 3.0
        return loss + reg_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'regularization_loss': keras.losses.serialize(self.regularization_loss)
        })
        return config