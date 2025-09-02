import tensorflow as tf
import keras
from .losses import VarianceCovarianceLoss


class MultiObjectiveTrainer:
    """A trainer for multi-objective optimization of a fixed-size embedding encoder and autoencoder.

    This trainer was developed to train an encoder and autoencoder simultaneously, fully unsupervised.

    Args:
        encoder (keras.Model): The encoder model to be trained.
        autoencoder (keras.Model): The autoencoder model to be trained.
        autoencoder_loss (keras.losses.Loss, optional): Loss function for the autoencoder.
        variance_loss (VarianceCovarianceLoss, optional): Loss function for variance and covariance.
        lambda_var (float): Weight for the variance loss in the total loss.
        batch_size (int): Batch size for training.
    Attributes:
        encoder (keras.Model): The encoder model.
        autoencoder (keras.Model): The autoencoder model.
        autoencoder_loss (keras.losses.Loss): Loss function for the autoencoder.
        variance_loss (VarianceCovarianceLoss): Loss function for variance and covariance.
        lambda_var (float): Weight for the variance loss.
    """

    def __init__(
        self,
        encoder,
        autoencoder,
        autoencoder_loss=None,
        variance_loss=None,
        lambda_var=0.5,
        batch_size=512,
        **kwargs
    ):
        self.encoder = encoder
        self.autoencoder = autoencoder

        self.autoencoder_loss = (
            autoencoder_loss
            if autoencoder_loss is not None
            else keras.losses.MeanSquaredError()
        )
        self.variance_loss = (
            variance_loss
            if variance_loss is not None
            else VarianceCovarianceLoss(cov_penalty=0.1)
        )
        self.lambda_var = lambda_var

        self.batch_size = batch_size

    @tf.function
    def train_encoder_step(
        self, dataset: tf.data.Dataset, encoder_optimizer, validation_dataset=None
    ):
        """Train the encoder using the reconstruction loss and variance loss.
        Args:
            dataset (tf.data.Dataset): Dataset containing input samples.
            encoder_optimizer (keras.optimizers.Optimizer): Optimizer for the encoder.
            validation_dataset (tf.data.Dataset, optional): Dataset for validation.
        Returns:
            Tuple of mean reconstruction loss and mean variance loss.
        """
        self.autoencoder.trainable = False
        self.encoder.trainable = True

        total_recon_loss = 0.0
        total_var_loss = 0.0
        num_batches = 0

        for input_sample in dataset:
            with tf.GradientTape() as tape:
                z = self.encoder(input_sample, training=True)
                z_hat = self.autoencoder(z, training=False)
                recon_loss = self.autoencoder_loss(z, z_hat)
                var_loss = self.variance_loss(z, z)
                loss = recon_loss + self.lambda_var * var_loss

            grads = tape.gradient(loss, self.encoder.trainable_variables)
            encoder_optimizer.apply_gradients(
                zip(grads, self.encoder.trainable_variables)
            )

            total_recon_loss += recon_loss
            total_var_loss += var_loss
            num_batches += 1

        mean_recon_loss = total_recon_loss / tf.cast(num_batches, tf.float32)
        mean_var_loss = total_var_loss / tf.cast(num_batches, tf.float32)
        return mean_recon_loss, mean_var_loss

    @tf.function
    def train_autoencoder_step(
        self,
        dataset: tf.data.Dataset,
        ae_optimizer: keras.optimizers.Optimizer,
        num_steps=5,
    ):
        """Train the autoencoder using the latent representations from the encoder.
        Args:
            dataset (tf.data.Dataset): Dataset containing input samples.
            ae_optimizer (keras.optimizers.Optimizer): Optimizer for the autoencoder.
            num_steps (int): Number of training steps to perform.
        Returns:
            List of losses for each training step.
        """
        self.encoder.trainable = False
        self.autoencoder.trainable = True
        latent_dataset = dataset.map(self.encoder, num_parallel_calls=tf.data.AUTOTUNE)
        losses = []
        for _ in range(num_steps):
            total_recon_loss = 0.0
            num_batches = 0

            for z in latent_dataset:
                with tf.GradientTape() as tape:
                    z_hat = self.autoencoder(z, training=True)
                    recon_loss = self.autoencoder_loss(z, z_hat)

                grads = tape.gradient(recon_loss, self.autoencoder.trainable_variables)
                ae_optimizer.apply_gradients(
                    zip(grads, self.autoencoder.trainable_variables)
                )

                total_recon_loss += recon_loss
                num_batches += 1
            losses.append(total_recon_loss / tf.cast(num_batches, tf.float32))
        return losses

    @tf.function
    def train_encoder_variance_step(
        self, dataset: tf.data.Dataset, encoder_optimizer: keras.optimizers.Optimizer
    ):
        """Train the encoder to maximize the variance of its output.
        Args:
            dataset (tf.data.Dataset): Dataset containing input samples.
            encoder_optimizer (keras.optimizers.Optimizer): Optimizer for the encoder.
        Returns:
            Mean variance loss across the dataset.
        """
        self.encoder.trainable = True
        self.autoencoder.trainable = False
        num_batches = 0
        total_var_loss = 0.0

        for input_sample in dataset:
            with tf.GradientTape() as tape:
                z = self.encoder(input_sample, training=True)
                var_loss = self.variance_loss(z, z)
                loss = self.lambda_var * var_loss

            grads = tape.gradient(loss, self.encoder.trainable_variables)
            encoder_optimizer.apply_gradients(
                zip(grads, self.encoder.trainable_variables)
            )

            total_var_loss += var_loss
            num_batches += 1

        mean_var_loss = total_var_loss / tf.cast(num_batches, tf.float32)
        return mean_var_loss

    @tf.function
    def train_reconstruction_step(
        self, dataset: tf.data.Dataset, encoder_optimizer
    ):
        """Train the encoder using reconstruction loss only.
        Args:
            dataset (tf.data.Dataset): Dataset containing input samples.
            encoder_optimizer (keras.optimizers.Optimizer): Optimizer for the encoder.
            validation_dataset (tf.data.Dataset, optional): Dataset for validation.
        Returns:
            Mean reconstruction loss across the dataset.
        """
        self.autoencoder.trainable = False
        self.encoder.trainable = True

        total_recon_loss = 0.0
        num_batches = 0

        for input_sample in dataset:
            with tf.GradientTape() as tape:
                z = self.encoder(input_sample, training=True)
                z_hat = self.autoencoder(z, training=False)
                recon_loss = self.autoencoder_loss(z, z_hat)
                loss = recon_loss

            grads = tape.gradient(loss, self.encoder.trainable_variables)
            encoder_optimizer.apply_gradients(
                zip(grads, self.encoder.trainable_variables)
            )

            total_recon_loss += recon_loss
            num_batches += 1

        mean_recon_loss = total_recon_loss / tf.cast(num_batches, tf.float32)
        return mean_recon_loss
