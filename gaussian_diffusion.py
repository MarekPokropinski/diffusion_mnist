import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tqdm import tqdm
import matplotlib.pyplot as plt

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = tf.linspace(0, timesteps, steps)
    alphas_cumprod = tf.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return tf.clip_by_value(betas, 0, 0.999)

class GaussianDiffusion:
    def __init__(self, model) -> None:
        self.model: tf.keras.Model = model
        self.diffusion_steps = 1000
        # betas = tf.cast(tf.linspace(1e-4, 0.02, self.diffusion_steps), tf.float64)
        betas = tf.cast(cosine_beta_schedule(self.diffusion_steps), tf.float64)
        alphas = 1.0-betas
        alphas_cumprod = tf.math.cumprod(alphas)
        alphas_cumprod_prev = tf.math.cumprod(alphas, axis=0, exclusive=True)

        # posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_variance = betas

        self.betas = tf.cast(betas, tf.float32)
        self.alphas = tf.cast(alphas, tf.float32)
        self.alphas_cumprod = tf.cast(alphas_cumprod, tf.float32)
        self.alphas_cumprod_prev = tf.cast(alphas_cumprod_prev, tf.float32)
        self.alphas_cumprod_sqrt = tf.sqrt(alphas_cumprod)
        self.alphas_1mcumprod_sqrt = tf.sqrt(1.0-alphas_cumprod)
        self.posterior_variance = tf.cast(posterior_variance, tf.float32)

        self.sqrt_recip_alphas_cumprod = tf.cast(tf.sqrt(1./alphas_cumprod), tf.float32)
        self.sqrt_recipm1_alphas_cumprod = tf.cast(tf.sqrt(1./alphas_cumprod-1), tf.float32)

        self.sqrt_alpha_inv = tf.cast(tf.sqrt(1./alphas), tf.float32)
        self.sampling_coeff = tf.cast(betas/tf.sqrt((1.0-alphas_cumprod)*alphas), tf.float32)

        self.posterior_mean_coef1 = tf.cast(betas * tf.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod), tf.float32)
        self.posterior_mean_coef2 = tf.cast((1. - alphas_cumprod_prev) * tf.sqrt(alphas) / (1. - alphas_cumprod), tf.float32)

        # self.loss_fn = MeanSquaredError()
        self.loss_fn = MeanAbsoluteError()
        lr = 1e-4
        # lr = 2e-5
        self.optim = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def sample_q(self, x_start, t, noise):
        coeff1 = self.gather(self.alphas_cumprod_sqrt, t)
        coeff2 = self.gather(self.alphas_1mcumprod_sqrt, t)
        return tf.cast(coeff1*tf.cast(x_start, tf.float64) + coeff2*tf.cast(noise, tf.float64), tf.float32)

    @tf.function
    def loss_step(self, x_start):
        noise = tf.random.normal(x_start.shape)
        t = tf.random.uniform(x_start.shape[0:1], minval=0, maxval=self.diffusion_steps-1, dtype=tf.int32)

        x = self.sample_q(x_start, t, noise)
        predicted_noise = self.model(x, t)

        return self.loss_fn(noise, predicted_noise)

    def loss_step_debug(self, x_start):
        noise = tf.random.normal(x_start.shape)
        t = tf.random.uniform(x_start.shape[0:1], minval=0, maxval=self.diffusion_steps-1, dtype=tf.int32)

        x = self.sample_q(x_start, t, noise)
        # x = noise
        predicted_noise = self.model(x, t)

        def reconstruct(x_t, t, pred_noise):
            coeff1 = self.gather(self.sqrt_recip_alphas_cumprod, t)
            coeff2 = self.gather(self.sqrt_recipm1_alphas_cumprod, t)
            return coeff1*x_t - coeff2*pred_noise

        reconstruction = reconstruct(x, t, predicted_noise)
        return t.numpy(), [x_start.numpy(), x.numpy(), reconstruction.numpy(), noise.numpy(), predicted_noise.numpy(), np.abs(noise.numpy()-predicted_noise.numpy())]

    @tf.function
    def gather(self, x, index):
        return tf.gather(x, index)[..., tf.newaxis, tf.newaxis, tf.newaxis]

    @tf.function
    def train_step(self, x_batch):
        with tf.GradientTape() as tape:
            loss = self.loss_step(x_batch)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        grads_and_vars = zip(gradients, self.model.trainable_variables)
        self.optim.apply_gradients(grads_and_vars)
        return loss

    def sample(self, shape):
        # image_size = (batch_size, ) + self.model.input_shape[1:]
        images = []
        img = tf.random.normal(shape)

        for i in tqdm(reversed(range(0, self.diffusion_steps)), desc='sampling loop time step', total=self.diffusion_steps):
            t = tf.constant([i])
            img = self.p_sample(img, t)
            images.append(img.numpy())
            # print(tf.shape(img))

        return np.stack(images)

    @tf.function
    def p_sample(self, img, t):
        mean, variance = self.p_mean_variance(img, t)
        # mean = tf.clip_by_value(mean, -1, 1)
        stddev = tf.sqrt(variance)
        noise = tf.random.normal(img.shape)
        nonzero_mask = (1 - tf.cast(t == 0, tf.float32))[..., tf.newaxis, tf.newaxis, tf.newaxis]
        return mean + nonzero_mask*stddev*noise

    @tf.function
    def p_mean_variance(self, x, t):
        pred = self.model(x, t, training=False)
        x_start = self.predict_start_from_noise(x, t, noise=pred)
        x_start = tf.clip_by_value(x_start, -1, 1)
        return self.q_posterior(x_start, x, t)
    
    # @tf.function
    # def p_mean_variance(self, x, t):
    #     pred = self.model(x, t)
    #     mean = self.gather(self.sqrt_alpha_inv, t)*x - self.gather(self.sampling_coeff, t)*pred
    #     mean = tf.clip_by_value(mean, -1, 1)

    #     posterior_variance = self.gather(self.betas, t)
    #     return mean, posterior_variance

    @tf.function
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.gather(self.posterior_mean_coef1, t) * x_start + \
            self.gather(self.posterior_mean_coef2, t) * x_t

        posterior_variance = self.gather(self.betas, t)
        return posterior_mean, posterior_variance

    @tf.function
    def predict_start_from_noise(self, x_t, t, noise):
        return self.gather(self.sqrt_recip_alphas_cumprod, t) * x_t - self.gather(self.sqrt_recipm1_alphas_cumprod, t) * noise
