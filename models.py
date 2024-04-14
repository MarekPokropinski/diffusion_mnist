from functools import partial
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow_addons as tfa

initializer = tf.random_normal_initializer(0., 0.02)

BASE_SIZE = 64


class AlwaysDropout(tf.keras.layers.Dropout):
    '''
    Dropout that is active during training and evaluation
    '''

    def call(self, x):
        return super().call(x, training=True)


@tf.function
def rgb_to_yuv(rgb):
    return tf.image.rgb_to_yuv(rgb)
    rgb_to_yuv_mat = tf.constant([[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]], dtype=tf.float32)
    return tf.matmul(rgb, rgb_to_yuv_mat)


@tf.function
def rgb_to_grayscale(rgb):
    rgb_to_gray = tf.ones((3, 3), dtype=tf.float32) / 3.0
    return tf.matmul(rgb, rgb_to_gray)


def downsample_block(filters, kernel_size, dropout=False, norm=None, use_spectral_norm=False):
    conv1 = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=2, padding='same', kernel_initializer=initializer)
    if use_spectral_norm:
        conv1 = tfa.layers.SpectralNormalization(conv1)
    block = tf.keras.Sequential()
    block.add(conv1)
    if norm == 'BatchNorm':
        block.add(tf.keras.layers.BatchNormalization())
    elif norm == 'InstanceNorm':
        block.add(tfa.layers.InstanceNormalization())
    elif norm == 'LayerNorm':
        block.add(tf.keras.layers.LayerNormalization())
    elif norm is not None:
        raise Exception(f'Unknown normalization: {norm}')
    if dropout:
        block.add(AlwaysDropout(0.5))
    block.add(tf.keras.layers.LeakyReLU())
    return block


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')
        self.inst_norm = tfa.layers.InstanceNormalization()

    def build(self, input_shape):
        self.conv.build(input_shape)
        # self.inst_norm.build(self.conv.output.shape)
        super().build(input_shape)

    def call(self, x):
        x = self.conv(x)
        x = self.inst_norm(x)
        x = tf.nn.leaky_relu(x)
        return x


class DSConv(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')
        self.inst_norm = tfa.layers.InstanceNormalization()
        self.conv_block = ConvBlock(filters, strides=strides)

    def build(self, input_shape):
        # self.depthwise_conv.build(input_shape)
        # self.inst_norm.build(input_shape)
        # self.conv_block.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        x = self.depthwise_conv(x)
        x = self.inst_norm(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv_block(x)
        return x


class IRB(tf.keras.layers.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')
        self.inst_norm = tfa.layers.InstanceNormalization()
        self.conv_block = ConvBlock(filters=512, strides=1, kernel_size=1)

        self.conv = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same')
        self.inst_norm2 = tfa.layers.InstanceNormalization()

    def build(self, input_shape):
        # self.depthwise_conv.build(input_shape)
        # self.inst_norm.build(input_shape)
        # self.conv_block.build(input_shape)
        # self.conv.build(input_shape)
        # self.inst_norm2.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        inp = x

        x = self.conv_block(x)
        x = self.depthwise_conv(x)
        x = self.inst_norm(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv(x)
        x = self.inst_norm2(x)

        return x + inp


class UpConv(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs) -> None:
        super().__init__(**kwargs)
        self.DSConv = DSConv(filters)

    def build(self, input_shape):
        self.DSConv.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
        x = self.DSConv(x)
        return x


class DownConv(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs) -> None:
        super().__init__(**kwargs)
        self.DSConv1 = DSConv(filters, strides=1)
        self.DSConv2 = DSConv(filters, strides=2)

    def build(self, input_shape):
        # self.DSConv1.build(input_shape)
        # self.DSConv2.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        # x1 = tf.nn.avg_pool(x, 2, 2, padding='VALID')
        x1 = tf.nn.avg_pool2d(x, 2, 2, padding='VALID')
        x1 = self.DSConv1(x1)
        x2 = self.DSConv2(x)

        return x1 + x2


def ae_model(input_shape) -> tf.keras.Model:
    image = tf.keras.layers.Input(shape=input_shape)
    x = image

    x = ConvBlock(64)(x)
    x = ConvBlock(64)(x)

    x = DownConv(128)(x)
    x = ConvBlock(128)(x)
    x = DSConv(128)(x)

    x = DownConv(256)(x)
    x = ConvBlock(256)(x)

    for _ in range(8):
        x = IRB()(x)

    x = ConvBlock(256)(x)

    x = UpConv(128)(x)
    x = DSConv(128)(x)
    x = ConvBlock(128)(x)

    x = UpConv(64)(x)
    x = ConvBlock(64)(x)
    x = ConvBlock(64)(x)

    x = tf.keras.layers.Conv2D(input_shape[-1], kernel_size=3, activation='linear', padding='same')(x)

    return tf.keras.Model(inputs=image, outputs=x)


class SinusoidalPosEmb(Model):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def call(self, x):
        x = tf.cast(x, tf.float32)
        half_dim = self.dim // 2
        emb = tf.math.log(10000.0) / (half_dim - 1.0)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = tf.concat((tf.sin(emb), tf.cos(emb)), axis=-1)
        return emb


class Block(Model):
    def __init__(self, dim_out, groups=8):
        super().__init__()
        self.proj = layers.Conv2D(dim_out, 3, padding="same")
        self.norm = tfa.layers.GroupNormalization(groups)

    def call(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = tf.nn.silu(x)
        return x


class ResnetBlock(Model):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.use_scale_shift = time_emb_dim is not None
        if self.use_scale_shift:
            self.scale_mlp = layers.Dense(dim_out)
            self.shift_mlp = layers.Dense(dim_out)

        self.block1 = Block(dim_out, groups=groups)
        self.block2 = Block(dim_out, groups=groups)
        self.res_conv = layers.Conv2D(dim_out, 1, padding="same") if dim != dim_out else tf.identity

    def call(self, x, time_emb=None):
        scale_shift = None
        if self.use_scale_shift and time_emb is not None:
            scale = self.scale_mlp(time_emb)[:, tf.newaxis, tf.newaxis, :]
            shift = self.shift_mlp(time_emb)[:, tf.newaxis, tf.newaxis, :]
            scale_shift = scale, shift

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


class UNet(Model):
    def __init__(self, init_dim=64, dim_mults=(1, 2, 4, 8), resnet_block_groups=8):
        super().__init__()
        dim = init_dim
        dims = [init_dim,  *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_class = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim)

        self.time_mlp = Sequential([
            sinu_pos_emb,
            layers.Dense(time_dim, activation="gelu"),
            layers.Dense(time_dim)
        ])

        self.init_conv = layers.Conv2D(init_dim, 7, padding="same")

        self.downs = []
        self.ups = []

        num_resolutions = len(in_out)

        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= (num_resolutions - 1)

            block = [
                block_class(dim_in, dim_out, time_emb_dim=time_dim),
                block_class(dim_out, dim_out, time_emb_dim=time_dim),
            ]
            if not is_last:
                block.append(layers.Conv2D(dim_out, 4, 2, padding="same"))
            else:
                block.append(tf.identity)

            self.downs.append(block)

        mid_dim = dims[-1]

        self.mid_block1 = block_class(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = block_class(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            block = [
                block_class(dim_out*2, dim_in, time_emb_dim=time_dim),
                block_class(dim_in, dim_in, time_emb_dim=time_dim),
            ]
            if not is_last:
                block.append(layers.Conv2DTranspose(dim_out, 4, 2, padding="same"))
            else:
                block.append(tf.identity)

            self.ups.append(block)

        self.final_res_block = block_class(dim*2, dim, time_emb_dim=time_dim)
        self.final_conv = layers.Conv2D(64, 1, padding="same")

    def call(self, x, time):
        x = self.init_conv(x)
        r = x

        t = self.time_mlp(time)

        h = []

        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for block1, block2, upsample in self.ups:
            x = tf.concat((x, h.pop()), axis=-1)
            x = block1(x, t)
            x = block2(x, t)
            x = upsample(x)

        x = tf.concat((x, r), axis=-1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)[:, :, :, 0:1]
        return x
