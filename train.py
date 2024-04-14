import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from gaussian_diffusion import GaussianDiffusion
import matplotlib.pyplot as plt

from models import UNet, ae_model

# BATCH_SIZE = 64
BATCH_SIZE = 256
steps_per_epoch = 60000//BATCH_SIZE


def preprocess_image(x):
    x = tf.cast(x["image"], dtype=tf.float32)
    x = tf.image.resize(x, [32, 32], method=tf.image.ResizeMethod.BICUBIC)
    x = x/127.5-1.0
    return x


train_ds = tfds.load("mnist", split=tfds.Split.TRAIN).map(preprocess_image).repeat().shuffle(1024).batch(BATCH_SIZE)

# model = tf.keras.models.load_model("diffusion_model.tf")
# inp = tf.keras.layers.Input(shape=(28, 28, 1))
# x = model(inp)
# x = x[..., 0]
# model = tf.keras.Model(inp, x)
# model = ae_model((28, 28, 1))
# model.load_weights("diffusion_model.tf")
model = UNet()
# model.load_weights("diffusion_model.tf")
diffusion = GaussianDiffusion(model)

losses = []
e = 0

t = tqdm(enumerate(train_ds), total=steps_per_epoch)
for i, x in t:
    loss = diffusion.train_step(x)
    losses.append(loss.numpy())
    # print(f"{np.mean(losses):.6f}")
    if i % steps_per_epoch == 0:
        print(f"epoch {e}, loss: {np.mean(losses):.6f}")
        model.save("diffusion_model.keras")
        e += 1
        losses.clear()
        t.reset()


# for i, x in tqdm(enumerate(train_ds)):
#     t, outputs = diffusion.loss_step_debug(x)
#     fig, ax = plt.subplots(BATCH_SIZE, len(outputs))

#     for i in range(BATCH_SIZE):
#         for j, out in enumerate(outputs):
#             print(out.shape)
#             ax[i, j].imshow(out[i], cmap="gray_r", vmin=-1, vmax=1)
#             ax[i, j].set_title(f'{t[i]}')
#             ax[i, j].axis("off")
#     print(np.square(outputs[-1]).mean())

#     plt.show()
#     plt.close(fig)
