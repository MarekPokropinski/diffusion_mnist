import tensorflow as tf
import matplotlib.pyplot as plt

from gaussian_diffusion import GaussianDiffusion
from models import UNet

model = UNet()
diffusion = GaussianDiffusion(model)
model(tf.zeros([4, 32, 32, 1]), tf.zeros([4]))

model.load_weights("diffusion_model.keras")
pred = diffusion.sample([25, 32, 32, 1])
fig, ax = plt.subplots(5, 5)
for i in range(25):
    ax[i//5, i%5].imshow((pred[-1, i]+1.0)/2.0, cmap="gray_r")
plt.show()