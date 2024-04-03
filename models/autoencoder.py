import jax
import flax.linen as nn

class AutoEncoder4MNIST(nn.Module):

    nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))

        encoder_output = x

        x = nn.Dense(features=14 * 14 * 32)(x)

        x = nn.Dense(features=28 * 28 * 3)(x)

        x = x.reshape(x.shape[0], 3, 28, 28, 1)

        decoder_output = x

        return decoder_output, encoder_output


