import torch
import torch.nn as nn
# For MNIST 28x28 images
class DiscriminativeNetwork(nn.Module):
    def __init__(self, channels_img, features_d):
        super(DiscriminativeNetwork, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 28x28
            nn.Conv2d(channels_img, features_d, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2), # 14x14
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d    , features_d * 4, 4, 2, 1), # 7x7
            self._block(features_d * 4, features_d * 16, 4, 3, 0), # 2x2
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 16, 1, kernel_size=4, stride=2, padding=1), #1x1
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class GenerativeNetwork(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(GenerativeNetwork, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 2, 1), #2x2
            self._block(features_g * 16, features_g * 4, 4, 3, 0), #7x7
            self._block(features_g * 4, features_g * 1, 4, 2, 1), #14x14
            nn.ConvTranspose2d(
                features_g, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 28x28
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def init_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = DiscriminativeNetwork(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = GenerativeNetwork(noise_dim, in_channels, 8)
    init_weights(gen)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")


if __name__ == "__main__":
    test()
