import torch
from torch import nn
import torch.nn.functional as F
from math import log2

factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]


def activation_func(activation: str) -> nn.Module:
    """
    Returns the corresponding activation function module.

    Args:
        activation (str): Name of the activation function. Options: 'relu', 'leaky_relu', 'selu', 'none'.

    Returns:
        nn.Module: The corresponding activation function module.
    """
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class DenseBlock(nn.Module):
    """
    Dense fully connected block with optional batch normalization and activation function.

    Attributes:
        dense (nn.Linear): Fully connected layer.
        norm (nn.BatchNorm1d): Batch normalization layer (if enabled).
        activate (nn.Module): Activation function.
    """

    def __init__(self, input_dim: int, output_dim: int, activation: str, normalization: bool = True) -> None:
        """
        Initializes a DenseBlock.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            activation (str): Activation function to use.
            normalization (bool): Whether to use batch normalization. Default is True.
        """
        super(DenseBlock, self).__init__()
        self.normalization = normalization
        self.dense = nn.Linear(input_dim, output_dim)
        if self.normalization:
            self.norm = nn.BatchNorm1d(output_dim)
        self.activate = activation_func(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.dense(x)
        if self.normalization:
            x = self.norm(x)
        x = self.activate(x)
        return x


class WSConv2d(nn.Module):
    """
    Weight-scaled convolutional layer.

    Attributes:
        conv (nn.Conv2d): Standard convolutional layer.
        scale (float): Scaling factor for weight normalization.
        bias (torch.Tensor): Bias term extracted separately.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, gain: int = 2) -> None:
        """
        Initializes a weight-scaled convolutional layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size. Default is 3.
            stride (int): Stride value. Default is 1.
            padding (int): Padding value. Default is 1.
            gain (int): Gain factor for weight normalization. Default is 2.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    """
    Pixel-wise feature normalization.

    Attributes:
        epsilon (float): Small constant for numerical stability.
    """

    def __init__(self) -> None:
        """
        Initializes the PixelNorm layer.
        """
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized output tensor.
        """
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
    """
    Convolutional block with weight-scaled convolutions, LeakyReLU activation, and optional pixel normalization.

    Attributes:
        conv1 (WSConv2d): First convolutional layer.
        conv2 (WSConv2d): Second convolutional layer.
        leaky (nn.LeakyReLU): Leaky ReLU activation function.
        pn (PixelNorm): Pixel normalization layer.
        use_pn (bool): Whether to use pixel normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, use_pixelnorm: bool = True) -> None:
        """
        Initializes a ConvBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_pixelnorm (bool): Whether to use pixel normalization. Default is True.
        """
        super().__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()
        self.use_pn = use_pixelnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x

class Generator(nn.Module):
    """
    Progressive GAN generator with conditional labels (ACGAN-style).

    Attributes:
        label_emb (nn.Embedding): Embedding layer for class labels.
        initial (nn.Sequential): Initial block for 4x4 feature map generation.
        initial_rgb (WSConv2d): Initial RGB layer for image generation.
        prog_blocks (nn.ModuleList): List of progressive convolutional blocks.
        rgb_layers (nn.ModuleList): List of RGB layers for different resolutions.
    """

    def __init__(self, z_dim: int, n_classes: int, in_channels: int, img_channels: int = 3) -> None:
        """
        Initializes the generator.

        Args:
            z_dim (int): Latent space dimension.
            n_classes (int): Number of classes for conditional generation.
            in_channels (int): Number of feature map channels.
            img_channels (int): Number of output image channels. Default is 3.
        """
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, z_dim)

        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim * 2, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList([self.initial_rgb])

        for i in range(len(factors) - 1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))

    def fade_in(self, alpha: float, upscaled: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        """
        Applies fade-in transition between resolutions.

        Args:
            alpha (float): Alpha value for interpolation (0 to 1).
            upscaled (torch.Tensor): Upscaled image from previous resolution.
            generated (torch.Tensor): New resolution output.

        Returns:
            torch.Tensor: Blended output.
        """
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x: torch.Tensor, label: torch.Tensor, alpha: float, steps: int) -> torch.Tensor:
        """
        Forward pass of the generator.

        Args:
            x (torch.Tensor): Latent vector (batch_size, z_dim, 1, 1).
            label (torch.Tensor): Class labels.
            alpha (float): Alpha for fade-in effect.
            steps (int): Number of progressive steps (0 for 4x4 images, increasing for larger).

        Returns:
            torch.Tensor: Generated image.
        """
        label_embedding = self.label_emb(label).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, label_embedding], dim=1)
        out = self.initial(x)

        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)

        return self.fade_in(alpha, final_upscaled, final_out)


class Discriminator(nn.Module):
    """
    Progressive GAN discriminator with conditional labels (ACGAN-style).

    Attributes:
        prog_blocks (nn.ModuleList): List of progressive convolutional blocks.
        rgb_layers (nn.ModuleList): List of RGB layers for input at different resolutions.
        embeddings (nn.ModuleList): List of embedding layers for conditioning.
        leaky (nn.LeakyReLU): Leaky ReLU activation.
        avg_pool (nn.AvgPool2d): Average pooling layer for downsampling.
        final_block (nn.Sequential): Final classification layers.
    """

    def __init__(self, in_channels: int, n_classes: int, img_channels: int = 3) -> None:
        """
        Initializes the discriminator.

        Args:
            in_channels (int): Number of feature map channels.
            n_classes (int): Number of classes for conditional discrimination.
            img_channels (int): Number of input image channels. Default is 3.
        """
        super().__init__()
        self.prog_blocks, self.rgb_layers, self.embeddings = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.leaky = nn.LeakyReLU(0.2)

        resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        for resolution in resolutions:
            self.embeddings.append(nn.Embedding(n_classes, resolution * resolution))

        for i in range(len(factors) - 1, 0, -1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c, use_pixelnorm=False))
            self.rgb_layers.append(WSConv2d(img_channels + 1, conv_in_c, kernel_size=1, stride=1, padding=0))

        self.initial_rgb = WSConv2d(img_channels + 1, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1),
        )

    def fade_in(self, alpha: float, downscaled: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """
        Applies fade-in transition between resolutions.

        Args:
            alpha (float): Alpha value for interpolation (0 to 1).
            downscaled (torch.Tensor): Downscaled image from previous resolution.
            out (torch.Tensor): Current resolution output.

        Returns:
            torch.Tensor: Blended output.
        """
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes minibatch standard deviation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with additional standard deviation channel.
        """
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor, alpha: float, steps: int) -> torch.Tensor:
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): Input image tensor.
            labels (torch.Tensor): Class labels.
            alpha (float): Alpha for fade-in effect.
            steps (int): Number of progressive steps.

        Returns:
            torch.Tensor: Discriminator output (logit values).
        """
        cur_step = len(self.prog_blocks) - steps
        embeddings = self.embeddings[steps](labels).view(labels.shape[0], 1, x.shape[2], x.shape[2])
        x = torch.cat([x, embeddings], dim=1)
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            out = self.final_block(out).view(out.shape[0], -1)
            return out

        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)


