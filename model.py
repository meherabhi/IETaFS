import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm

#This class is used to initialize the weights of the network
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()
# apply(fn): Applies fn recursively to every submodule
# init_weights(self) applies the weight_init_fn defined below for every submodule of self ( i.e all the layers of self)
    def init_weights(self):
        self.apply(self._weights_init_fn)
# This function initializes weights of the network  
    def _weights_init_fn(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class ResidualBlock(BaseNetwork):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
# Define the residual network architecture
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
# Initialize the weights of the network
        self.init_weights()
# In the forward pass: the input added with the output of the network is returned?
    def forward(self, x):
        return x + self.main(x)


class Generator(BaseNetwork):
    """Generator network."""
# c_dim: number of Action Units to use to train the model.
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()
# The network architecture is appended layer by layer to the list called layers
        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7,
                                stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(
            conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
# self.debug1 holds the layers sequence
        self.debug1 = nn.Sequential(*layers)

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2,
                                    kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(
                curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
# To the list of layers in self.debug1, few more are appended and this new extended list is stored in self.debug2?
        self.debug2 = nn.Sequential(*layers)

        # Bottleneck layers.
# To the list of layers in self.debug2, few more are appended and this new extended list is stored in self.debug3?
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.debug3 = nn.Sequential(*layers)

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim //
                                             2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(
                curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        self.debug4 = nn.Sequential(*layers)
# similarily the main and debug4 parts are formed

# The color regression and the attention networks are defined below

        # Same architecture for the color regression
        layers = []
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7,
                                stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.im_reg = nn.Sequential(*layers)

        # One Channel output and Sigmoid function for the attention layer
        layers = []
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7,
                                stride=1, padding=3, bias=False))
        layers.append(nn.Sigmoid())  # Values between 0 and 1
        self.im_att = nn.Sequential(*layers)
# Weights Initialization for color regression and the attention networks 
        self.init_weights()

    def forward(self, x, c):
# Here c is the activation unit tensor which is replicated spatially and concatenated to the input image.
# Add a dimension 1 at the 2nd dim and 3rd dim respectively, so now c becomes 4 dimensional
        c = c.unsqueeze(2).unsqueeze(3)
# Say embed1= [1, 2, 3] shape is (3), Now I want to expand embed1 to [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
# This is done using expand
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))

        x = torch.cat([x, c], dim=1)
# This new Input is passed through the generator and the attention, color regression networks
        features = self.main(x)

        reg = self.im_reg(features)
        att = self.im_att(features)

        return att, reg


class Discriminator(BaseNetwork):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
# Define the layers of the discrminator by appending them initially to a list and then using nn.sequential
        layers = []
        layers.append(
            nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2,
                                    kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(
            curr_dim, c_dim, kernel_size=kernel_size, bias=False)

        self.init_weights()

    def forward(self, x):
# The main network is common for both the discriminators 
# so first pass through the main and then spearately through the other 2 
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
# .squueze:Returns a tensor with all the dimensions of input of size 1 removed.
# so essentially returning a scalar in this case
        # out_cls.view(out_cls.size(0), out_cls.size(1))
        return out_src.squeeze(), out_cls.squeeze()
    
class AU_Torch(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size =(5,1), stride=(2,1))
    self.conv1_bn   = nn.BatchNorm2d(32)
    self.pad1 = nn.ZeroPad2d((0,0,2,2))
    self.conv2 = nn.Conv2d(32, 64, kernel_size =(3,1), stride=(2,1))
    self.conv2_bn   = nn.BatchNorm2d(64)
    self.pad2 = nn.ZeroPad2d((0,0,1,1))
    self.conv3 = nn.Conv2d(64,128, kernel_size =(3,1), stride=(2,1))
    self.conv3_bn   = nn.BatchNorm2d(128)
    self.pad3 = nn.ZeroPad2d((0,0,1,1))
    self.conv4 = nn.Conv2d(128,256, kernel_size =(3,1),stride=(2,1))
    self.conv4_bn   = nn.BatchNorm2d(256)
    self.pad4 = nn.ZeroPad2d((0,0,1,1))
    self.conv5 = nn.Conv2d(256,512, kernel_size =(3,1), stride=(2,1))
    self.conv5_bn   = nn.BatchNorm2d(512)
    self.pad5 = nn.ZeroPad2d((0,0,1,1))
    self.conv6 = nn.Conv2d(512,512, kernel_size = (1,3), stride=(1,2))
    self.conv6_bn   = nn.BatchNorm2d(512)
    self.pad6 = nn.ZeroPad2d((1,1,0,0))
    self.conv7 = nn.Conv2d(512,512, kernel_size =(1,3), stride=(1,2))
    self.conv7_bn   = nn.BatchNorm2d(512)
    self.pad7 = nn.ZeroPad2d((1,1,0,0))
    self.conv8 = nn.Conv2d(512,512, kernel_size =(1,3), stride= (1,2))
    self.conv8_bn   = nn.BatchNorm2d(512)
    self.pad8 = nn.ZeroPad2d((1,1,0,0))
    self.fc1 = nn.Linear(512*4*4, 1024)
    self.fc2 = nn.Linear(1024, 17)
  def forward (self , x):
    x = F.leaky_relu_(self.conv1_bn(self.conv1(self.pad1(x))), negative_slope=0.2)
    x = F.leaky_relu_(self.conv2_bn(self.conv2(self.pad2(x))), negative_slope=0.2)
    x = F.leaky_relu_(self.conv3_bn(self.conv3(self.pad3(x))), negative_slope=0.2)
    x = F.leaky_relu_(self.conv4_bn(self.conv4(self.pad4(x))), negative_slope=0.2)
    x = F.leaky_relu_(self.conv5_bn(self.conv5(self.pad5(x))), negative_slope=0.2)
    x = F.leaky_relu_(self.conv6_bn(self.conv6(self.pad6(x))), negative_slope=0.2)
    x = F.leaky_relu_(self.conv7_bn(self.conv7(self.pad7(x))), negative_slope=0.2)
    x = F.leaky_relu_(self.conv8_bn(self.conv8(self.pad8(x))), negative_slope=0.2)
    x = x.view(-1,512*4*4 )
    x = torch.tanh(self.fc1(x))
    x = F.dropout(x, p=0.2, training=self.training)
    x = torch.sigmoid(self.fc2(x))
    return x

class audio_network(nn.Module):
  def __init__(self):
    super().__init__()
    # 1x128x32
    self.conv1 = nn.Conv2d(1, 32, kernel_size =(5,1), padding=(2, 0), stride=(2,1))
    self.conv1_bn   = nn.BatchNorm2d(32)
    # 32x64x32
    self.conv2 = nn.Conv2d(32, 64, kernel_size =(3,1), padding=(1, 0), stride=(2,1))
    self.conv2_bn   = nn.BatchNorm2d(64)
    # 64x32x32
    self.conv3 = nn.Conv2d(64,128, kernel_size =(3,1),  padding=(1, 0),stride=(2,1))
    self.conv3_bn   = nn.BatchNorm2d(128)
    # 128x16x32
    self.conv4 = nn.Conv2d(128,256, kernel_size =(3,1),  padding=(1, 0),stride=(2,1))
    self.conv4_bn   = nn.BatchNorm2d(256)
    # 256x8x32
    self.conv5 = nn.Conv2d(256,512, kernel_size =(1,3),  padding=(0, 1), stride=(1,2))
    self.conv5_bn   = nn.BatchNorm2d(512)
    # 512x8x16
    self.conv6 = nn.Conv2d(512,512, kernel_size = (1,3),  padding=(0, 1),stride=(1,2))
    self.conv6_bn   = nn.BatchNorm2d(512)
    # 512x8x8
    self.fc1 = nn.Linear(512*8*8, 2048)
    # 1x2048
    self.fc2 = nn.Linear(2048, 256)
    # 1x256
  def forward (self , x):
    x = F.leaky_relu_(self.conv1_bn(self.conv1(x)), negative_slope=0.2)
    x = F.leaky_relu_(self.conv2_bn(self.conv2(x)), negative_slope=0.2)
    x = F.leaky_relu_(self.conv3_bn(self.conv3(x)), negative_slope=0.2)
    x = F.leaky_relu_(self.conv4_bn(self.conv4(x)), negative_slope=0.2)
    x = F.leaky_relu_(self.conv5_bn(self.conv5(x)), negative_slope=0.2)
    x = F.leaky_relu_(self.conv6_bn(self.conv6(x)), negative_slope=0.2)
    x = x.view(-1,512*8*8 )
    x = F.leaky_relu_(self.fc1(x))
    x = self.fc2(x)
    x = F.normalize(x, p=2, dim=1)
    return x

class video_network(nn.Module):
  def __init__(self):
    super().__init__()
    # 3x128x128
    self.conv1 = nn.Conv2d(3, 32, kernel_size =(3,3), padding=(1, 1), stride=(2,2))
    self.conv1_bn   = nn.BatchNorm2d(32)
    # 32x64x64
    self.conv2 = nn.Conv2d(32, 64, kernel_size =(3,3), padding=(1, 1), stride=(2,2))
    self.conv2_bn   = nn.BatchNorm2d(64)
    # 64x32x32
    self.conv3 = nn.Conv2d(64,128, kernel_size =(3,3),  padding=(1, 1),stride=(2,2))
    self.conv3_bn   = nn.BatchNorm2d(128)
    # 128x16x16
    self.conv4 = nn.Conv2d(128,256, kernel_size =(3,3),  padding=(1, 1),stride=(2,2))
    self.conv4_bn   = nn.BatchNorm2d(256)
    # 256x8x8
    self.conv5 = nn.Conv2d(256,512, kernel_size =(3,3),  padding=(1, 1), stride=(2,2))
    self.conv5_bn   = nn.BatchNorm2d(512)
    # 512x4x4
    self.conv6 = nn.Conv2d(512,512, kernel_size = (3,3),  padding=(1, 1),stride=(2,2))
    self.conv6_bn   = nn.BatchNorm2d(512)
    # 512x2x2
    self.fc1 = nn.Linear(512*2*2, 1024)
    # 1x1024
    self.fc2 = nn.Linear(1024, 256)
    # 1x256
  def forward (self , x):
    x = F.leaky_relu_(self.conv1_bn(self.conv1(x)), negative_slope=0.2)
    x = F.leaky_relu_(self.conv2_bn(self.conv2(x)), negative_slope=0.2)
    x = F.leaky_relu_(self.conv3_bn(self.conv3(x)), negative_slope=0.2)
    x = F.leaky_relu_(self.conv4_bn(self.conv4(x)), negative_slope=0.2)
    x = F.leaky_relu_(self.conv5_bn(self.conv5(x)), negative_slope=0.2)
    x = F.leaky_relu_(self.conv6_bn(self.conv6(x)), negative_slope=0.2)
    x = x.view(-1,512*2*2 )
    x =  F.leaky_relu_(self.fc1(x))
    x = self.fc2(x)
    x = F.normalize(x, p=2, dim=1)
    return x


def conv2d_with_spectral_norm(in_channels, out_channels, kernel_size, stride, padding):
    # return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    return spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))

def marginal_pdf(values: torch.Tensor, bins: torch.Tensor, sigma: torch.Tensor,
                 epsilon: float = 1e-10):
    """Function that calculates the marginal probability distribution function of the input tensor
        based on the number of histogram bins.

    Args:
        values (torch.Tensor): shape [BxNx1].
        bins (torch.Tensor): shape [NUM_BINS].
        sigma (torch.Tensor): shape [1], gaussian smoothing factor.
        epsilon: (float), scalar, for numerical stability.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
          - torch.Tensor: shape [BxN].
          - torch.Tensor: shape [BxNxNUM_BINS].

    """

    if not isinstance(values, torch.Tensor):
        raise TypeError("Input values type is not a torch.Tensor. Got {}"
                        .format(type(values)))

    if not isinstance(bins, torch.Tensor):
        raise TypeError("Input bins type is not a torch.Tensor. Got {}"
                        .format(type(bins)))

    if not isinstance(sigma, torch.Tensor):
        raise TypeError("Input sigma type is not a torch.Tensor. Got {}"
                        .format(type(sigma)))

    if not values.dim() == 3:
        raise ValueError("Input values must be a of the shape BxNx1."
                         " Got {}".format(values.shape))

    if not bins.dim() == 1:
        raise ValueError("Input bins must be a of the shape NUM_BINS"
                         " Got {}".format(bins.shape))

    if not sigma.dim() == 0:
        raise ValueError("Input sigma must be a of the shape 1"
                         " Got {}".format(sigma.shape))

    residuals = values - bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
    pdf = pdf / normalization

    return (pdf, kernel_values)


def joint_pdf(kernel_values1: torch.Tensor, kernel_values2: torch.Tensor,
              epsilon: float = 1e-10) -> torch.Tensor:
    """Function that calculates the joint probability distribution function of the input tensors
       based on the number of histogram bins.

    Args:
        kernel_values1 (torch.Tensor): shape [BxNxNUM_BINS].
        kernel_values2 (torch.Tensor): shape [BxNxNUM_BINS].
        epsilon (float): scalar, for numerical stability.

    Returns:
        torch.Tensor: shape [BxNUM_BINSxNUM_BINS].

    """

    if not isinstance(kernel_values1, torch.Tensor):
        raise TypeError("Input kernel_values1 type is not a torch.Tensor. Got {}"
                        .format(type(kernel_values1)))

    if not isinstance(kernel_values2, torch.Tensor):
        raise TypeError("Input kernel_values2 type is not a torch.Tensor. Got {}"
                        .format(type(kernel_values2)))

    if not kernel_values1.dim() == 3:
        raise ValueError("Input kernel_values1 must be a of the shape BxN."
                         " Got {}".format(kernel_values1.shape))

    if not kernel_values2.dim() == 3:
        raise ValueError("Input kernel_values2 must be a of the shape BxN."
                         " Got {}".format(kernel_values2.shape))

    if kernel_values1.shape != kernel_values2.shape:
        raise ValueError("Inputs kernel_values1 and kernel_values2 must have the same shape."
                         " Got {} and {}".format(kernel_values1.shape, kernel_values2.shape))

    joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
    normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + epsilon
    pdf = joint_kernel_values / normalization

    return pdf

def histogram2d(
        x1: torch.Tensor,
        x2: torch.Tensor,
        bin1: torch.Tensor,
        bin2: torch.Tensor,
        bandwidth: torch.Tensor,
        epsilon: float = 1e-10) -> torch.Tensor:
    """Function that estimates the 2d histogram of the input tensor.

    The calculation uses kernel density estimation which requires a bandwidth (smoothing) parameter.

    Args:
        x1 (torch.Tensor): Input tensor to compute the histogram with shape :math:`(B, D1)`.
        x2 (torch.Tensor): Input tensor to compute the histogram with shape :math:`(B, D2)`.
        bins (torch.Tensor): The number of bins to use the histogram :math:`(N_{bins})`.
        bandwidth (torch.Tensor): Gaussian smoothing factor with shape shape [1].
        epsilon (float): A scalar, for numerical stability. Default: 1e-10.

    Returns:
        torch.Tensor: Computed histogram of shape :math:`(B, N_{bins}), N_{bins})`.
    """

    pdf1, kernel_values1 = marginal_pdf(x1.unsqueeze(2), bin1, bandwidth, epsilon)
    pdf2, kernel_values2 = marginal_pdf(x2.unsqueeze(2), bin2, bandwidth, epsilon)

    pdf = joint_pdf(kernel_values1, kernel_values2)

    return pdf

def rgb_to_linear_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an sRGB image to linear RGB. Used in colorspace conversions.

    Args:
        image (torch.Tensor): sRGB Image to be converted to linear RGB of shape :math:`(*,3,H,W)`.

    Returns:
        torch.Tensor: linear RGB version of the image with shape of :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_linear_rgb(input) # 2x3x4x5
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W).Got {}"
                         .format(image.shape))

    lin_rgb: torch.Tensor = torch.where(image > 0.04045, torch.pow(
        ((image + 0.055) / 1.055), 2.4), image / 12.92)

    return lin_rgb

def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    r"""Converts a RGB image to XYZ.

    Args:
        image (torch.Tensor): RGB Image to be converted to XYZ with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: XYZ version of the image with shape :math:`(*, 3, H, W)`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    x: torch.Tensor = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y: torch.Tensor = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z: torch.Tensor = 0.019334 * r + 0.119193 * g + 0.950227 * b

    out: torch.Tensor = torch.stack([x, y, z], -3)

    return out

def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    r"""Converts a RGB image to Lab.

    The image data is assumed to be in the range of :math:`[0, 1]`. Lab
    color is computed using the D65 illuminant and Observer 2.

    Args:
        image (torch.Tensor): RGB Image to be converted to Lab with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: Lab version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_lab(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)

    xyz_im: torch.Tensor = rgb_to_xyz(lin_rgb)

    # normalize for D65 white point
    xyz_ref_white = torch.tensor(
        [0.95047, 1., 1.08883], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    threshold = 0.008856
    power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.)
    scale = 7.787 * xyz_normalized + 4. / 29.
    xyz_int = torch.where(xyz_normalized > threshold, power, scale)

    x: torch.Tensor = xyz_int[..., 0, :, :]
    y: torch.Tensor = xyz_int[..., 1, :, :]
    z: torch.Tensor = xyz_int[..., 2, :, :]

    L: torch.Tensor = (116. * y) - 16.
    a: torch.Tensor = 500. * (x - y)
    _b: torch.Tensor = 200. * (y - z)

    out: torch.Tensor = torch.stack([L, a, _b], dim=-3)

    return out

class Color_D(nn.Module):
    def __init__(self):#, bandwidth = torch.tensor(0.5)):
        super(Color_D, self).__init__()
        self.n_bins = torch.linspace(0, 1, 64)
        # self.bandwidth = bandwidth
        # self.hist = lambda x,y : histogram2d(x, y, bins = self.n_bins, bandwidth = self.bandwidth)
        base_dim = 64
        self.net = nn.Sequential(conv2d_with_spectral_norm(in_channels = 3, out_channels = base_dim, kernel_size = 3, stride = 2, padding = 1),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 conv2d_with_spectral_norm(in_channels = base_dim, out_channels = base_dim * 2, kernel_size = 3, stride = 2, padding = 1),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 conv2d_with_spectral_norm(in_channels = base_dim * 2, out_channels = base_dim * 4, kernel_size = 3, stride = 2, padding = 1),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 conv2d_with_spectral_norm(in_channels = base_dim * 4, out_channels = base_dim * 8, kernel_size = 3, stride = 2, padding = 1),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 conv2d_with_spectral_norm(in_channels = base_dim * 8, out_channels = 1, kernel_size = 3, stride = 2, padding = 0)
                                 )

    def forward(self, x):
        """
        x : torch.tensor of shape (B, C, H, W)
        out : torch.tensor of shape (B,)
        """
        nbatches = x.shape[0]
        x = (x+1)/2
        x = rgb_to_lab(x)
        L, a, b = x[:, 0].view(nbatches, -1), x[:, 1].view(nbatches, -1), x[:, 2].view(nbatches, -1)
        L_bins = torch.linspace(0, 100, 64).to(x.device)
        ab_bins = torch.linspace(-128, 127, 64).to(x.device)
        bandwidth = torch.tensor(0.5).to(x.device)
        hist = lambda x, y, bin1, bin2 : histogram2d(x, y, bin1 = bin1, bin2 = bin2, bandwidth = bandwidth)
        hist = torch.cat([hist(x, y, bin1, bin2).unsqueeze(1) for x, y, bin1, bin2 in zip([L, a, b], [a, b, L], [L_bins, ab_bins, ab_bins], [ab_bins, ab_bins, L_bins])], dim = 1)
        out = self.net(hist).squeeze()
        return out


# class Pose_D(BaseNetwork):
#     """Pose Discriminator network."""

#     def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
#         super(Pose_D, self).__init__()
# # Define the layers of the discrminator by appending them initially to a list and then using nn.sequential
#         layers = []
#         layers.append(
#             nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
#         layers.append(nn.LeakyReLU(0.01))

#         curr_dim = conv_dim
#         for i in range(1, repeat_num):
#             layers.append(nn.Conv2d(curr_dim, curr_dim*2,
#                                     kernel_size=4, stride=2, padding=1))
#             layers.append(nn.LeakyReLU(0.01))
#             curr_dim = curr_dim * 2

#         kernel_size = int(image_size / np.power(2, repeat_num))
#         self.main = nn.Sequential(*layers)
#         self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.conv2 = nn.Conv2d(
#             curr_dim, c_dim, kernel_size=kernel_size, bias=False)

#         self.init_weights()

#     def forward(self, x):
# # The main network is common for both the discriminators 
# # so first pass through the main and then spearately through the other 2 
#         h = self.main(x)
#         out_src = self.conv1(h)
#         out_cls = self.conv2(h)
# # .squueze:Returns a tensor with all the dimensions of input of size 1 removed.
# # so essentially returning a scalar in this case
#         # out_cls.view(out_cls.size(0), out_cls.size(1))
#         return out_src.squeeze(), out_cls.squeeze()





