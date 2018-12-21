import torch.nn as nn


def CBR_block(in_planes, out_planes, upsample=False, kernel=3, stride=1, padding=1, output_padding=0):
    """
    Stacks convolution (or transpose convolution, if upsample=True),
    batch normalization and ReLU into one sequential layer.
    """

    if upsample:
        conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel, stride=stride,
                                  padding=padding, output_padding=output_padding)
    else:
        conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride,
                         padding=padding)

    return nn.Sequential(conv,
                         nn.BatchNorm2d(num_features=out_planes),
                         nn.ReLU(True))


class SegmenterModel(nn.Module):
    """
    Simplified SegNet.
    5 convolution layers in downsample (symmetrical to upsample)
    2 max-pooling layers in downsample (symmetrical to upsample)
    """

    def __init__(self, in_size=3):
        super(SegmenterModel, self).__init__()

        self.downsize = CBR_block(3, 16, kernel=3, stride=2)
        # Downsample
        self.conv_h0 = CBR_block(16, 32)
        self.pool_h = nn.MaxPool2d(2, return_indices=True)
        self.conv_m0 = CBR_block(32, 32)
        self.conv_m1 = CBR_block(32, 64)
        self.pool_m = nn.MaxPool2d(2, return_indices=True)
        self.conv_l0 = CBR_block(64, 64)

        # Upsample
        self.conv_l1 = CBR_block(64, 64)
        self.unpool_m = nn.MaxUnpool2d(2)
        self.conv_m3 = CBR_block(64, 32)
        self.conv_m4 = CBR_block(32, 32)
        self.unpool_h = nn.MaxUnpool2d(2)
        self.conv_h1 = CBR_block(32, 16)

        self.upsize = CBR_block(16, 2, upsample = True, kernel=3, stride=2, output_padding=1)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.downsize(x)

        #Downsample
        x = self.conv_h0(x)
        x, indices_h = self.pool_h(x)
        x = self.conv_m0(x)
        x = self.conv_m1(x)
        x, indices_m = self.pool_m(x)
        x = self.conv_l0(x)

        # Upsample
        x = self.conv_l1(x)
        x = self.unpool_m(x, indices_m)
        x = self.conv_m3(x)
        x = self.conv_m4(x)
        x = self.unpool_h(x, indices_h)
        x = self.conv_h1(x)

        x = self.upsize(x)
        return self.softmax(x)
