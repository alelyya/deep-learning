class SegmenterModel(nn.Module):
    def __init__(self, in_size=3):
        super(SegmenterModel, self).__init__()

        # Downsample
        self.conv_1_0 = CBR_block(3, 32)
        self.conv_1_1 = CBR_block(32, 32)
        self.pool_1 = nn.MaxPool2d(2, return_indices=True)

        self.conv_2_0 = CBR_block(32, 64)
        self.conv_2_1 = CBR_block(64, 64)
        self.pool_2 = nn.MaxPool2d(2, return_indices=True)

        self.conv_3_0 = CBR_block(64, 128)
        self.conv_3_1 = CBR_block(128, 128)
        self.conv_3_2 = CBR_block(128, 128)
        self.pool_3 = nn.MaxPool2d(2, return_indices=True)

        self.conv_4_0 = CBR_block(128, 256)
        self.conv_4_1 = CBR_block(256, 256)
        self.conv_4_2 = CBR_block(256, 256)
        self.pool_4 = nn.MaxPool2d(2, return_indices=True)

        self.conv_5_0 = CBR_block(256, 256)
        self.conv_5_1 = CBR_block(256, 256)
        self.conv_5_2 = CBR_block(256, 256)
        self.pool_5 = nn.MaxPool2d(2, return_indices=True)

        # Upsample
        self.unpool_5 = nn.MaxUnpool2d(2)
        self.trconv_5_2 = CBR_block(256, 256)
        self.trconv_5_1 = CBR_block(256, 256)
        self.trconv_5_0 = CBR_block(256, 256)

        self.unpool_4 = nn.MaxUnpool2d(2)
        self.trconv_4_2 = CBR_block(256, 256)
        self.trconv_4_1 = CBR_block(256, 256)
        self.trconv_4_0 = CBR_block(256, 128)

        self.unpool_3 = nn.MaxUnpool2d(2)
        self.trconv_3_2 = CBR_block(128, 128)
        self.trconv_3_1 = CBR_block(128, 128)
        self.trconv_3_0 = CBR_block(128, 64)

        self.unpool_2 = nn.MaxUnpool2d(2)
        self.trconv_2_1 = CBR_block(64, 64)
        self.trconv_2_0 = CBR_block(64, 32)

        self.unpool_1 = nn.MaxUnpool2d(2)
        self.trconv_1_1 = CBR_block(32, 32)
        self.trconv_1_0 = CBR_block(32, 2)

        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        # Downsample
        x = self.conv_1_0(x)
        x = self.conv_1_1(x)
        x, indices_1 = self.pool_1(x)

        x = self.conv_2_0(x)
        x = self.conv_2_1(x)
        x, indices_2 = self.pool_2(x)

        x = self.conv_3_0(x)
        x = self.conv_3_1(x)
        x = self.conv_3_2(x)
        x, indices_3 = self.pool_3(x)

        x = self.conv_4_0(x)
        x = self.conv_4_1(x)
        x = self.conv_4_2(x)
        x, indices_4 = self.pool_4(x)

        x = self.conv_5_0(x)
        x = self.conv_5_1(x)
        x = self.conv_5_2(x)
        x, indices_5 = self.pool_5(x)

        # Upsample
        x = self.unpool_5(x, indices_5)
        x = self.trconv_5_2(x)
        x = self.trconv_5_1(x)
        x = self.trconv_5_0(x)

        x = self.unpool_4(x, indices_4)
        x = self.trconv_4_2(x)
        x = self.trconv_4_1(x)
        x = self.trconv_4_0(x)

        x = self.unpool_3(x, indices_3)
        x = self.trconv_3_2(x)
        x = self.trconv_3_1(x)
        x = self.trconv_3_0(x)

        x = self.unpool_2(x, indices_2)
        x = self.trconv_2_1(x)
        x = self.trconv_2_0(x)

        x = self.unpool_1(x, indices_1)
        x = self.trconv_1_1(x)
        x = self.trconv_1_0(x)

        return self.softmax(x)

ds = CarvanaDataset(train, train_masks)
dl = dt.DataLoader(ds, shuffle=True, num_workers=4, batch_size=32)

mean = torch.Tensor(3)
std = torch.Tensor(3)
num = 0
for input_, label in dl:
    
    current_mean = torch.mean(input_, dim = 0).mean(1).mean(1)
    current_std = torch.std(input_, dim = 0).mean(1).mean(1)
    mean += current_mean
    std += current_std

    num+=1
    
mean = mean / num
std = std / num
    
print(mean, std, sep = '\n')


mean=[0.6981, 0.6909, 0.6839]
std=[0.1628, 0.1638, 0.1615]