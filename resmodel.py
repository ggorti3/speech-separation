import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float)

class TwoSpeakerRCPNet(nn.Module):
    def __init__(self, dim_f, dim_t):
        super().__init__()
        self.visual_layers = ResNetVisualModule()
        self.audio_layers = ResNetAudioModule()

        dim_h = 128 + 2 * (dim_f - 1)
        layer = nn.TransformerEncoderLayer(d_model=dim_h, nhead=8, dim_feedforward=1024, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 4)
        self.mask_head1 = nn.Linear(dim_h, dim_f * 2)
        self.mask_head2 = nn.Linear(dim_h, dim_f * 2)

        self.activation = nn.ReLU()

        self.dim_f = dim_f
        self.dim_t = dim_t

    def forward(self, z, s1, s2):
        """
        z.shape = (N, dim_f (always odd), dim_t, 2)
        si.shape = (N, 75, 128)
        """
        n = z.shape[0]

        zp = torch.permute(z, (0, 3, 1, 2))
        a = self.audio_layers(zp)
        a = a.reshape(n, 2 * (self.dim_f - 1), self.dim_t)

        s1 = torch.transpose(s1, 1, 2)
        v1 = self.visual_layers(s1)
        v1 = F.interpolate(v1, size=self.dim_t, mode='nearest')

        s2 = torch.transpose(s2, 1, 2)
        v2 = self.visual_layers(s2)
        v2 = F.interpolate(v2, size=self.dim_t, mode='nearest')

        x = torch.cat([a, v1, v2], dim=1)
        x = x.transpose(1, 2)


        x = self.encoder(x)

        m1 = self.mask_head1(x)
        m1 = m1.reshape(-1, self.dim_t, self.dim_f, 2)
        m1 = m1.transpose(1, 2)
        m2 = self.mask_head2(x)
        m2 = m2.reshape(-1, self.dim_t, self.dim_f, 2)
        m2 = m2.transpose(1, 2)

        # # complex multiplication
        # z1_real = z[:, :, :, 0] * m1[:, :, :, 0] - z[:, :, :, 1] * m1[:, :, :, 1]
        # z1_imag = z[:, :, :, 0] * m1[:, :, :, 1] + z[:, :, :, 1] * m1[:, :, :, 0]
        # z2_real = z[:, :, :, 0] * m2[:, :, :, 0] - z[:, :, :, 1] * m2[:, :, :, 1]
        # z2_imag = z[:, :, :, 0] * m2[:, :, :, 1] + z[:, :, :, 1] * m2[:, :, :, 0]

        # z1 = torch.stack([z1_real, z1_imag], dim=3)
        # z2 = torch.stack([z2_real, z2_imag], dim=3)

        # regular multiplication
        z1 = z * m1
        z2 = z * m2

        return z1, z2

class ResNetAudioModule(nn.Module):
    """
    ResNet Convolutional layers that process audio inputs
    """
    def __init__(self, ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, stride=(2, 1), padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.activation = nn.ReLU()

        self.resblock1 = ResBlock(64)
        self.resblock2 = ResBlock(64)
        self.resblock3 = ResBlock(64)
        self.resblock4 = ResBlock(64)
        self.resblock5 = ResBlock(64)
        self.resblock6 = ResBlock(64)
        self.resblock7 = ResBlock(64)
        self.resblock8 = ResBlock(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(num_features=8)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)

        x = self.activation(self.bn2(self.conv2(x)))
        return x

class ResNetVisualModule(nn.Module):
    """
    ResNet Convolutional layers that process video face encoding inputs
    """
    def __init__(self, ):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.activation = nn.ReLU()

        self.resblock1 = ResBlock(64, twoD=False)
        self.resblock2 = ResBlock(64, twoD=False)
        self.resblock3 = ResBlock(64, twoD=False)
        self.resblock4 = ResBlock(64, twoD=False)
    
    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, n_channels, twoD=True):
        super().__init__()

        if twoD:
            ConvClass = nn.Conv2d
            BNClass = nn.BatchNorm2d
        else:
            ConvClass = nn.Conv1d
            BNClass = nn.BatchNorm1d

        self.conv1 = ConvClass(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding='same')
        self.bn1 = BNClass(num_features=n_channels)
        self.conv2 = ConvClass(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding='same')
        self.bn2 = BNClass(num_features=n_channels)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x_orig = x
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + x_orig
        x = self.activation(x)
        return x
    
if __name__ == "__main__":
    from synthetic_data import TwoSpeakerData

    n_fft = 512
    win_length = 300
    hop_length = 150
    dim_f = 257
    dim_t = 295
    model = TwoSpeakerRCPNet(dim_f, dim_t)
    dataset = TwoSpeakerData("data/train_dataset/", n_fft, win_length, hop_length)

    iterator = iter(dataset)
    z, audio1, audio2, z1, z2, s1, s2 = next(iterator)
    z = z.unsqueeze(0)
    s1 = s1.unsqueeze(0)
    s2 = s2.unsqueeze(0)
    print(z.shape)
    z1_hat, z2_hat = model(z, s1, s2)
    print(z1_hat.shape)
    print(z2_hat.shape)