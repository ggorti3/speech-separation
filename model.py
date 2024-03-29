import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float)

class TwoSpeakerCPNet(nn.Module):
    def __init__(self, dim_f, dim_t):
        super().__init__()
        self.visual_layers = VisualModule()
        self.audio_layers = AudioModule()
        #self.backbone = LSTMBackbone(dim_f)
        self.backbone = TransformerBackbone(dim_f)
        self.linear1 = nn.Linear(400, 600)
        self.bn1 = nn.BatchNorm1d(num_features=dim_t)
        self.linear2 = nn.Linear(600, 600)
        self.bn2 = nn.BatchNorm1d(num_features=dim_t)
        self.linear3 = nn.Linear(600, 600)
        self.bn3 = nn.BatchNorm1d(num_features=dim_t)
        self.mask_head1 = nn.Linear(600, dim_f * 2)
        self.mask_head2 = nn.Linear(600, dim_f * 2)

        self.activation = nn.ReLU()

        self.dim_f = dim_f
        self.dim_t = dim_t

    def forward(self, z, s1, s2):
        """
        z.shape = (N, 257, 295, 2)
        si.shape = (N, 75, 128)
        """

        zp = torch.permute(z, (0, 3, 1, 2))
        a = self.audio_layers(zp)
        a = a.reshape(-1, 8 * self.dim_f, self.dim_t)

        s1 = torch.transpose(s1, 1, 2)
        v1 = self.visual_layers(s1)
        v1 = F.interpolate(v1, size=self.dim_t, mode='nearest')

        s2 = torch.transpose(s2, 1, 2)
        v2 = self.visual_layers(s2)
        v2 = F.interpolate(v2, size=self.dim_t, mode='nearest')

        x = torch.cat([a, v1, v2], dim=1)
        x = x.transpose(1, 2)
        x = self.backbone(x)
        x = self.bn1(self.activation(self.linear1(x)))
        x = self.bn2(self.activation(self.linear2(x)))
        x = self.bn3(self.activation(self.linear3(x)))

        m1 = self.mask_head1(x)
        m1 = m1.reshape(-1, self.dim_t, self.dim_f, 2)
        m1 = m1.transpose(1, 2)
        m2 = self.mask_head1(x)
        m2 = m2.reshape(-1, self.dim_t, self.dim_f, 2)
        m2 = m2.transpose(1, 2)

        # complex multiplication
        z1_real = z[:, :, :, 0] * m1[:, :, :, 0] - z[:, :, :, 1] * m1[:, :, :, 1]
        z1_imag = z[:, :, :, 0] * m1[:, :, :, 1] + z[:, :, :, 1] * m1[:, :, :, 0]
        z2_real = z[:, :, :, 0] * m2[:, :, :, 0] - z[:, :, :, 1] * m2[:, :, :, 1]
        z2_imag = z[:, :, :, 0] * m2[:, :, :, 1] + z[:, :, :, 1] * m2[:, :, :, 0]

        z1 = torch.stack([z1_real, z1_imag], dim=3)
        z2 = torch.stack([z2_real, z2_imag], dim=3)

        return z1, z2

class AudioModule(nn.Module):
    """
    Convolutional layers that process audio inputs
    """
    def __init__(self, ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(2, 96, kernel_size=(1, 7), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=(7, 1), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=5, dilation=(2, 1), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=5, dilation=(4, 1), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=5, dilation=(8, 1), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=5, dilation=(16, 1), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=5, dilation=(32, 1), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=5, dilation=(2, 2), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=5, dilation=(4, 4), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=5, dilation=(8, 8), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=5, dilation=(16, 16), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, kernel_size=5, dilation=(32, 32), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 8, kernel_size=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )
    
    def forward(self, x):
        return self.layer(x)

class VisualModule(nn.Module):
    """
    Convolutional layers that process visual inputs
    """
    def __init__(self,):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=7, padding="same"),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=5, dilation=2, padding="same"),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=5, dilation=4, padding="same"),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=5, dilation=8, padding="same"),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=5, dilation=16, padding="same"),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
    
    def forward(self, x):
        return self.layer(x)

class LSTMBackbone(nn.Module):
    def __init__(self, dim_f):
        super().__init__()
        self.lstm = nn.LSTM(input_size=8*dim_f + 128, hidden_size=200, batch_first=True, bidirectional=True)
    
    def forward(self, x):
        x, (_, _) = self.lstm(x)
        return x

class TransformerBackbone(nn.Module):
    def __init__(self, dim_f):
        super().__init__()
        self.linear = nn.Linear(8*dim_f + 128, 400)
        layer = nn.TransformerEncoderLayer(d_model=400, nhead=8, dim_feedforward=1024, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1)

        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.linear(x))
        x = self.encoder(x)
        return x
    
if __name__ == "__main__":
    from synthetic_data import TwoSpeakerData

    n_fft = 256
    win_length = 256
    hop_length = 128
    dim_f = 129
    dim_t = 345
    model = TwoSpeakerCPNet(dim_f, dim_t)
    dataset = TwoSpeakerData("data/train_dataset", n_fft, win_length, hop_length)

    iterator = iter(dataset)
    z, audio1, audio2, z1, z2, s1, s2 = next(iterator)
    z = z.unsqueeze(0)
    s1 = s1.unsqueeze(0)
    s2 = s2.unsqueeze(0)
    print(z.shape)
    z1_hat, z2_hat = model(z, s1, s2)
    print(z1_hat.shape)
    print(z2_hat.shape)