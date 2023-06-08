import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional.audio import signal_distortion_ratio
from tqdm import tqdm

from run import loss_func

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device {}".format(DEVICE))

def process_audio(model, z, audio1, audio2, z1, z2, s1, s2, n_fft, win_length, hop_length):
    with torch.no_grad():
        model.eval()
        z = z.to(DEVICE)
        z1 = z1.to(DEVICE)
        z2 = z2.to(DEVICE)
        s1 = s1.to(DEVICE)
        s2 = s2.to(DEVICE)
        z1_hat, z2_hat = model(z, s1, s2)
        
        order1 = torch.sum(F.mse_loss(z1_hat, z1, reduction="none") + F.mse_loss(z2_hat, z2, reduction="none"), dim=(1, 2, 3))
        order2 = torch.sum(F.mse_loss(z1_hat, z2, reduction="none") + F.mse_loss(z2_hat, z1, reduction="none"), dim=(1, 2, 3))
        loss = torch.sum(torch.minimum(order1, order2))

        order_bool = (order1 <= order2).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        ordered_z1_hat = torch.where(
            order_bool,
            z1_hat,
            z2_hat
        )

        ordered_z2_hat = torch.where(
            order_bool,
            z2_hat,
            z1_hat
        )

        z = z.detach().cpu()
        z1_hat = ordered_z1_hat.detach().cpu()
        z2_hat = ordered_z2_hat.detach().cpu()
        audio1_hat = torch.istft(torch.view_as_complex(z1_hat)**(1/0.3), n_fft=n_fft, win_length=win_length, hop_length=hop_length, onesided=True)
        audio2_hat = torch.istft(torch.view_as_complex(z2_hat)**(1/0.3), n_fft=n_fft, win_length=win_length, hop_length=hop_length, onesided=True)
        audio = torch.istft(torch.view_as_complex(z)**(1/0.3), n_fft=n_fft, win_length=win_length, hop_length=hop_length, onesided=True)

        sdr1 = signal_distortion_ratio(audio1_hat, audio1, load_diag=1e-6)
        sdr2 = signal_distortion_ratio(audio2_hat, audio2, load_diag=1e-6)
        
    return z1_hat, z2_hat, audio, audio1_hat, audio2_hat, sdr1, sdr2, loss

def plot_spectrograms(z, z1, z2):
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    ax1.imshow(z[:, :, 0])
    ax1.set_title("z real")
    ax2.imshow(z[:, :, 1])
    ax2.set_title("z imag")
    ax3.imshow(z1[:, :, 0])
    ax3.set_title("z1 real")
    ax4.imshow(z1[:, :, 1])
    ax4.set_title("z1 imag")
    ax5.imshow(z2[:, :, 0])
    ax5.set_title("z2 real")
    ax6.imshow(z2[:, :, 1])
    ax6.set_title("z2 imag")
    plt.show()
    fig.savefig("spectrograms")


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from resmodel import TwoSpeakerRCPNet
    from synthetic_data import TwoSpeakerData
    import scipy.io.wavfile

    n_fft = 512
    win_length = 300
    hop_length = 150
    dim_f = 257
    dim_t = 295

    test_dataset = TwoSpeakerData("data/test_dataset", n_fft, win_length, hop_length)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
    )
    iterator = iter(test_dataloader)

    for i in range(21):
        _ = next(iterator)

    z, audio1, audio2, z1, z2, s1, s2 = next(iterator)

    model = TwoSpeakerRCPNet(dim_f, dim_t, 8)
    model.load_state_dict(torch.load("rcpnet_epoch3.pt"))
    model = model.to(DEVICE)
    z1_hat, z2_hat, audio1_hat, audio2_hat, audio, sdr1, sdr2, loss = process_audio(
        model=model,
        z=z,
        audio1=audio1,
        audio2=audio2,
        z1=z1,
        z2=z2,
        s1=s1,
        s2=s2,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length
    )


    audio1_hat = audio1_hat[0]
    audio2_hat = audio2_hat[0]
    audio = audio[0]

    audio1_hat[audio1_hat > 32767 / 32768] = 32767 / 32768
    audio1_hat[audio1_hat < -1] = -1
    audio1_hat = (audio1_hat * 32768).type(torch.int16)
    audio1_hat = audio1_hat.numpy()
    scipy.io.wavfile.write("audio1.wav", 14700, audio1_hat)

    audio2_hat[audio2_hat > 32767 / 32768] = 32767 / 32768
    audio2_hat[audio2_hat < -1] = -1
    audio2_hat = (audio2_hat * 32768).type(torch.int16)
    audio2_hat = audio2_hat.numpy()
    scipy.io.wavfile.write("audio2.wav", 14700, audio2_hat)

    audio[audio > 32767 / 32768] = 32767 / 32768
    audio[audio < -1] = -1
    audio = (audio * 32768).type(torch.int16)
    audio = audio.numpy()
    scipy.io.wavfile.write("audio.wav", 14700, audio)

    print(loss)
    print(sdr1)
    print(sdr2)

    plot_spectrograms(z[0], z1_hat[0], z2_hat[0])
    #plt.imsave("example_z2.png", z2[0, :, :, 0])
    