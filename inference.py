import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional.audio import signal_distortion_ratio
from tqdm import tqdm

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
        loss = (F.mse_loss(z1_hat, z1, reduction="sum") + F.mse_loss(z2_hat, z2, reduction="sum")) / z.shape[0]
        z1_hat = z1_hat.detach().cpu()
        z2_hat = z2_hat.detach().cpu()
        audio1_hat = torch.istft(torch.view_as_complex(z1_hat), n_fft=n_fft, win_length=win_length, hop_length=hop_length, onesided=True)
        audio2_hat = torch.istft(torch.view_as_complex(z2_hat), n_fft=n_fft, win_length=win_length, hop_length=hop_length, onesided=True)

        sdr1 = signal_distortion_ratio(audio1_hat, audio1, load_diag=1e-6)
        sdr2 = signal_distortion_ratio(audio2_hat, audio2, load_diag=1e-6)
    return audio1_hat, audio2_hat, sdr1, sdr2, loss


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

    val_dataset = TwoSpeakerData("../avspeech_data/", n_fft, win_length, hop_length)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )
    iterator = iter(val_dataloader)

    z, audio1, audio2, z1, z2, s1, s2 = next(iterator)

    model = TwoSpeakerRCPNet(dim_f, dim_t)
    #model.load_state_dict(torch.load("rcpnet_epoch4.pt"))
    model = model.to(DEVICE)
    audio1_hat, audio2_hat, sdr1, sdr2, loss = process_audio(
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
    