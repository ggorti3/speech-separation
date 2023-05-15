import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional.audio import signal_distortion_ratio
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device {}".format(DEVICE))

def two_speaker_train(model, train_dataloader, val_dataloader, epochs, lr, n_fft, win_length, hop_length, save_path):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        cum_loss = 0
        model.train()
        for i, (z, _, _, z1, z2, s1, s2) in tqdm(enumerate(train_dataloader)):
            z = z.to(DEVICE)
            z1 = z1.to(DEVICE)
            z2 = z2.to(DEVICE)
            s1 = s1.to(DEVICE)
            s2 = s2.to(DEVICE)
            z1_hat, z2_hat = model(z, s1, s2)

            loss = (F.mse_loss(z1_hat, z1, reduction="sum") + F.mse_loss(z2_hat, z2, reduction="sum")) / z.shape[0]
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                cum_loss += loss

                if i % 50000 == 0:
                    print("Train Running Loss: {}".format(cum_loss / (i + 1)))

        with torch.no_grad():
            print("Epoch {}".format(e))
            print("    Train Running Loss: {}".format(cum_loss / len(train_dataloader)))
            avg_sdr = two_speaker_evaluate(model, val_dataloader, n_fft, win_length, hop_length)
            print("    Val Average sdr: {}".format(avg_sdr))

            torch.save(model.state_dict(), save_path + "rcpnet_epoch{}.pt".format(e))

def two_speaker_evaluate(model, val_dataloader, n_fft, win_length, hop_length):
    model.eval()
    cum_sdr = 0
    n_samples = 0
    for i, (z, audio1, audio2, _, _, s1, s2) in tqdm(enumerate(val_dataloader)):
        z = z.to(DEVICE)
        s1 = s1.to(DEVICE)
        s2 = s2.to(DEVICE)
        z1_hat, z2_hat = model(z, s1, s2)

        z1_hat = z1_hat.detach().cpu()
        z2_hat = z2_hat.detach().cpu()
        audio1_hat = torch.istft(torch.view_as_complex(z1_hat), n_fft=n_fft, win_length=win_length, hop_length=hop_length, onesided=True)
        audio2_hat = torch.istft(torch.view_as_complex(z2_hat), n_fft=n_fft, win_length=win_length, hop_length=hop_length, onesided=True)

        sdr1 = signal_distortion_ratio(audio1_hat, audio1, load_diag=1e-6)
        sdr2 = signal_distortion_ratio(audio2_hat, audio2, load_diag=1e-6)

        n_samples += sdr1.shape[0] + sdr2.shape[0]
        cum_sdr += (torch.sum(sdr1) + torch.sum(sdr2)).item()
    avg_sdr = (cum_sdr / n_samples) if n_samples > 0 else 0
    return avg_sdr

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from resmodel import TwoSpeakerRCPNet
    from synthetic_data import TwoSpeakerData

    lr = 3e-5
    epochs = 5
    batch_size = 50

    n_fft = 512
    win_length = 300
    hop_length = 150
    dim_f = 257
    dim_t = 295

    train_dataset = TwoSpeakerData("data/train_dataset", n_fft, win_length, hop_length)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_dataset = TwoSpeakerData("data/val_dataset", n_fft, win_length, hop_length)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )

    model = TwoSpeakerRCPNet(dim_f, dim_t)
    model = model.to(DEVICE)
    two_speaker_train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        lr=lr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        save_path = "./"
    )
    


        