import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional.audio import signal_distortion_ratio
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device {}".format(DEVICE))

def loss_func(z1_hat, z2_hat, z1, z2):
    return torch.sum(torch.maximum(torch.sum(F.mse_loss(z1_hat, z1, reduction="none"), dim=(1, 2, 3)), torch.sum(F.mse_loss(z2_hat, z2, reduction="none"), dim=(1, 2, 3))))
    #return F.mse_loss(z1_hat, z1, reduction="sum") + F.mse_loss(z2_hat, z2, reduction="sum")

def two_speaker_train(model, train_dataloader, val_dataloader, epochs, lr, n_fft, win_length, hop_length, save_path):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.5)

    for e in range(epochs):
        cum_loss = 0
        n_samples = 0
        model.train()
        for i, (z, _, _, z1, z2, s1, s2) in tqdm(enumerate(train_dataloader)):
            z = z.to(DEVICE)
            z1 = z1.to(DEVICE)
            z2 = z2.to(DEVICE)
            s1 = s1.to(DEVICE)
            s2 = s2.to(DEVICE)
            z1_hat, z2_hat = model(z, s1, s2)

            loss = loss_func(z1_hat, z2_hat, z1, z2)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                cum_loss += loss
                n_samples += z.shape[0]

                if i % 1000 == 0:
                    print("Train Running Loss: {}".format(cum_loss / n_samples))

        with torch.no_grad():
            print("Epoch {}".format(e))
            print("    Train Avg Loss: {}".format(cum_loss / n_samples))
            avg_sdr, median_sdr, avg_loss = two_speaker_evaluate(model, val_dataloader, n_fft, win_length, hop_length)
            print("    Val Avg Loss: {}".format(avg_loss))
            print("    Val Avg sdr: {}".format(avg_sdr))
            print("    Val Median sdr: {}".format(median_sdr))

            torch.save(model.state_dict(), save_path + "rcpnet_epoch{}.pt".format(e))
        
        scheduler.step()

def two_speaker_evaluate(model, val_dataloader, n_fft, win_length, hop_length):
    model.eval()
    cum_sdr = 0
    cum_loss = 0
    n_samples = 0
    sdrs = torch.zeros((0,))
    for i, (z, audio1, audio2, z1, z2, s1, s2) in tqdm(enumerate(val_dataloader)):
        with torch.no_grad():
            z = z.to(DEVICE)
            z1 = z1.to(DEVICE)
            z2 = z2.to(DEVICE)
            s1 = s1.to(DEVICE)
            s2 = s2.to(DEVICE)
            z1_hat, z2_hat = model(z, s1, s2)

            cum_loss += loss_func(z1_hat, z2_hat, z1, z2)

            z1_hat = z1_hat.detach().cpu()
            z2_hat = z2_hat.detach().cpu()
            audio1_hat = torch.istft(torch.view_as_complex(z1_hat)**(1/0.3), n_fft=n_fft, win_length=win_length, hop_length=hop_length, onesided=True)
            audio2_hat = torch.istft(torch.view_as_complex(z2_hat)**(1/0.3), n_fft=n_fft, win_length=win_length, hop_length=hop_length, onesided=True)

            sdr1 = signal_distortion_ratio(audio1_hat, audio1, load_diag=1e-6)
            sdr2 = signal_distortion_ratio(audio2_hat, audio2, load_diag=1e-6)

            sdrs = torch.cat([sdrs, sdr1, sdr2])
            n_samples += z.shape[0]
    avg_sdr = torch.mean(sdrs)
    median_sdr = torch.median(sdrs)
    avg_loss = cum_loss / n_samples
    return avg_sdr, median_sdr, avg_loss

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from resmodel import TwoSpeakerRCPNet
    from synthetic_data import TwoSpeakerData

    lr = 5e-7
    epochs = 5
    batch_size = 28

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

    model = TwoSpeakerRCPNet(dim_f, dim_t, 8)
    #model.load_state_dict(torch.load("rcpnet_epoch0.pt"))
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
    # avg_sdr, median_sdr, avg_loss = two_speaker_evaluate(
    #     model,
    #     val_dataloader,
    #     n_fft,
    #     win_length,
    #     hop_length
    # )
    # print("Val Avg Loss: {}".format(avg_loss))
    # print("Val Avg sdr: {}".format(avg_sdr))
    # print("Val Median sdr: {}".format(median_sdr))
    


        