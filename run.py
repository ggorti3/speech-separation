import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def two_speaker_train(model, dataloader, epochs, lr):

    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for i, (z, z1, z2, s1, s2) in enumerate(dataloader):
        z = z.to(DEVICE)
        z1 = z1.to(DEVICE)
        z2 = z2.to(DEVICE)
        s1 = s1.to(DEVICE)
        s2 = s2.to(DEVICE)
        z1_hat, z2_hat = model(z, s1, s2)

        loss = F.mse_loss(z1_hat, z1, reduction="sum") + F.mse_loss(z2_hat, z2, reduction="sum")
        loss.backward()
        optimizer.step()

        print(loss)

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from model import TwoSpeakerCPNet
    from synthetic_data import TwoSpeakerData

    dataset = TwoSpeakerData("../avspeech_data/")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )

    lr = 3e-5
    epochs = 1

    model = TwoSpeakerCPNet()
    two_speaker_train(
        model=model,
        dataloader=dataloader,
        epochs=epochs,
        lr=lr
    )
    


        