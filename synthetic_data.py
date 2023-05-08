import math
import numpy as np
import os
import os.path
import random
import scipy.io.wavfile
import torch
from torch import stft, hann_window
from torch.utils.data import Dataset

torch.set_default_dtype(torch.float)

class TwoSpeakerData(Dataset):
    def __init__(self, dataset_path):
        self.paths = []
        for root, dirs, _ in os.walk(dataset_path, topdown=False):
            for name in dirs:
                self.paths.append(os.path.join(root, name))
    
    def __len__(self):
        return (len(self.paths) * (len(self.paths) - 1)) // 2
    
    def __getitem__(self, idx):
        val = (math.sqrt(1 + 8 * idx) - 1) / 2
        i = math.floor(val + 1e-8) + 1
        j = idx - (i * (i - 1)) // 2

        path1 = self.paths[i]
        path2 = self.paths[j]

        # load encodings and frequency
        encoding_stream1 = np.loadtxt(os.path.join(path1, "encoding_stream.csv"), delimiter=",", dtype=np.float32)
        freq1, audio1 = scipy.io.wavfile.read(os.path.join(path1, "audio.wav"))
        # verify audio frequency and length
        assert freq1 == 44100, "{} frequency is not 44100".format(path1)
        assert audio1.shape[0] >= 132300 and encoding_stream1.shape[0] >= 75, "{} is shorter than 3 seconds".format(path1)
        # normalize to float in [0, 1]
        audio1 = audio1.astype(np.float32) / 32768
        # resample to 1/3rd frequency
        audio1 = audio1[::3]
        if len(audio1.shape) == 2:
            audio1 = audio1[:, 0]
        # uniformly sample random 3 second length
        i = random.randrange(0, ((audio1.shape[0] - 44100) // (588)) + 1)
        audio1 = audio1[588*i:588*i+44100]
        encoding_stream1 = encoding_stream1[i:i+75]
        # perform stft
        #_, _, z1 = stft(audio1, nperseg=400, noverlap=200, nfft=512)
        audio1 = torch.tensor(audio1)
        z1 = stft(
            audio1,
            n_fft=512,
            win_length=300,
            hop_length=150,
            window=hann_window(300),
            onesided=True,
            return_complex=True
        )
        z1 = torch.view_as_real(z1)

        # repeat for second clip
        encoding_stream2 = np.loadtxt(os.path.join(path2, "encoding_stream.csv"), delimiter=",", dtype=np.float32)
        freq2, audio2 = scipy.io.wavfile.read(os.path.join(path2, "audio.wav"))
        assert freq2 == 44100, "{} frequency is not 44100".format(path2)
        assert audio2.shape[0] >= 132300 and encoding_stream2.shape[0] >= 75, "{} is shorter than 3 seconds".format(path2)
        audio2 = audio2.astype(np.float32) / 32768
        audio2 = audio2[::3]
        if len(audio2.shape) == 2:
            audio2 = audio2[:, 0]
        i = random.randrange(0, ((audio2.shape[0] - 44100) // (588)) + 1)
        audio2 = audio2[588*i:588*i+44100]
        encoding_stream2 = encoding_stream2[i:i+75]
        #_, _, z2 = stft(audio2, nperseg=400, noverlap=200, nfft=512)
        audio2 = torch.tensor(audio2)
        z2 = stft(
            audio2,
            n_fft=512,
            win_length=300,
            hop_length=150,
            window=hann_window(300),
            onesided=True,
            return_complex=True
        )
        z2 = torch.view_as_real(z2)

        z = z1 + z2
        s1 = torch.tensor(encoding_stream1)
        s2 = torch.tensor(encoding_stream2)

        return z, audio1, audio2, z1, z2, s1, s2

    def collate_fn(self, items):
        var_lists = [[] for i in range(len(items[0]))]
        for item in items:
            for i, var in enumerate(item):
                var_lists[i].append(var)
        
        var_tup = tuple()
        for var_list in var_lists:
            var = torch.stack(var_list, dim=0)
            var_tup = var_tup + (var,)
            
        return var_tup

if __name__ == "__main__":
    from torch import istft

    dataset = TwoSpeakerData("../avspeech_data/")
    iterator = iter(dataset)
    for i in range(0):
        next(iterator)
    z, _, _, _, _, _, _ = next(iterator)
    z = torch.complex(z[:, :, 0], z[:, :, 1])
    audio = istft(z, n_fft=512, win_length=300, hop_length=150, onesided=True)
    audio = (audio * 32768).type(torch.int16)
    audio = audio.numpy()
    print(z.shape)
    print(audio.shape)
    scipy.io.wavfile.write("sample.wav", 14700, audio)
