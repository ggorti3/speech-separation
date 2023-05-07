import math
import numpy as np
import os
import os.path
import random
import scipy.io.wavfile
from scipy.signal import stft
import torch
from torch.utils.data import Dataset

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
        encoding_stream1 = np.loadtxt(os.path.join(path1, "encoding_stream.csv"), delimiter=",")
        freq1, audio1 = scipy.io.wavfile.read(os.path.join(path1, "audio.wav"))
        # verify audio frequency and length
        assert freq1 == 44100, "{} frequency is not 44100".format(path1)
        assert audio1.shape[0] >= 132300 and encoding_stream1.shape[0] >= 75, "{} is shorter than 3 seconds".format(path1)
        # normalize to float in [0, 1]
        audio1 = audio1.astype(np.float64) / 32768
        # resample to 1/3rd frequency
        audio1 = audio1[::3, 0]
        # uniformly sample random 3 second length
        i = random.randrange(0, ((audio1.shape[0] - 44100) // (588)) + 1)
        audio1 = audio1[588*i:588*i+44100]
        encoding_stream1 = encoding_stream1[i:i+75]
        # perform stft
        _, _, z1 = stft(audio1, nperseg=400, noverlap=200, nfft=512)
        z1 = np.stack([z1.real, z1.imag], axis=2)

        # repeat for second clip
        encoding_stream2 = np.loadtxt(os.path.join(path2, "encoding_stream.csv"), delimiter=",")
        freq2, audio2 = scipy.io.wavfile.read(os.path.join(path2, "audio.wav"))
        assert freq2 == 44100, "{} frequency is not 44100".format(path2)
        assert audio2.shape[0] >= 132300 and encoding_stream2.shape[0] >= 75, "{} is shorter than 3 seconds".format(path2)
        audio2 = audio2.astype(np.float64) / 32768
        audio2 = audio2[::3, 0]
        i = random.randrange(0, ((audio2.shape[0] - 44100) // (588)) + 1)
        audio2 = audio2[588*i:588*i+44100]
        encoding_stream2 = encoding_stream2[i:i+75]
        _, _, z2 = stft(audio2, nperseg=400, noverlap=200, nfft=512)
        z2 = np.stack([z2.real, z2.imag], axis=2)

        z1 = torch.tensor(z1, dtype=torch.double)
        z2 = torch.tensor(z2, dtype=torch.double)
        z = z1 + z2
        s1 = torch.tensor(encoding_stream1, dtype=torch.double)
        s2 = torch.tensor(encoding_stream2, dtype=torch.double)

        return z, z1, z2, s1, s2

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

# if __name__ == "__main__":
#     from scipy.signal import istft

#     dataset = TwoSpeakerData("../avspeech_data/")
#     iterator = iter(dataset)
#     for i in range(119):
#         next(iterator)
#     z, _, _ = next(iterator)
#     z = z[: ,: ,0].astype(np.complex128) + 1j * z[: ,: ,1].astype(np.complex128)
#     _, audio = istft(z, nperseg=400, noverlap=200, nfft=512)
#     audio = (audio * 32768).astype(np.int16)
#     scipy.io.wavfile.write("sample.wav", 14700, audio)
