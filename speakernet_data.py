# this file will contain functions that help read and process saved avspeech data for use with speakernet model
# synthetic mixtures will be a crucial part of this
# will contain the custom torch Dataset class to be used with speakernet

import torch
import torchaudio
import numpy as np

def load_to_spectrogram(file_path):
    # load waveform from audio file
    waveform, rate = torchaudio.load("../avspeech_data/GNRPRH-E-sI/audio.wav")

    # resample waveform to 16k kHz -- decrease dimensionality for faster computation
    effects = [
        ["rate", "16k"],
    ]
    waveform, rate = torchaudio.sox_effects.apply_effects_tensor(waveform, rate, effects)

    # perform STFT as discussed in AVSpeech paper
    transform = torchaudio.transforms.Spectrogram(n_fft=512, win_length=400, hop_length=160, power=None)
    spectrogram = transform(waveform[0]) # process only left stereo channel
    
    # seperate compex number into real and imaginary components
    spectrogram = torch.stack([spectrogram.real, spectrogram.imag], dim=2)

    # compression as discussed in AVSpeech paper
    spectrogram = spectrogram ** 0.3
    
    return spectrogram

def find_clean_videos(data_path):
    # iterate through directory to find videos with only one associated face
    pass

if __name__ == "__main__":
    spectrogram = load_to_spectrogram("../avspeech_data/GNRPRH-E-sI")
    print(spectrogram.shape)
    pass

    