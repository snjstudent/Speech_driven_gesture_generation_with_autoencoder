import argparse
import ctypes

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Acoustic signal processing
import scipy.io.wavfile as wav
from pydub import AudioSegment
from python_speech_features import mfcc
import scipy

MFCC_INPUTS = 26  # How many features we will store for each MFCC vector
N_OUTPUT = 384  # Number of gesture features (position)
WINDOW_LENGTH = 50  # in miliseconds
N_CONTEXT = 60
N_INPUT = 26


def average(arr, n):
    """ Replace every "n" values by their average
    Args:
        arr: input array
        n:   number of elements to average on
    Returns:
        resulting array
    """
    end = n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)


def calculate_mfcc(audio_filename):
    """
    Calculate MFCC features for the audio in a given file
    Args:
        audio_filename: file name of the audio
    Returns:
        feature_vectors: MFCC feature vector for the given audio file
    """
    fs, audio = wav.read(audio_filename)

    # Make stereo audio being mono
    if len(audio.shape) == 2:
        audio = (audio[:, 0] + audio[:, 1]) / 2

    # Calculate MFCC feature with the window frame it was designed for
    input_vectors = mfcc(audio, winlen=0.02, winstep=0.01,
                         samplerate=fs, numcep=MFCC_INPUTS)

    input_vectors = [average(input_vectors[:, i], 5)
                     for i in range(MFCC_INPUTS)]

    feature_vectors = np.transpose(input_vectors)

    return feature_vectors


def pad_sequence(input_vectors):
    silence_vectors = calculate_mfcc("data_processing/silence.wav")
    mfcc_empty_vector = silence_vectors[0]

    empty_vectors = np.array([mfcc_empty_vector] * int(N_CONTEXT / 2))
    # append N_CONTEXT/2 "empty" mfcc vectors to past
    new_input_vectors = np.append(empty_vectors, input_vectors, axis=0)
    # append N_CONTEXT/2 "empty" mfcc vectors to future
    new_input_vectors = np.append(new_input_vectors, empty_vectors, axis=0)

    return new_input_vectors


parser = argparse.ArgumentParser()
parser.add_argument('audio_filename')
args = parser.parse_args()
audio_filename = args.audio_filename
input_vectors = calculate_mfcc(audio_filename)
input_with_context = np.array([])
strides = len(input_vectors)
input_vectors = pad_sequence(input_vectors)
for i in range(strides):
    stride = i + int(N_CONTEXT/2)
    if i == 0:
        input_with_context = input_vectors[stride - int(N_CONTEXT/2): stride + int(
            N_CONTEXT/2) + 1].reshape(1, N_CONTEXT+1, N_INPUT)
    else:
        input_with_context = np.append(input_with_context, input_vectors[stride - int(
            N_CONTEXT/2): stride + int(N_CONTEXT/2) + 1].reshape(1, N_CONTEXT+1, N_INPUT), axis=0)

np.save('test.npy', input_with_context)
