import librosa
import soundfile
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def extract_feature(file_name, mfcc, chroma, mel):
    # Open the audio file
    with soundfile.SoundFile(file_name) as sound_file:
        
        # Read the audio data as 32-bit floating point values
        X = sound_file.read(dtype="float32")
        
        # Get the sample rate of the audio
        sample_rate = sound_file.samplerate
        
        # Compute the Short-Time Fourier Transform (STFT) if chroma features are needed
        if chroma:
            stft = np.abs(librosa.stft(X))
        
        # Initialize an empty array to store extracted features
        result = np.array([])
        
        # Extract MFCC features
        if mfcc:
            mfccs = np.mean(
                librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,
                axis=0
            )
            # Add MFCC features to the result
            result = np.hstack((result, mfccs))
        
        # Extract chroma features
        if chroma:
            chroma_feat = np.mean(
                librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,
                axis=0
            )
            # Add chroma features to the result
            result = np.hstack((result, chroma_feat))
        
        # Extract Mel spectrogram features
        if mel:
            mel_feat = np.mean(
                librosa.feature.melspectrogram(y=X, sr=sample_rate).T,
                axis=0
            )
            # Add Mel spectrogram features to the result
            result = np.hstack((result, mel_feat))
    
    # Return all extracted audio features
    return result