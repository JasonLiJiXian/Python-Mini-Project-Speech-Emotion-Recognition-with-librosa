import librosa
import soundfile
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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

# Map emotion codes in the file names to emotion labels
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Select the emotions that will be used in the model
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']


def load_data(test_size=0.25):
    # Initialize empty lists to store features and emotion labels
    x, y = [], []

    # Find all WAV audio files in the actor folders
    for file in glob.glob("Audio_Speech_Actors_01-24/Actor_*/*.wav"):
        
        # Get the file name from the full file path
        file_name = os.path.basename(file)
        
        # Extract the emotion code from the file name
        # and convert it to the corresponding emotion label
        emotion = emotions[file_name.split("-")[2]]

        # Skip emotions that are not included in observed_emotions
        if emotion not in observed_emotions:
            continue

        # Extract MFCC, chroma, and Mel spectrogram features
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        
        # Store the extracted features
        x.append(feature)
        
        # Store the corresponding emotion label
        y.append(emotion)

    # Split the dataset into training and testing sets
    return train_test_split(
        np.array(x), y,
        test_size=test_size,
        random_state=9
    )


# 1. Load the data and split it into training and testing sets
x_train, x_test, y_train, y_test = load_data(test_size=0.25)

# 2. Check the dataset size
print("Training/Testing samples:", (x_train.shape[0], x_test.shape[0]))
print("Number of features:", x_train.shape[1])

# 3. Initialize the MLP model using the tutorial parameters
model = MLPClassifier(
    alpha=0.01,
    batch_size=256,
    epsilon=1e-08,
    hidden_layer_sizes=(300,),
    learning_rate='adaptive',
    max_iter=500
)

# 4. Train the model
model.fit(x_train, y_train)

# 5. Make predictions on the test set
y_pred = model.predict(x_test)

# 6. Calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))