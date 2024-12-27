import pickle
import tkinter as tk
import wave
from tkinter import ttk

import librosa
import numpy as np
import pyaudio
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow import keras
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

class AudioClassifierApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Audio Classifier")
        self.root.geometry("680x400")

        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()
        with open('Y.pkl', 'rb') as file:
            y = pickle.load(file)
            # Fit the encoder on the training data
            Y = self.encoder.fit_transform(np.array(y).reshape(-1, 1)).toarray()

        # Load x_train from the file
        with open('x_train.pkl', 'rb') as file:
            x_train = pickle.load(file)
            # Fit the scaler on the training data
            self.scaler.fit(x_train)

        # Use a predefined theme
        style = ttk.Style()
        style.theme_use("clam")

        # Add a background image
        self.background_img = tk.PhotoImage(file="background.png")
        background_label = tk.Label(root, image=self.background_img)
        background_label.place(relwidth=1, relheight=1)

        # Create a container frame for organization
        container = ttk.Frame(root)
        container.place(relx=0.5, rely=0.5, anchor="center")

        self.start_record_button = tk.Button(container, text="Start Recording", command=self.start_recording)
        self.start_record_button.grid(row=0, column=0, padx=10, pady=10)

        self.stop_record_button = tk.Button(container, text="Stop Recording", command=self.stop_recording,
                                            state=tk.DISABLED)
        self.stop_record_button.grid(row=1, column=0, padx=10, pady=10)

        self.predict_button = tk.Button(container, text="Predict", command=self.predict)
        self.predict_button.grid(row=2, column=0, padx=10, pady=10)

        self.audio_path = "recorded_audio.wav"
        self.frames = []
        self.recording = None
        self.stream = None
        self.p = pyaudio.PyAudio()

    # Function to start recording
    def start_recording(self):
        if self.audio_path:
            self.start_record_button.config(state=tk.DISABLED)
            self.stop_record_button.config(state=tk.NORMAL)
            self.predict_button.config(state=tk.DISABLED)

            self.stream = self.p.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=44100,
                                      input=True,
                                      frames_per_buffer=1024)

            self.recording = True
            self.frames = []

            self.root.after(100, self.update_audio_stream)

    def update_audio_stream(self):
        if self.recording:
            data = self.stream.read(1024)
            self.frames.append(data)
            self.root.after(10, self.update_audio_stream)

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

            wf = wave.open(self.audio_path, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(self.frames))
            wf.close()

            self.start_record_button.config(state=tk.NORMAL)
            self.stop_record_button.config(state=tk.DISABLED)
            self.predict_button.config(state=tk.NORMAL)

    # Calculate the threshold rate of audio data - returns a 1-dimensional array
    def zcr(self, data, frame_length, hop_length):
        zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
        return np.squeeze(zcr)

    # function to calculate root mean square error
    def rmse(self, data, frame_length=2048, hop_length=512):
        rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
        return np.squeeze(rmse)

    # mfcc
    def mfcc(self, data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
        mfcc = librosa.feature.mfcc(y=data, sr=sr)
        return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

    '''
    feature extraction This function combines ZCR, RMSE, and MFCC features
    to create a complete feature vector for the audio data.
    '''

    def extract_features(self, data, sr=22050, frame_length=2048, hop_length=512):
        result = np.array([])

        result = np.hstack((result,
                            self.zcr(data, frame_length, hop_length),
                            self.rmse(data, frame_length, hop_length),
                            self.mfcc(data, sr, frame_length, hop_length)
                            ))
        return result

    # NOISE
    def noise(self, data):
        noise_amp = 0.045 * np.random.uniform() * np.amax(data)
        data = data + noise_amp * np.random.normal(size=data.shape[0])
        return data

    # STRETCH
    def stretch(self, data, rate=0.8):
        return librosa.effects.time_stretch(data, rate=rate)

    # SHIFT
    def shift(self, data):
        shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
        return np.roll(data, shift_range)

    # PITCH
    def pitch(self, data, sampling_rate, pitch_factor=0.7):
        return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

    def get_features(self, path, duration=2.5, offset=0.6):
        data, sr = librosa.load(path=path, duration=duration, offset=offset)
        aud = self.extract_features(data)  # feature extraction
        audio = np.array(aud)  # convert features to array

        noised_audio = self.noise(data)
        aud2 = self.extract_features(noised_audio)
        audio = np.vstack((audio, aud2))

        pitched_audio = self.pitch(data, sr)
        aud3 = self.extract_features(pitched_audio)
        audio = np.vstack((audio, aud3))

        pitched_audio1 = self.pitch(data, sr)
        pitched_noised_audio = self.noise(pitched_audio1)
        aud4 = self.extract_features(pitched_noised_audio)
        audio = np.vstack((audio, aud4))

        return audio

    def predict(self):
        # Make prediction using the loaded model and extracted features
        print("Predicting...")
        audio_path = 'recorded_audio.wav'
        self.model = keras.models.load_model('cnn.h5')
        if audio_path and hasattr(self, 'model'):
            features = self.get_features(audio_path)
            standardized_input_feature = self.scaler.transform(features)
            expanded_input_feature = np.expand_dims(standardized_input_feature, axis=2)
            predictions = self.model.predict(expanded_input_feature)
            predicted_label = self.encoder.inverse_transform(predictions)[0][0]
            print("Predicted label:", predicted_label)


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioClassifierApp(root)
    root.mainloop()
