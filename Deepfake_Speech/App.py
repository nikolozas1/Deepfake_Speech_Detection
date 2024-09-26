import tkinter as tk
from tkinter import filedialog
from tensorflow.keras import models
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
import librosa
import csv
import pandas as pd
import numpy as np
scaler = StandardScaler()
dir = str(Path(__file__).resolve().parent)
model = models.load_model(dir+"/Models/deepfake_label_v1.h5")
scaler.fit(pd.read_csv(dir+"/DATASET-balanced-binarized.csv").iloc[:, :-1].values)

def audio_prediction(x):
    y, sr = librosa.load(x)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr)).round(6)
    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)).round(6)
    rms = np.mean(librosa.feature.rms(y=y)).round(6)
    bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)).round(6)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)).round(6)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y)).round(6)
    for i in range(20):
        globals()[f'mfcc{i+1}'] = np.mean(mfcc[i]).round(6)
    header = ['chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth',
       'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4',
       'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11',
       'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18',
       'mfcc19', 'mfcc20']
    values = [chroma, rms, cent, bw, rolloff, zcr,mfcc1, mfcc2, mfcc3, mfcc4,
       mfcc5, mfcc6, mfcc7, mfcc8, mfcc9, mfcc10, mfcc11,
       mfcc12, mfcc13, mfcc14, mfcc15, mfcc16, mfcc17, mfcc18,
       mfcc19, mfcc20]
    with open("temp.csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow(values)
    temp = pd.read_csv("temp.csv")
    x = model.predict(scaler.transform(temp.values))
    ans = (x[0][0]*100).round(1)
    os.remove("temp.csv")
    return ans
    
def label_change(label_prediction,label_color):
    result_label.configure(text=str(label_prediction)+"% Authenticity", bg=label_color)
    browse_button.configure(bg=label_color)
    text_title.configure(fg=label_color)

def browse_audio():
    file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav *.flac *.ogg")])
    if file_path:
        file = audio_path_label.configure(text="" + file_path)
        predictions = audio_prediction(file_path)
        if predictions > 50:
            label_change(predictions,"#00ff00")
        elif predictions < 50:
            label_change(predictions,"#dc143c")
        else:
            label_change(predictions, "#c0c0c0")
        
        

root = tk.Tk()
root.title("Audio Authenticity App")


root.geometry("500x150")


text_title = tk.Label(root, text=" 🔊 Audio Authenticity App 🔊 ", font=("Helvetica", 13, 'bold'), fg="#989898")
text_title.pack(pady=5, padx=0)


browse_button = tk.Button(root, text="Browse Audio", command=browse_audio, bg="#d3d3d3", fg="black")
browse_button.pack(pady=2)


text_desc = tk.Label(root, text="Acceptable audio formats: *.mp3 *.wav *.flac *.ogg", font=("Helvetica", 10))
text_desc.pack(pady=0, padx=0)

audio_path_label = tk.Label(root)
audio_path_label.pack(pady=0)

result_label = tk.Label(root, text="Choose an Audio File", font=("Helvetica", 12, 'bold'), bg="#c0c0c0", fg="black", padx=500, pady=0)
result_label.pack(pady=2)

root.mainloop()