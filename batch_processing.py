# imports
import pandas as pd
import glob
import os 
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as tf_hub
import onnxruntime as ort
import urllib.request

# Setting up directories
base_dir = os.path.abspath(os.path.dirname(__file__)) if '__file__' in locals() else os.getcwd()
input_dir = os.path.join(base_dir, "cleaned_audio")
output_dir = os.path.join(base_dir, "features")

os.makedirs(output_dir, exist_ok = True)
output_file = os.path.join(base_dir, "perch_embeddings.csv")
annotations_path = os.path.join(base_dir, "Annotations.csv")

# Downloading optimised perch model
model_path = os.path.join(base_dir, "perch_v2.onnx")
if not os.path.exists(model_path):
    url = "https://huggingface.co/justinchuby/Perch-onnx/resolve/main/perch_v2.onnx"
    
    # monitoring progress of download  
    def progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownloading: {min(percent, 100)}%", end="")
    
    urllib.request.urlretrieve(url, model_path)
    print("Download complete")

# loading model
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

# finding all .wav files in the clip directory 
clip_files = glob.glob(os.path.join(input_dir, "*.wav"))
# counting number of clips
print(f"Found {len(clip_files)} audio clips")
# checking names of clips
print("File names found:", [os.path.basename(f) for f in clip_files])

# creating empty list for embeddings data
all_embeddings = []

# creating extraction loop - creating index for files
for idx, file_path in enumerate(clip_files):
    # finding file name 
    file_name = os.path.basename(file_path)
    # reading audio 
    audio_array, sr = sf.read(file_path) 

    # padding/cutting audio array for perch onnx
    if len(audio_array) < 160000:
        audio_array = np.pad(audio_array, (0, 160000 - len(audio_array)), 'constant')
    elif len(audio_array) > 160000:
        audio_array = audio_array[:160000]
    # changing shape of data to be compatible with tensor flow
    audio_tensor = np.expand_dims(audio_array, axis = 0).astype(np.float32)

    # running the model
    model_outputs = session.run(None, {input_name: audio_tensor})
    # extract vector 
    embedding_vectors = model_outputs[0].flatten()

    # creating dictionary with filnames
    summary_dict = {"Filename": file_name}

    # looping through each value in a vector - creating columns for each value in the vector and adding to dictionary
    for i, val in enumerate(embedding_vectors):
        summary_dict[f"vector{i}"] = val 
    # append all embedding rows to dictionary 
    all_embeddings.append(summary_dict)
    
# saving dictionary as dataframe
embeddings_df = pd.DataFrame(all_embeddings)

# merging embeddings with subpopulation data
if os.path.exists(annotations_path):
    # reading annotions file to extract subpopulation data  
    df_annotations = pd.read(annotations_path)
    # identifying columns with filename and subpopulation data 
    df_labels = df_annotations['SoundFile', 'Ecotype']

    embeddings_df = pd.merge(embeddings_df, 
                    df_labels,
                    left_on = 'Filename', 
                    right_on = 'SoundFile',
                    how = 'inner')
    embeddings_df = embeddings_df.drop(columns = ['SoundFile'])

    print(f"Matched and added subpopulation labels")
else:
    print(f"Could not find annotation file.")

# saving dataframe as .csv file for further inspection
embeddings_df.to_csv(output_dir, index= False)
print(f"Features saved to {output_dir}")
