
import pandas as pd
import glob
import os 
import librosa
import numpy as np
from scipy.signal import butter, sosfilt
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as tf_hub


# finding all annotation files 
csv_files = glob.glob("*.csv")

# loading master annotation file
metadata = pd.read_csv("~/Documents/project/Annotations.csv")
test_metadata = metadata.iloc[196104:196118]
print(test_metadata)
# setting target as column iwth pod info
target = ["pod"]

# making empty set for columns of data set
annotation_summary = {}

pod_data_inventory = {}

for file_path in csv_files:
    file_name = os.path.basename(file_path)
    # reading first few rows
    df_heads = pd.read_csv(file_path, nrows = 5)
    # retreiving columns in dataset
    columns = list(df_heads.columns)

    targets_found = [ col for col in columns if col in target]
    # counting number of rows in datasets with a pod column
    if targets_found:
        df_full = pd.read_csv(file_path)
        total_rows = len(df_full)

        # counting number of rows with pod input 
        col_counts = {}
        for col in targets_found:
            valid_pods = df_full[col].notna().sum()
            unique_pods = set(df_full[col].dropna().astype(str))

            col_counts[col] = { 
                "valid_annotations": int(valid_pods),
                "total_rows": int(total_rows),
                "unique_values": unique_pods
            }
        pod_data_inventory[file_name] = col_counts

    annotation_summary[file_path] = columns

    print(f" FILE: {file_path}")
    print(f" COLUMNS: {columns}")
    print("-" * 50)
    
#total_pod_counts = 0

#for file_name, metrics in pod_data_inventory.items():
    # print name of dataset we are currently looking at
#    print("File: ", file_name)

 #   for col_name, counts in metrics.items():
 #       print( col_name, "has", counts["valid_annotations"],
 #         "labels out of",
  #        counts["total_rows"])
   #     print("     Unique pods:", counts["unique_values"])

    #total_pod_counts = total_pod_counts + counts["valid_annotations"]

    #print("------------------------------------")
#print("Total usable rows for validation:", total_pod_counts)



# defining function to link file name to row in annotation table 
def link_audio_to_metadata(audio_file_path, metadata_df):
    # checking that audio file exists 
    if not os.path.exists(audio_file_path):
        return{"status": "ERROR", "data": None}

    # extracting base name 
    base_name = os.path.basename(audio_file_path)

    # checking if there is a row that matches an audio file name 
    matched_rows = metadata_df[
        metadata_df["filename"].apply(lambda x: str(x) in base_name)]
    
    # returning errors if there isn't a match 
    if matched_rows.empty:
        return{"status": "ERROR", "data": None}
    
    # extracting row data if it is a clean match 
    metadata_info = matched_rows.iloc[0].to_dict()

    return{"status": "SUCCESS", "data": metadata_info}

# loading sample metadata as pandas dataframe
smru_metadata = pd.read_csv("smru_annot2.csv")

# loading sample audio data 
test_audio = "dclde_2027_dclde_2027_killer_whales_smru_audio_lime-kiln_LK_20190705_142000_000.wav"

test_result = link_audio_to_metadata(test_audio, smru_metadata)

if test_result["status"] == "SUCCESS":
    print(f" File found in metadata: {test_audio}")
    file_metadata = test_result["data"]
    print(f" Community of test audio: {file_metadata.get('kw_ecotype')}")

else:
    print("Could not find metadata")

# defining resampling and bandpassing function

def resample_and_bandpass (audio_file_path, target_sr = 32000):

    #retrieving original sampling rate of audio
    original_sr = librosa.get_samplerate(audio_file_path)

    nyquist_target = 0.5 * target_sr

    # setting up preprocessing for audio with low sample rate
    if original_sr < target_sr:
        # loading low SR audio
        audio_array, sr = librosa.load(audio_file_path, sr = None)
        original_nyquist = 0.5 * original_sr

        # defining lower limit for bandpass filter
        low = 500/original_nyquist

        # defining upper limit for bandpass filter 
        high = (original_nyquist-100)/original_nyquist

        # creating butterworth filter
        sos = butter(N=3, Wn=[low, high], btype="bandpass", output="sos")
        # applying butterworth filter 
        filtered_original = sosfilt(sos, audio_array)

        # upsampling 
        preproc_audio = librosa.resample(
            filtered_original, orig_sr = original_sr, target_sr = target_sr)
        
        # setting up preprocessing for audio with high SR
    else:
        resampled_audio, sr = librosa.load(audio_file_path, sr = target_sr)

        # defining lower limit for bandpass 
        low = 500 / nyquist_target
        # defining higher limit for bandpass
        high = 15000 / nyquist_target
        
        # creating butterworth filter 
        sos = butter(N=3, Wn=[low, high], btype="bandpass", output="sos")
        # applying butterworth filter 
        final_audio = sosfilt(sos, resampled_audio)

        return{
            "audio_data": final_audio,
            "status": "SUCCESS",
            "original_sr": original_sr,
            "final_sr": target_sr
        }
    

# Testing preprocessing function 

test_audio_path = "/Users/anyasen/Documents/project/dclde_2027_dclde_2027_killer_whales_smru_audio_lime-kiln_LK_20190705_142000_000.wav"

result = resample_and_bandpass(test_audio_path, target_sr=32000)

if result["status"] == "SUCCESS":
    clean_audio_array = result["audio_data"]

    print(f"Succesfully resampled audio from {result['original_sr']} Hz to {result['final_sr']} Hz")


    # saving as audio file to make sure vocalisation weren't cut out 
    export_path = (
        "/Users/anyasen/Documents/project/test_file_preproc.wav"
    )
    sf.write(export_path, clean_audio_array, 32000)

    # Cutting audio into 5s clips 
    target_sr = 32000
    clip_length = 5 * target_sr

    # looping through rows of test metadata (each call in the entire audio clip)
    for idx, row in test_metadata.iterrows():

        start_time_sec = row["FileBeginSec"]
        end_time_sec = row["FileEndSec"]
        call_label = row["AnnotationLevel"]
        whale_ecotype = row["Ecotype"]
        
        # calculating duration of call
        call_duration = end_time_sec - start_time_sec

        # determining bounds for 5s clip
        if call_duration <= 5.0:
            midpoint = start_time_sec + (call_duration/2.0)
            start_sec = max(0.0, midpoint - 2.5)
        else:
            start_sec - start_time_sec
        
        # converting time coordinates to indices on 32kHz grid
        first_cut = int(start_sec * target_sr)
        second_cut = first_cut + clip_length

        call_clip = clean_audio_array[first_cut:second_cut]

        if len(call_clip) < clip_length:
            padding = (clip_length - len(call_clip))/2
            call_clip = np.pad(call_clip, (padding, padding), "constant")

        export_filename = f"clip_{idx}.wav"
        export_path = f"/Users/anyasen/Documents/project/audio_tests/{export_filename}"
        sf.write(export_path, call_clip, 32000)

else: 
    print("Pipeline failed")






