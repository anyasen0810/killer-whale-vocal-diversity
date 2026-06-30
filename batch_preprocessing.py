# imports
import pandas as pd
import glob
import os 
import librosa
import numpy as np
from scipy.signal import butter, sosfilt
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as tf_hub

# setup 
target_sr = 32000
clip_length = 5
clip_points = clip_length * target_sr 

# defining directories
## using relative paths so that it works on mac and windows
base_dir = os.path.abspath(os.path.dirname(__file__)) if '__file__' in locals() else os.getcwd()
data_dir = os.path.join(base_dir, "data")
output_dir = os.path.join(base_dir, "cleaned_audio")

# finding .wav files 
wav_files = glob.glob(os.path.join(data_dir, "*.wav"))
annotations_path = os.path.join(base_dir, "Annotations.csv")                      
# loading master annotation file
if os.path.exists(annotations_path):
    metadata = pd.read_csv(annotations_path)
else:
    raise FileNotFoundError(f"Could not find annotation file")


# Defining functions 

# function that links audio to metadata
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
    
  ##############################  


for file in wav_files:

    # linking audio data to metadata
    link_result = link_audio_to_metadata(file, metadata)

    if link_result["status"] != "SUCCESS":
        print(f"Could not find annotation row for file")
        continue
    file_annotation = link_result["data"]

    # resampling and bandpassing 
    resample_result = resample_and_bandpass(file, target_sr = target_sr)
    if resample_result["status"] != "SUCCESS":
        print(f"  Failed to resample and/or bandpass")
        continue

    # extracting array from dictionary response 
    clean_audio_array = resample_result["audio_data"]

    # looping through calls in file to slice them into 5s clips
    for idx, row in metadata.iterrows():
            start_time_sec = row["FileBeginSec"]
            end_time_sec = row["FileEndSec"]
            call_label = row.get("AnnotationLevel", "Unknown")
            
        
        # calculating duration of call
            call_duration = end_time_sec - start_time_sec

        # determining bounds for 5s clip 
            if call_duration <= clip_length:
                midpoint = start_time_sec + (call_duration/2.0)
                start_sec = max(0.0, midpoint-(clip_length/2))
            else:
                start_sec = start_time_sec

        
        # converting time coordinates to indices on 32kHz grid
            first_cut = int(start_sec * target_sr)
            second_cut = first_cut + clip_points
            # cutting clip
            call_clip = clean_audio_array[first_cut:second_cut]

            # padding shorter clips
            if len(call_clip) < clip_points:
                # identifying how much is missing
                shortfall = clip_points - len(call_clip)
                # splitting padding between beginning and end of clip
                pad_left = shortfall // 2
                pad_right = shortfall - pad_left
                # adding padding 
                call_clip = np.pad(call_clip, (pad_left, pad_right), "constant")
            
            export_filename = f"clip_{idx}.wav"
            export_path = os.path.join(output_dir, export_filename)
            sf.write(export_path, call_clip, target_sr)

 
print("Processing complete!")







