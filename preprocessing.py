#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
print("GPUs detected:", tf.config.list_physical_devices('GPU'))
tf.debugging.set_log_device_placement(True)
import subprocess 
import json 
import matplotlib.pyplot as plt

# setup 
target_sr = 32000
clip_length = 5
clip_points = clip_length * target_sr 

# defining directories
## using relative paths so that it works on mac and windows
base_dir = os.path.abspath(os.path.dirname(__file__)) if '__file__' in locals() else os.getcwd()
data_dir = os.path.join(base_dir, "data")
output_dir = os.path.join(base_dir, "cleaned_audio_v2")
os.makedirs(output_dir, exist_ok=True)

# finding .wav files 
wav_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive = True )

# defining functions
# defining function to determine bandpass bounds 
def get_bandpass_bounds(nyquist, low_freq=500, high_freq_target=15000):
    
    # defining upper bound
    high_freq = min(high_freq_target, nyquist - 100)
    # preventing invalid upper bound 
    high_freq = max(low_freq + 100, high_freq)  
    low = low_freq / nyquist
    # defining upper bound
    high = min(0.99, high_freq / nyquist)
    return low, high

# defining resampling and bandpassing function
def resample_and_bandpass(audio_file_path, target_sr=32000):

    # defining function to retieve sampling rate 
    def get_sr_ffprobe(audio_file_path):
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", audio_file_path],
            capture_output=True, text=True
        )
        info = json.loads(result.stdout)
        if "streams" not in info or len(info["streams"]) == 0:
            raise ValueError(f"ffprobe could not read streams for {audio_file_path}")
        return int(info["streams"][0]["sample_rate"])

    # defining function to load audio with either librosa or soundfile 
    def load_audio_with_fallback(path, sr=None):
        try:
            audio_array, actual_sr = librosa.load(path, sr=sr)
            return audio_array, actual_sr
        # using soundfile as backup option 
        except Exception as e:
            print(f"librosa.load failed for {path} ({e}). Trying soundfile fallback")
            audio_array, actual_sr = sf.read(path)
            # taking first channel if multi-channel
            if audio_array.ndim > 1:
                audio_array = audio_array[:, 0]  
            if sr is not None and actual_sr != sr:
                audio_array = librosa.resample(audio_array, orig_sr=actual_sr, target_sr=sr)
                actual_sr = sr
            return audio_array, actual_sr

    # retrieving original sampling rate of audio
    try:
        original_sr = librosa.get_samplerate(audio_file_path)
    except Exception:
        original_sr = get_sr_ffprobe(audio_file_path)

    # determining the nyquist frequency of target sampling rate (32 kHz)
    nyquist_target = 0.5 * target_sr

    # setting up preprocessing for audio with low sample rate
    if original_sr < target_sr:
        try:
            audio_array, sr = load_audio_with_fallback(audio_file_path, sr=None)
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}
        
        # retrieving nyquist frequency of clips native sampling rate 
        original_nyquist = 0.5 * original_sr
        # getting bandpassing bounds for the native sampling rate
        low, high = get_bandpass_bounds(original_nyquist)
        # bandpassing 
        sos = butter(N=3, Wn=[low, high], btype="bandpass", output="sos")
        filtered_original = sosfilt(sos, audio_array)
        
        # resampling to 32 kHz
        preproc_audio = librosa.resample(filtered_original, orig_sr=original_sr, target_sr=target_sr)

        return {
            "audio_data": preproc_audio,
            "status": "SUCCESS",
            "original_sr": original_sr,
            "final_sr": target_sr
        }

    # setting up preprocessing for audio with high SR
    else:
        try:
            # resampling to 32 kHz
            resampled_audio, sr = load_audio_with_fallback(audio_file_path, sr=target_sr)
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}
    
        # determing bounds for bandpass
        low, high = get_bandpass_bounds(nyquist_target)
        # bandpassing 
        sos = butter(N=3, Wn=[low, high], btype="bandpass", output="sos")
        final_audio = sosfilt(sos, resampled_audio)

        return {
            "audio_data": final_audio,
            "status": "SUCCESS",
            "original_sr": original_sr,
            "final_sr": target_sr
        }

    # going through each 15 s clip  
    for file in wav_files:

    # resampling and bandpassing 
    resample_result = resample_and_bandpass(file, target_sr = target_sr)
    if resample_result["status"] != "SUCCESS":
        print(f"  Failed to resample and/or bandpass: {resample_result.get('error', 'unknown error')}")
        continue

    # extracting array from dictionary response 
    clean_audio_array = resample_result["audio_data"]

    # checking how much of the clip is real audio vs padding 
   expected_full_points = int(15.0 * target_sr)
    actual_len = len(clean_audio_array)
    if actual_len < expected_full_points:
        buffer_shortfall_sec = (expected_full_points - actual_len) / target_sr
        # skipping clips with over 7.5 seconds of padding 
        if buffer_shortfall_sec > 7.5:
            print(f"Skipping {os.path.basename(file)} - original clip needed "
                  f"{buffer_shortfall_sec:.1f}s of padding (exceeds 7.5s threshold)")
            continue
    
    # slicing calls into 5s clips (keeping middle 5 seconds)
    total_len = len(clean_audio_array)
    center = total_len // 2
    half_window = clip_points // 2
    start_idx = max(0, center - half_window)
    end_idx = start_idx + clip_points
    call_clip = clean_audio_array[start_idx:end_idx]
    
    # padding the final 5s crop if it's still short (should be rare, given the check above)
    if len(call_clip) < clip_points:
        shortfall = clip_points - len(call_clip)
        pad_left = shortfall // 2
        pad_right = shortfall - pad_left
        call_clip = np.pad(call_clip, (pad_left, pad_right), "constant")
    
    # saving cleaned audio to output folder
    export_filename = os.path.basename(file)  
    subpop = os.path.basename(os.path.dirname(file))
    # saving files by subpopulation 
    subpop_output_dir = os.path.join(output_dir, subpop)
    os.makedirs(subpop_output_dir, exist_ok=True)
    export_path = os.path.join(subpop_output_dir, export_filename)
    
    sf.write(export_path, call_clip, target_sr)

print("Pre-processing complete!")

# removing clips near the start of the clip that would primarily be padding 
# loading metadata with cluster_midpoint
calls = pd.read_csv(os.path.join(data_dir, "downloaded_calls_metadata.csv"))

# finding cleaned audio files
all_files = glob.glob(os.path.join(output_dir, "**", "*.wav"), recursive=True)
all_files_by_basename = {os.path.basename(f): f for f in all_files}

#identifying clusters with midpoints too close to the start of the clip
calls["near_start_edge"] = calls["cluster_midpoint"] < 7.5
flagged = calls[calls["near_start_edge"]]

print(f"{len(flagged)} out of {len(calls)} total clusters need to be removed.")
print(flagged["Subpopulation"].value_counts())

# removing clusters with midpoints too close to the start of the clip
removed_count = 0
for _, row in flagged.iterrows():
    fname = f"clip_{os.path.splitext(row['Soundfile'])[0].replace('.', '_')}_{row['cluster_id']}.wav"
    match = all_files_by_basename.get(fname)
    if match:
        os.remove(match)
        removed_count += 1

print(f"Removed {removed_count} files")

# creating set for number of files in each subpopulation folder
subpop_counts = {}

if os.path.exists(data_dir):
    for subpop in sorted(os.listdir(data_dir)):
        # skipping hiden folders 
        if subpop.startswith("."):  
            continue
            
        subpop_path = os.path.join(data_dir, subpop)

        # inspecting single subpopulation folder 
        if os.path.isdir(subpop_path):
            # counting .wav files inside the subfolder
            wav_files = [
                f for f in os.listdir(subpop_path) if f.endswith(".wav")
            ]
            subpop_counts[subpop] = len(wav_files)
            

    # collecting all subpopulation counts in dataframe 
    counts_df = pd.DataFrame(
        list(subpop_counts.items()), columns=["Subpopulation", "File Count"]
    )
    counts_df.loc[len(counts_df)] = ["Total", counts_df["File Count"].sum()]

    print(counts_df.to_string(index=False))
else:
    print(f"Directory '{data_dir}' does not exist.")
    

# dropping total for bar chart 
plot_df = counts_df[counts_df["Subpopulation"] != "Total"].copy()

# sorting by count so imbalance is clear
plot_df = plot_df.sort_values("File Count", ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
bars = sns.barplot(data=plot_df, x="Subpopulation", y="File Count", ax=ax, color="steelblue")

# annotating bars with raw count values
for bar in bars.patches:
    height = bar.get_height()
    ax.annotate(f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  
                textcoords="offset points",
                ha="center", va="bottom", fontsize=10)

ax.set_xlabel("Subpopulation")
ax.set_ylabel("Number of clusters")
ax.set_title("Distribution of clusters across subpopulations")

plt.tight_layout()
plt.savefig("subpopulation_clip_counts.png", dpi=300)
plt.show()

