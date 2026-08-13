# lmports
import os
import pandas as pd
from google.cloud import storage
import soundfile as sf
import scipy.io.wavfile as wav
import numpy as np

# making file for data
local_data_dir = "data"
os.makedirs(local_data_dir, exist_ok=True)

# converting annotation file to dataframe
csv_filename = "dclde_2027_dclde_2027_killer_whales_Annotations.csv"
df = pd.read_csv(csv_filename)

# loading supplemental uaf transient data
uaf_df = pd.read_csv("UAF_transient_data.csv")
uaf_df.columns = uaf_df.columns.str.strip()
uaf_sp = uaf_df[["FileName", "Population"]].copy()

# identifying files with killer whale calls and subpopulation data
cleaned_df = df[(df["ClassSpecies"] == "KW") & (df["Ecotype"].notna())]

# merging master annotation df with uaf df 
cleaned_df = pd.merge(cleaned_df, uaf_sp, left_on = "Soundfile", right_on = "FileName", how = "left")

cleaned_df["Subpopulation"] = np.where(cleaned_df["Population"].notna(), 
                                       cleaned_df["Population"],
                                      cleaned_df["Ecotype"])
# replacing all remaining instances of tranisnets as WCT 
cleaned_df['Subpopulation'] = cleaned_df['Subpopulation'].replace('TKW', 'WCT')

print(cleaned_df.head)
confound_check = pd.crosstab(cleaned_df['Subpopulation'], 
                             cleaned_df['Provider'], 
                             margins = True)
print(confound_check)

# defining the number of files to download
sample_size = 20
sample_calls = cleaned_df.head(sample_size).copy() # remove for final run 

# for final download 
# calls = cleaned_df.copy

# grouping calls by the files they belong to 
calls_by_file = sample_calls.groupby("Soundfile")

total_num_calls = len(sample_calls)
total_num_files = sample_calls["Soundfile"].nunique()

print(f"total number of killer whale calls: {total_num_calls}")
print(f"total number of raw files with killer whale calls: {total_num_files}")

# finding relevant files on cloud
client = storage.Client.create_anonymous_client()
bucket = client.bucket("noaa-passive-bioacoustic")
cloud_prefix = "dclde/2027/dclde_2027_killer_whales/dfo_crp/audio/northbc/"

# creating column to track future filename - CHANGE TO CALLS FOR FINAL BIT
sample_calls["ExportFilename"] = [f"{row['Ecotype']}/clip_{idx}.wav" for idx in sample_calls.index]

# creating download loop
for soundfile, group_df in calls_by_file:
    # ensuring extension type is consistent 
    base_name = soundfile.replace(".flac","").replace(".wav","")

    # temporary path for uncut audio file
    temp_raw_path = None 
    
    # determining whether cloud file is .flac or .wav 
    flac_cloud_path = f"{cloud_prefix}{base_name}.flac"
    wav_cloud_path = f"{cloud_prefix}{base_name}.wav"
    
    print(f"{soundfile} has {len(group_df)} calls")

    try: 
        # check if file already exists as .wav file
        if bucket.blob(wav_cloud_path).exists():
          # download raw .wav file to temp path
            temp_raw_path = os.path.join(local_data_dir, f"temp_{base_name}.wav")
            bucket.blob(wav_cloud_path).download_to_filename(temp_raw_path)

        elif bucket.blob(flac_cloud_path).exists():
            # download raw.flac file to temp path
            temp_raw_path = os.path.join(local_data_dir, f"temp_{base_name}.flac")
            bucket.blob(flac_cloud_path).download_to_filename(temp_raw_path)
                                         
         # cutting slices if download succeeded                          
        if temp_raw_path and os.path.exists(temp_raw_path):
            # read audio file at native sr
            data, samplerate = sf.read(temp_raw_path)
            total_samples = len(data)
                                         
            # defining 15 seconds relative to native sample rate
            clip_length_sec = 15.0
            clip_points = int(clip_length_sec * samplerate)
                                         
            # determining when to make cuts depending on midpoint of vocalisation
            for idx, row in group_df.iterrows():
                start_time_sec = row["FileBeginSec"]
                end_time_sec = row["FileEndSec"]
                call_duration = end_time_sec - start_time_sec 
                midpoint = start_time_sec + (call_duration / 2.0)
                
                start_sec = max(0.0, midpoint - (clip_length_sec / 2.0))
                
                # converting time coordinates to indices in array  
                first_cut = int(start_sec * samplerate)
                second_cut = first_cut + clip_points
                
                # cutting array
                call_clip = data[first_cut:second_cut]
                
                # padding in case original audio was less than 15 seconds 
                if len(call_clip) < clip_points: 
                    missing = clip_points - len(call_clip)
                    call_clip = np.pad(call_clip, (0, missing), "constant") 
                
                ecotype_dir = os.path.join(local_data_dir, row["Ecotype"])
                os.makedirs(ecotype_dir, exist_ok=True)
                # converting to wav 
                export_filename = f"clip_{idx}.wav"
                export_path = os.path.join(ecotype_dir, export_filename)
                sf.write(export_path, call_clip, samplerate)
                
                print(f"{row['Ecotype']}/{export_filename} saved as 15s clip")
                                         
            # deleting original clip to free up space 
            os.remove(temp_raw_path)
            print(f"Deleted {soundfile} from disk\n")
        

        else:
            print(f"Could not find {soundfile} as .wav or .flac on cloud.\n")

    except Exception as e:
        print(f"Error processing {base_name}: {e}")
        # deleting raw file even if processing fails 
        if temp_raw_path and os.path.exists(temp_raw_path):
            os.remove(temp_raw_path)

print("Download complete.")

metadata_export_path = os.path.join(local_data_dir, "downloaded_calls_metadata.csv")
sample_calls.to_csv(metadata_export_path, index=False) # change to calls for final one
print(f"Saved matched metadata table to {metadata_export_path}")




