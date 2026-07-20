# Imports 
import os
import pandas as pd
from google.cloud import storage
import soundfile as sf
import scipy.io.wavfile as wav
import numpy as np
import librosa
import gc 
import subprocess 
import json
import glob
import shutil

import os
import pandas as pd
from google.cloud import storage
import soundfile as sf
import scipy.io.wavfile as wav
import numpy as np
import librosa
import gc 
import subprocess 
import json
import glob
import shutil

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
cleaned_df['Provider'] = cleaned_df['Provider'].replace('UAF_NGOS', 'UAF')

print(cleaned_df.head)
confound_check = pd.crosstab(cleaned_df['Subpopulation'], 
                             cleaned_df['Provider'], 
                             margins = True)
print(confound_check)

# defining function to retrieve sample rate if librosa fails
def get_sr_ffprobe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
        capture_output=True, text=True
    )
    info = json.loads(result.stdout)
    
    return int(info["streams"][0]["sample_rate"])

# defining function cluster calls within 10 second window 
def clustering_calls(calls_df, analysis_window_sec=5.0, max_cluster_span_sec=10.0):
    
    # creating empty lists for calls within a cluster and clusters to exclude 
    clustered_rows = []
    excluded_clusters = []

    for soundfile, group_df in calls_df.groupby("Soundfile"):
        # sorting calls in sound file chronologically 
        df = group_df.sort_values("FileBeginSec").copy()
        # defining mid point of cluster
        df["midpoint"] = (df["FileBeginSec"] + df["FileEndSec"]) / 2.0
        df["crop_start"] = df["midpoint"] - analysis_window_sec / 2
        df["crop_end"] = df["midpoint"] + analysis_window_sec / 2

        clusters = []
        current_cluster = [df.iloc[0]]
        current_max_end = df.iloc[0]["crop_end"]
        cluster_span_start = df.iloc[0]["FileBeginSec"]
        cluster_span_end = df.iloc[0]["FileEndSec"]
        
        # making sure that clusters do not go beyond 10 seconds 
        for i in range(1, len(df)):
            row = df.iloc[i]
            would_be_span_end = max(cluster_span_end, row["FileEndSec"])
            would_be_span = row["FileEndSec"] - cluster_span_start
            
            if row["crop_start"] < current_max_end and would_be_span <= max_cluster_span_sec:
                current_cluster.append(row)
                current_max_end = max(current_max_end, row["crop_end"])
                cluster_span_end = would_be_span_end
            else:
                clusters.append(pd.DataFrame(current_cluster))
                current_cluster = [row]
                current_max_end = row["crop_end"]
                cluster_span_start = row["FileBeginSec"]
                cluster_span_end = row["FileEndSec"]
                
        clusters.append(pd.DataFrame(current_cluster))

        for cluster_id, cluster_df in enumerate(clusters):
            subpops = cluster_df["Subpopulation"].unique()
            if len(subpops) > 1:
                print(f"WARNING: {soundfile} cluster {cluster_id} spans multiple "
                      f"subpopulations: {subpops}. Dropping cluster.")
                excluded_clusters.append({
                    "Soundfile": soundfile,
                    "cluster_id": cluster_id,
                    "subpopulations_found": list(subpops),
                    "num_calls_in_cluster": len(cluster_df)})
                continue

            # avereraging midpoint of all calls in cluster to get cluster midpoint 
            cluster_midpoint = cluster_df["midpoint"].mean()

            # creating dictionary of cluster info
            clustered_rows.append({
                "Soundfile": soundfile,
                "Provider": cluster_df["Provider"].iloc[0],
                "Subpopulation": subpops[0],
                "cluster_id": cluster_id,
                "num_calls_in_cluster": len(cluster_df),
                "cluster_midpoint": cluster_midpoint,
                "cluster_span_sec": cluster_df["FileEndSec"].max() - cluster_df["FileBeginSec"].min(),
            })

    return pd.DataFrame(clustered_rows), pd.DataFrame(excluded_clusters)

# removing duplicates 
deduplicated_df = cleaned_df.drop_duplicates(subset=["Soundfile", "FileBeginSec", "FileEndSec"])

# removing clusters that span over 10 seconds 
deduplicated_df = deduplicated_df[
    (deduplicated_df["FileEndSec"] - deduplicated_df["FileBeginSec"]) <= 10.0
].copy()

# creating dataframe with clusters as rows 
calls, excluded_df = clustering_calls(deduplicated_df, analysis_window_sec=5.0)

# grouping calls by the files they belong to 
calls_by_file = calls.groupby("Soundfile")

total_num_clusters = len(calls)
total_num_files = calls["Soundfile"].nunique()

print(f"total number of call clusters: {total_num_clusters}")
print(f"total number of raw files with killer whale calls: {total_num_files}")

# finding relevant files on cloud
client = storage.Client.create_anonymous_client()
bucket = client.bucket("noaa-passive-bioacoustic")

# creating column to track future filename - CHANGE TO CALLS FOR FINAL BIT
calls["ExportFilename"] = [
    f"{row['Subpopulation']}/clip_{soundfile}_{cluster_id}.wav" for idx, row in calls.iterrows()]

# creating download loop
for soundfile, group_df in calls_by_file:
    
    # skipping download if file has already been downloaded
    all_clips_exist = True
    for idx, row in group_df.iterrows():
        subpop_dir = os.path.join(local_data_dir, str(row["Subpopulation"]))
        export_filename = f"clip_{soundfile.replace('.', '_')}_{row['cluster_id']}.wav"
        export_path = os.path.join(subpop_dir, export_filename)
        
        if not os.path.exists(export_path):
            all_clips_exist = False
            break 
            
    if all_clips_exist:
        print(f"Skipping {soundfile} - all {len(group_df)} clips already cut and saved locally.")
        continue
    # ensuring extension type is consistent 
    base_name = soundfile.replace(".flac","").replace(".wav","")
    
    # getting provider name and changing to lowercase to match cloud structure
    provider = str(group_df['Provider'].iloc[0]).lower()
    base_name = soundfile.replace(".flac","").replace(".wav","")
    print(f"{soundfile} has {len(group_df)} clusters")
    
    # creating broad search directory to capture hydrophone subfolders
    search_dir = f"dclde/2027/dclde_2027_killer_whales/{provider}/audio"
    
    
    # searching for a name match anywhere in the file path
    
    blobs_iterator = client.list_blobs(bucket, prefix=search_dir)
    matching_blobs = [b.name for b in blobs_iterator if base_name in b.name]
    
    # if no files match
    if len(matching_blobs) == 0:
        print(f"Could not find match for {soundfile} anywhere under {search_dir}\n")
        continue

    # if more than one file matches
    elif len(matching_blobs) > 1:
        raise ValueError(f"Ambiguous match for {soundfile}: found {matching_blobs}")

    matched_blob_path = matching_blobs[0]

    # temporary path for uncut audio files 
    temp_raw_path = None    
    
    try: 
        if matched_blob_path.lower().endswith(".wav"):
            temp_raw_path = os.path.join(local_data_dir, f"temp_{base_name}.wav")
        elif matched_blob_path.lower().endswith(".flac"):
            temp_raw_path = os.path.join(local_data_dir, f"temp_{base_name}.flac")
        
        # downloading file if match found
        if temp_raw_path:
            bucket.blob(matched_blob_path).download_to_filename(temp_raw_path)

        # ensuring that download is complete by comparing file sizes  
        if temp_raw_path and os.path.exists(temp_raw_path):
            blob_obj = bucket.blob(matched_blob_path)
            blob_obj.reload()  
            actual_size = os.path.getsize(temp_raw_path)
            expected_size = blob_obj.size

            # raising error if corrupted download led to incomplete file
            if actual_size != expected_size:
                raise ValueError(f"Download size mismatch for {soundfile}: "
                                  f"expected {expected_size}, got {actual_size}")
        
         # cutting slices if download succeeded                          
        if temp_raw_path and os.path.exists(temp_raw_path):
            # get native sr of audio file 
            try: 
                info = sf.info(temp_raw_path)
                samplerate = librosa.get_samplerate(temp_raw_path)
            
            # using previously defined function if librosa failed to retrieve sr
            except Exception: 
                samplerate = get_sr_ffprobe(temp_raw_path)
                print(f"ffprobe successfully parsed RF64 container. Native rate: {samplerate} Hz")
                
            # defining 15 seconds relative to native sample rate
            clip_length_sec = 15.0
            clip_points = int(clip_length_sec * samplerate)
                                         
            # determining when to make cuts depending on midpoint of vocalisation
            for idx, row in group_df.iterrows():
                
                # using midpoint found in clustering calls step
                midpoint = row["cluster_midpoint"]
                # defining clip length
                clip_length_sec = 15.0  

                start_sec = max(0.0, midpoint - (clip_length_sec / 2.0))
                
                # loading clip
                call_clip, _ = librosa.load(
                    temp_raw_path,
                    offset=start_sec,
                    duration=clip_length_sec,
                    sr=None,
                    mono=False
                )
                # choosing first channel if multiple channels are present 
                if call_clip.ndim > 1 and call_clip.shape[0] > 1:
                    call_clip = call_clip[0, :]
                    
                # padding in case clip is less than 15 seconds 
                if len(call_clip) < clip_points:
                    missing = clip_points - len(call_clip)
                    if missing > clip_points * 0.3:  # more than 30% padding - worth flagging
                        print(f"WARNING: {soundfile} cluster {row['cluster_id']} needed "
                              f"{missing/samplerate:.1f}s of padding out of {clip_length_sec}s - "
                              f"check if this ran past file end")
                    call_clip = np.pad(call_clip, (0, missing), "constant")
                
                # adding clip to subpopulation folder
                subpop_dir = os.path.join(local_data_dir, row["Subpopulation"])
                os.makedirs(subpop_dir, exist_ok=True)
                
                # labelling file by cluster
                export_filename = f"clip_{soundfile.replace('.', '_')}_{row['cluster_id']}.wav"
                export_path = os.path.join(subpop_dir, export_filename)
                sf.write(export_path, call_clip, samplerate)

                print(f"{row['Subpopulation']}/{export_filename} saved as 15s clip "
                      f"({row['num_calls_in_cluster']} call(s) merged)")
                
            # deleting original clip to free up space 
            os.remove(temp_raw_path)
            print(f"Deleted {soundfile} from disk\n")
            
            # wipe audio array from RAM
            if 'call_clip' in locals():
                del call_clip
            # release freed memory back to the system
            gc.collect()

        else:
            print(f"Could not find {soundfile} as .wav or .flac on cloud.\n")

    except Exception as e:
        print(f"Error processing {base_name}: {e}")
        
        # deleting raw file even if processing fails 
    finally:
        if temp_raw_path and os.path.exists(temp_raw_path):
            os.remove(temp_raw_path)
            print(f"Cleaned up {temp_raw_path} from disk\n")
            
print("Download complete.")

# creating export path for .csv file
metadata_export_path = os.path.join(local_data_dir, "downloaded_calls_metadata.csv")
# converting dataframe to .csv file
calls.to_csv(metadata_export_path, index = False) # change to calls for final one
print(f"Saved matched metadata table to {metadata_export_path}")