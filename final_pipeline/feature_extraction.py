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
import time
import csv

# setting up directories
base_dir = os.path.abspath(os.path.dirname(__file__)) if '__file__' in locals() else os.getcwd()
input_dir = os.path.join(base_dir, "cleaned_audio_v2")
output_dir = os.path.join(base_dir, "subpopulation_embeddings_v2")
os.makedirs(output_dir, exist_ok=True)

# defining batch size for parallel processing 
batch_size = 16 

# defining load and prep function
def load_and_prep(file_path):
    audio_array, sr = sf.read(file_path)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if sr != 32000:
        audio_array = librosa.resample(audio_array.astype(np.float32), orig_sr=sr, target_sr=32000)
    if len(audio_array) < 160000:
        audio_array = np.pad(audio_array, (0, 160000 - len(audio_array)), 'constant')
    elif len(audio_array) > 160000:
        audio_array = audio_array[:160000]
    return audio_array.astype(np.float32)

# downloading optimised perch model
model_path = os.path.join(base_dir, "perch_v2.onnx")
if not os.path.exists(model_path):
    url = "https://huggingface.co/justinchuby/Perch-onnx/resolve/main/perch_v2.onnx"

    def progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownloading: {min(percent, 100)}%", end="")

    urllib.request.urlretrieve(url, model_path)
    print("Download complete")

# loading model
session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


# finding subpopulation subfolders
subpop_dirs = [d for d in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(d)]
print(f"Found {len(subpop_dirs)} subpopulation folders: {[os.path.basename(d) for d in subpop_dirs]}")

# looping over each subpopulation folder
for subpop_path in subpop_dirs:
    subpop_name = os.path.basename(subpop_path)
    # creating final embedding file for single subpopulation 
    subpop_file = os.path.join(output_dir, f"{subpop_name}_embeddings.csv")
    # creating partial file in case downloading process is interrupted 
    partial_file = os.path.join(output_dir, f"{subpop_name}_embeddings.partial.csv")

    # skipping subpopulations that have been fully processed to avoid redoing work
    if os.path.exists(subpop_file):
        print(f"\n{subpop_name}: embeddings csv already exists, skipping ({subpop_file})")
        continue
    
    # finding .wav files for subpopulation
    clip_files = glob.glob(os.path.join(subpop_path, "*.wav"))
    
    # finding clips that have only been processed for this subpopulation to prevent redoing work 
    done_files = set()
    if os.path.exists(partial_file):
        # reading partial file as dataframe for editing
        existing_df = pd.read_csv(partial_file)
        done_files = set(existing_df["Filename"])
        print(f"\n{subpop_name}: found partial file with {len(done_files)} clips already done")
        
    # determining how many are left for this subpop     
    remaining_files = [f for f in clip_files if os.path.basename(f) not in done_files]
    print(f"{subpop_name}: {len(clip_files)} total clips, {len(remaining_files)} remaining. Starting extraction...")

    
    file_exists = os.path.exists(partial_file)
    start_time = time.time()
    processed_count = 0
    
    with open(partial_file, "a", newline="") as f:
        writer = None

        # processing in chunks depending on batch size 
        for batch_start in range(0, len(remaining_files), batch_size):
            batch_paths = remaining_files[batch_start:batch_start + batch_size]
            batch_names = [os.path.basename(p) for p in batch_paths]

            # using load and prep function for every clip in this batch
            batch_arrays = [load_and_prep(p) for p in batch_paths]
            # formatting arrays as tensors for batch processing 
            batch_tensor = np.stack(batch_arrays, axis=0)  

            # running the batch through the model in one call
            model_outputs = session.run(None, {input_name: batch_tensor})
            batch_embeddings = model_outputs[0] 

            # writing each row in the batch to csv
            for name, embedding_vectors in zip(batch_names, batch_embeddings):
                row = {"Filename": name}
                for i, val in enumerate(embedding_vectors.flatten()):
                    row[f"vector{i}"] = val

                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if not file_exists or os.path.getsize(partial_file) == 0:
                        writer.writeheader()

                writer.writerow(row)
            
            # writing to disk after each batch is processed to ensure completed work isn't lost in case of interruption
            f.flush()
            processed_count += len(batch_paths)

            elapsed = time.time() - start_time
            rate = processed_count / elapsed
            remaining_time = (len(remaining_files) - processed_count) / rate if rate > 0 else 0
            print(f"  [{subpop_name}] {processed_count}/{len(remaining_files)} clips "
                  f"({elapsed:.1f}s elapsed, ~{remaining_time:.1f}s remaining, {rate:.2f} clips/sec)"
                 )
    # renaming partial fine as subpop file after processing is complete for that subpopulation 
    os.rename(partial_file, subpop_file)
    print(f"  Completed {subpop_name}: renamed to {subpop_file}")

print("\nAll subpopulations processed.")

