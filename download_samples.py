# lmports
import os
import pandas as pd
from google.cloud import storage
import soundfile as sf
import scipy.io.wavfile as wav

# making file for data
local_data_dir = "data"
os.makedirs(local_data_dir, exist_ok=True)

# converting annotation file to dataframe
csv_filename = "dclde_2027_dclde_2027_killer_whales_Annotations.csv"
df = pd.read_csv(csv_filename)

# identifying files with killer whale calls and subpopulation data
cleaned_df = df[(df["ClassSpecies"] == "KW") & (df["ecotype"].notna())]
kw_files = cleaned_df["Soundfile"].unique()
# counting number of files with killer whale calls in total
print(f"Total unique killer whale files available: {len(kw_files)}")

# defining the number of files to download
sample_size = 20
files_to_download = kw_files[:sample_size]

# finding relevant files on cloud
client = storage.Client.create_anonymous_client()
bucket = client.bucket("noaa-passive-bioacoustic")
cloud_prefix = "dclde/2027/dclde_2027_killer_whales/dfo_crp/audio/northbc/"

# creating download loop
for idx, filename in enumerate(files_to_download, 1):
	# ensuring extension type is consistent 
	base_name = filename.replace(".flac","").replace(".wav","")
	local_wav_path = os.path.join(local_data_dir, f"{base_name}.wav")
	
	# determining whether cloud file is .flac or .wav 
	flac_cloud_path = f"{cloud_prefix}{base_name}.flac"
	wav_cloud_path = f"{cloud_prefix}{base_name}.wav"

	try: 
		# check if file already exists as .wav file
		if bucket.blob(wav_cloud_path).exists():
			bucket.blob(wav_cloud_path).download_to_filename(local_wav_path)

		elif bucket.blob(flac_cloud_path).exists():
			local_flac_path = os.path.join(local_data_dir, f"{base_name}.flac")
			bucket.blob(flac_cloud_path).download_to_filename(local_flac_path)
			
			# converting to wav 
			data, samplerate = sf.read(local_flac_path)
			wav.write(local_wav_path, samplerate, data)

			# removing flac file 
			os.remove(local_flac_path)

		else:
			print(f"Could not find {base_name} as .wav or .flac on cloud.")

	except Exception as e:
    	print(f"Error processing {base_name}: {e}")

print("Download complete.")

			
	
	
