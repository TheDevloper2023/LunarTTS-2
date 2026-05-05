import os
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from text import _clean_text

def prepare_align(config):
    in_dir = config["path"]["corpus_path"]      # Your source root (e.g., Twilight_full)
    out_dir = config["path"]["raw_path"]        # Where preprocessed files go (raw_data)
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    
    speaker = config["dataset"]
    # Usually "filelist.txt" or "metadata.csv" for TT2
    filelist_path = os.path.join(in_dir, "list.txt") 

    with open(filelist_path, encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("|")
            
            # TT2 filelist usually looks like: wavs/audio1.wav|The text transcript.
            rel_wav_path = parts[0] 
            text = parts[1]
            text = _clean_text(text, cleaners)

            # 1. Get the absolute path of the SOURCE file to read it
            # This joins your root dir with the relative path from the csv
            src_wav_path = os.path.join(in_dir, rel_wav_path)

            # 2. Extract just the filename for the output (e.g., "audio1")
            base_name = os.path.splitext(os.path.basename(rel_wav_path))[0]

            if os.path.exists(src_wav_path):
                # Ensure raw_data/Twilight_full/ exists
                save_dir = os.path.join(out_dir, speaker)
                os.makedirs(save_dir, exist_ok=True)
                
                # Load and normalize
                wav, _ = librosa.load(path=src_wav_path, sr=sampling_rate)
                
                wav = wav / (max(abs(wav)) + 1e-7) * max_wav_value
                
                # Save the new normalized wav and the .lab file
                # MFA/HGST expects: raw_data/Twilight_full/audio1.wav
                wavfile.write(
                    os.path.join(save_dir, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                
                with open(
                    os.path.join(save_dir, "{}.lab".format(base_name)),
                    "w",
                    encoding="utf-8"
                ) as f1:
                    f1.write(text)