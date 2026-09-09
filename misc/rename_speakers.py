import os
import hashlib
import sys
import json
import pandas as pd

def get_short_hash(text, length=6):
    """
    Generates a short, deterministic hash from a string.
    """
    if isinstance(text, str):
        text = text.encode('utf-8')
    hasher = hashlib.sha1(text)
    return hasher.hexdigest()[:length]

def update_speaker_names_libritts(root_dir):
    """
    Scans a directory of processed podcasts in LibriTTS format and renames speakers.
    It identifies speaker folders directly, without relying on a central JSON file.
    1. Finds all 'SPEAKER_XX' subdirectories in a podcast folder.
    2. Creates a unique, hash-based name for each speaker.
    3. Renames the speaker folders.
    4. Renames the audio and transcription text files within those folders.
    5. Updates the speaker ID column in the corresponding .trans.tsv files.
    """
    print(f"--- Starting LibriTTS speaker name update for directory: {root_dir} ---")
    if not os.path.isdir(root_dir):
        print(f"Error: Root directory not found at '{root_dir}'")
        return

    # root_dir contains directories for each podcast
    for podcast_name in os.listdir(root_dir):
        podcast_path = os.path.join(root_dir, podcast_name)
        if not os.path.isdir(podcast_path):
            continue

        print(f"\nProcessing podcast: {podcast_name}")

        # --- 1. Find speaker folders directly ---
        try:
            subdirs = [d for d in os.listdir(podcast_path) if os.path.isdir(os.path.join(podcast_path, d))]
        except OSError as e:
            print(f"  [!] Could not read subdirectories of '{podcast_path}'. Error: {e}")
            continue

        old_speakers = sorted([d for d in subdirs if d.startswith("SPEAKER_")])

        if not old_speakers:
            print(f"  [*] No 'SPEAKER_XX' folders found. Assuming already updated or empty. Skipping.")
            continue

        # --- 2. Generate new names and mapping ---
        file_hash = get_short_hash(podcast_name)
        speaker_mapping = {
            old_speaker: f"SPK_{file_hash}_{old_speaker.split('_')[-1]}"
            for old_speaker in old_speakers
        }
        print(f"  [*] Generated hash '{file_hash}'. Mapping created for {len(speaker_mapping)} speakers.")

        # --- 3. Rename folders, files, and update TSVs ---
        all_renames_successful = True
        for old_name, new_name in speaker_mapping.items():
            old_folder_path = os.path.join(podcast_path, old_name)
            new_folder_path = os.path.join(podcast_path, new_name)

            if not os.path.isdir(old_folder_path):
                print(f"  [*] Speaker folder '{old_name}' not found, skipping.")
                continue

            # --- 3a. Update the .trans.tsv file ---
            tsv_path = os.path.join(old_folder_path, f"{old_name}.trans.tsv")
            new_tsv_path = os.path.join(old_folder_path, f"{new_name}.trans.tsv")
            if os.path.exists(tsv_path):
                try:
                    df = pd.read_csv(tsv_path, sep='\t', header=None)
                    # The first column is the speaker ID
                    df[0] = new_name
                    df.to_csv(tsv_path, sep='\t', header=False, index=False)
                    os.rename(tsv_path, new_tsv_path)
                    print(f"  [+] Updated and renamed TSV for '{old_name}'")
                except Exception as e:
                    print(f"  [!] Error updating TSV file for '{old_name}': {e}. Stopping update for this podcast.")
                    all_renames_successful = False
                    break

            # --- 3b. Rename the audio files and associated text files inside ---
            try:
                for filename in os.listdir(old_folder_path):
                    if filename.startswith(old_name) and (filename.endswith(".wav") or filename.endswith(".txt")):
                        new_filename = filename.replace(old_name, new_name, 1)
                        os.rename(os.path.join(old_folder_path, filename), os.path.join(old_folder_path, new_filename))
                print(f"  [+] Renamed .wav and .txt files in folder '{old_name}'")
            except OSError as e:
                print(f"  [!] Error renaming contents in '{old_folder_path}': {e}. Stopping update for this podcast.")
                all_renames_successful = False
                break
            
            # --- 3c. Rename the folder itself ---
            try:
                os.rename(old_folder_path, new_folder_path)
                print(f"  [+] Renamed folder: '{old_name}' -> '{new_name}'")
            except OSError as e:
                print(f"  [!] Error renaming folder '{old_folder_path}': {e}. Stopping update for this podcast.")
                all_renames_successful = False
                break
        
        if not all_renames_successful:
            print(f"  [!] Aborted update for '{podcast_name}' due to errors.")
            # Here, we might need a rollback strategy, but for now, we stop.
            continue

    print("\n--- Update process finished. ---")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/rename_speakers.py <path_to_processed_folder>")
        print("Example: python tools/rename_speakers.py /data/workspace/pod_1/some_pod_for_test_processed")
        sys.exit(1)
        
    target_directory = sys.argv[1]
    # We call the new LibriTTS-specific function
    update_speaker_names_libritts(target_directory) 