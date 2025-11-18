import json
import os
import sys

from mutagen.aac import AAC
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.wave import WAVE
from tqdm import tqdm


def get_audio_files(folder_path):
    """Get all audio files in a folder."""
    audio_files = []
    for root, _, files in os.walk(folder_path):
        if "_processed" in root:
            continue
        for file in files:
            if ".temp" in file:
                continue
            if file.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac", ".mp4")):
                audio_files.append(os.path.join(root, file))
    return audio_files

def get_audio_duration(file_path):
    try:
        """Get the duration of an audio file in seconds."""
        if file_path.endswith('.mp3'):
            audio = MP3(file_path)
        elif file_path.endswith('.mp4'):
            audio = MP4(file_path)
        elif file_path.endswith('.m4a'):
            audio = MP4(file_path)
        elif file_path.endswith('.wav'):
            audio = WAVE(file_path)
        elif file_path.endswith('.flac'):
            audio = FLAC(file_path)
        elif file_path.endswith('.aac'):
            audio = AAC(file_path)
        else:
            sys.exit(1)
        return int(audio.info.length)
    except Exception as e:
        try:
            audio = MP3(file_path)
            return int(audio.info.length)
        except Exception as e:
            print(f"Error processing {file_path} with pydub: {str(e)}")
            return 3600


def analyze_podcasts(base_path):
    podcast_data = []
    podcast_total_seconds = 0

    audio_paths = get_audio_files(base_path)
    for audio_path in tqdm(audio_paths, desc="Processing Podcasts"):

        relative_path = os.path.relpath(os.path.dirname(audio_path), base_path)
        audio_duration_second = get_audio_duration(audio_path) / 3600
        podcast_data.append({
            "relative_path": relative_path,
            "audio_path": audio_path,
            "audio_duration_second": audio_duration_second,
        })
        podcast_total_seconds += audio_duration_second

    # Sort podcasts by audio duration
    podcast_data.sort(key=lambda x: x["audio_duration_second"], reverse=False)

    return {
        "podcast_total_hour": podcast_total_seconds / 3600,
        "podcast_data": podcast_data
    }

def main():
    # base_path = '/cfs-du3y2r4h/podcast-data/download'
    base_path = "examples"
    result = analyze_podcasts(base_path)
    with open('podcast_data.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
