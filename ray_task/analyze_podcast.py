import os
import sys
import json
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.wave import WAVE
from mutagen.flac import FLAC
from mutagen.aac import AAC

def get_audio_duration(file_path):
    try:
        """Get the duration of an audio file in seconds."""
        if file_path.endswith('.mp3'):
            audio = MP3(file_path)
        elif file_path.endswith('.m4a') or file_path.endswith('.mp4'):
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
        print(f"{file_path} exception {str(e)}")
        return 3600

def analyze_podcasts(base_path):
    podcast_data = []
    podcast_total_seconds = 0

    for podcast_name in os.listdir(base_path):
        podcast_path = os.path.join(base_path, podcast_name)
        if not os.path.isdir(podcast_path):
            continue

        episode_data = []
        episode_total_seconds = 0

        for episode_name in os.listdir(podcast_path):
            episode_path = os.path.join(podcast_path, episode_name)
            if not os.path.isfile(episode_path):
                continue

            if not episode_path.endswith((".mp3", ".wav", ".flac", ".m4a", ".aac", ".mp4")):
                continue

            audio_duration = get_audio_duration(episode_path)
            episode_total_seconds += audio_duration

            episode_data.append({
                "episode_name": episode_name,
                "audio_duration_second": audio_duration
            })

        # Sort episodes by audio duration
        episode_data.sort(key=lambda x: x["audio_duration_second"], reverse=False)

        podcast_data.append({
            "podcast_name": podcast_name,
            "episode_num": len(episode_data),
            "episode_total_hour": episode_total_seconds / 3600,
            "episode_data": episode_data
        })

        podcast_total_seconds += episode_total_seconds

    # Sort podcasts by total episode hours
    podcast_data.sort(key=lambda x: x["episode_total_hour"], reverse=False)

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