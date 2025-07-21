import argparse
import os
import random
import csv
import math

def create_manifest(podcast_names, total_count, base_podcast_path, output_path):
    """
    Selects a total number of episodes distributed evenly across the given podcasts
    and generates a CSV manifest.
    """
    print("Starting manifest creation...")
    print(f"Base podcast directory: {base_podcast_path}")
    print(f"Target total episodes: {total_count}")

    # 1. Discover available episodes for each podcast
    podcasts_info = []
    for name in podcast_names:
        podcast_dir = os.path.join(base_podcast_path, name)
        if not os.path.isdir(podcast_dir):
            print(f"\n[!] Warning: Directory not found for podcast '{name}', skipping.")
            continue
        
        try:
            episodes = [f for f in os.listdir(podcast_dir) if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.aac', '.mp4'))]
            if episodes:
                podcasts_info.append({
                    "name": name,
                    "available_count": len(episodes),
                    "episodes": episodes
                })
            else:
                print(f"\n[!] Warning: No audio files found for podcast '{name}', skipping.")
        except Exception as e:
            print(f"\n[!] Error processing directory for '{name}': {e}")

    if not podcasts_info:
        print("\nNo podcasts with available episodes found. Manifest will not be created.")
        return

    # 2. Allocate counts intelligently
    allocation = {p['name']: 0 for p in podcasts_info}
    podcasts_to_process = [p for p in podcasts_info]
    remaining_total = total_count

    while remaining_total > 0 and podcasts_to_process:
        num_active_podcasts = len(podcasts_to_process)
        avg_share = math.ceil(remaining_total / num_active_podcasts)

        podcasts_to_remove = []
        for p in podcasts_to_process:
            can_provide = p['available_count'] - allocation[p['name']]
            to_take = min(can_provide, avg_share)
            
            # Ensure we don't take more than the remaining total
            to_take = min(to_take, remaining_total)

            allocation[p['name']] += to_take
            remaining_total -= to_take
            
            if allocation[p['name']] == p['available_count']:
                podcasts_to_remove.append(p)
            
            if remaining_total <= 0:
                break
        
        # Remove podcasts that are maxed out
        podcasts_to_process = [p for p in podcasts_to_process if p not in podcasts_to_remove]

    print("\n--- Allocation Summary ---")
    final_total = 0
    for name, count in allocation.items():
        print(f"  - {name}: {count} episodes")
        final_total += count
    print(f"--------------------------\nTotal episodes to be selected: {final_total}")


    # 3. Sample episodes and write manifest
    all_selected_episodes = []
    for p_info in podcasts_info:
        podcast_name = p_info['name']
        num_to_select = allocation[podcast_name]
        
        if num_to_select > 0:
            selected_episodes = random.sample(p_info['episodes'], num_to_select)
            for episode_name in selected_episodes:
                file_path = os.path.abspath(os.path.join(base_podcast_path, podcast_name, episode_name))
                all_selected_episodes.append({
                    "PodcastName": podcast_name,
                    "EpisodeName": os.path.splitext(episode_name)[0],
                    "FilePath": file_path
                })

    if all_selected_episodes:
        print(f"\nWriting {len(all_selected_episodes)} entries to: {output_path}")
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["PodcastName", "EpisodeName", "FilePath"])
            writer.writeheader()
            writer.writerows(all_selected_episodes)
        print("Manifest created successfully.")
    else:
        print("\nNo files were selected. Manifest file will not be created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligently select podcast episodes and create a processing manifest.")
    parser.add_argument(
        "--podcasts",
        nargs='+',
        required=True,
        help="A list of podcast folder names. e.g., --podcasts \"My Podcast\" \"Another Show\""
    )
    parser.add_argument(
        "--total_count",
        type=int,
        required=True,
        help="The total number of episodes to select across all specified podcasts."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="The base directory where all podcast folders are stored."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="manifest.csv",
        help="The name of the output CSV manifest file."
    )
    args = parser.parse_args()

    create_manifest(args.podcasts, args.total_count, args.base_dir, args.output_file) 