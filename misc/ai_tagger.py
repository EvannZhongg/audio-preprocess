# -*- coding: utf-8 -*-
import os
import argparse
import google.generativeai as genai
from tqdm import tqdm
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.tool import load_cfg

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Updated Predefined Tag Categories ---
TAG_CATEGORIES = {
    "叙述": "(韵律平稳，无特别高亢的情感)",
    "表现": "(韵律跳跃，有更激昂的情感)",
}


class AITagger:
    """
    A class to tag audio files using the Gemini API based on predefined categories.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash-latest"):
        """
        Initializes the Gemini model and constructs the detailed prompt.
        """
        if not api_key:
            raise ValueError("Gemini API key is required.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.prompt = self._build_prompt()

    def _build_prompt(self):
        """Builds the detailed prompt for the Gemini model."""
        prompt_lines = [
            "你是一位专业的音频情感和风格分析师。",
            "你的任务是只根据给定的音频，从以下预设的分类中，选择一个最能描述说话者语气和情感的标签。",
            "要优先关注音频本身的语调、语速和情感，而不是文本的字面意思。",
            "\n可用的分类如下：",
        ]
        for tag, description in TAG_CATEGORIES.items():
            prompt_lines.append(f"- {tag} {description}")

        prompt_lines.extend(
            [
                "\n请严格遵守以下规则：",
                "1. 你的回答必须只能是上述分类中的一个，且仅包含标签的中文名称。",
                "2. 不要添加任何解释、理由或多余的文字。",
                "例如，如果判断结果是“表现”，你的完整回答就应该是：表现",
                "\n现在，请分析给定的音频。",
            ]
        )
        return "\n".join(prompt_lines)

    def _clean_response(self, text: str) -> str:
        """Cleans the model's response to ensure it's a valid tag."""
        for tag in TAG_CATEGORIES:
            if tag in text:
                return tag
        return "未知"

    def get_tag(self, audio_path: str) -> str:
        """
        Gets a single emotion/style tag for the given audio file.
        """
        try:
            audio_file = genai.upload_file(path=audio_path, display_name=os.path.basename(audio_path))
            logging.info(f"Uploading file: {audio_path}")

            response = self.model.generate_content(
                [self.prompt, audio_file],
                request_options={"timeout": 120},
            )

            genai.delete_file(audio_file.name)

            if response.text:
                cleaned_tag = self._clean_response(response.text)
                logging.info(f"File '{os.path.basename(audio_path)}' tagged as: {cleaned_tag}")
                return cleaned_tag
            else:
                logging.warning(f"No text response for {audio_path}. Skipping.")
                return "未知"

        except Exception as e:
            logging.error(f"Error processing {audio_path}: {e}")
            return "错误"


def tag_and_rename_file(wav_path: str, tagger: AITagger):
    """Worker function to tag a single audio file and rename it."""
    base_name = os.path.basename(wav_path).rsplit(".wav", 1)[0]
    txt_path = os.path.join(os.path.dirname(wav_path), f"{base_name}.normalized.txt")

    # Skip if the text file doesn't exist or file is already tagged
    if not os.path.exists(txt_path):
        logging.warning(f"No corresponding .txt file for {wav_path}. Skipping.")
        return None

    # Simple check to see if the file might already be tagged (contains a Chinese character)
    if re.search("[\u4e00-\u9fff]", base_name):
        logging.info(f"File {base_name} seems to be already tagged. Skipping.")
        return None

    tag = tagger.get_tag(wav_path)

    if tag not in TAG_CATEGORIES:
        logging.warning(f"Could not get a valid tag for {wav_path}. Skipping rename.")
        return None

    # Rename files
    try:
        # e.g., SPEAKER_00-00001 -> ('SPEAKER_00', '00001')
        parts = base_name.rsplit("-", 1)
        if len(parts) != 2:
            logging.warning(f"Filename '{base_name}' not in expected format 'SPEAKER-ID-NUMBER'. Skipping rename.")
            return None

        speaker_id, utt_num = parts
        new_base_name = f"{speaker_id}_{tag}-{utt_num}"

        new_wav_path = os.path.join(os.path.dirname(wav_path), f"{new_base_name}.wav")
        new_txt_path = os.path.join(os.path.dirname(wav_path), f"{new_base_name}.normalized.txt")

        os.rename(wav_path, new_wav_path)
        os.rename(txt_path, new_txt_path)
        # Suppress individual rename logs for cleaner concurrent output
        # logging.info(f"Renamed {base_name} -> {new_base_name}")
        return tag

    except Exception as e:
        logging.error(f"Failed to rename files for {base_name}: {e}")
        return None


def process_directory(folder_path: str, tagger: AITagger, max_workers: int):
    """
    Walks through a directory, tags audio files concurrently, and renames them.
    Returns a counter of the tags applied.
    """
    # Collect all wav files first
    wav_files_to_process = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                wav_files_to_process.append(os.path.join(root, file))

    if not wav_files_to_process:
        logging.warning("No .wav files found in the specified directory.")
        return Counter()

    tag_counts = Counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_wav = {
            executor.submit(tag_and_rename_file, wav_path, tagger): wav_path
            for wav_path in wav_files_to_process
        }

        for future in tqdm(
            as_completed(future_to_wav),
            total=len(wav_files_to_process),
            desc="Tagging Audio Files",
        ):
            tag = future.result()
            if tag:
                tag_counts[tag] += 1

    return tag_counts


def print_tag_summary(tag_counts: Counter):
    """Prints a formatted summary of the tag distribution."""
    total_tagged_files = sum(tag_counts.values())

    if total_tagged_files == 0:
        logging.info("No new files were tagged in this session.")
        return

    summary_lines = [
        "\n--- Tagging Summary ---",
        f"Total files tagged in this session: {total_tagged_files}",
    ]

    for tag in TAG_CATEGORIES:
        count = tag_counts.get(tag, 0)
        percentage = (count / total_tagged_files) * 100
        summary_lines.append(f"- {tag}: {count} files ({percentage:.2f}%)")

    summary_lines.append("---------------------\n")
    logging.info("\n".join(summary_lines))


def main():
    """Main function to run the AI Tagger."""
    parser = argparse.ArgumentParser(description="AI Tagger for audio files in LibriTTS format.")
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the root directory of the dataset (e.g., the folder containing SPEAKER_00, etc.)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.json",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of concurrent workers for API calls.",
    )
    args = parser.parse_args()

    # Load config to get API key
    cfg = load_cfg(args.config_path)
    gemini_cfg = cfg.get("gemini")
    if not gemini_cfg or not gemini_cfg.get("api_key"):
        raise ValueError("Gemini API key not found in config.json under 'gemini' section.")

    # Initialize the tagger
    tagger = AITagger(api_key=gemini_cfg["api_key"])

    # Process the directory
    tag_counts = process_directory(args.input_folder, tagger, args.workers)

    # Print the summary report
    print_tag_summary(tag_counts)

    logging.info("Tagging process completed.")


if __name__ == "__main__":
    main() 