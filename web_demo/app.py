import os

import subprocess
import uuid
import logging
import re
import json
import shutil
import pathlib
from flask import (Flask, request, jsonify, render_template, send_from_directory,
                   url_for)
from threading import Thread, Lock
from queue import Queue

# --- Project Imports ---
from main_init import EmiliaPipeline, load_cfg
import argparse

app = Flask(__name__, static_folder='static', template_folder='templates')

# --- Path Configuration ---
# Get the absolute path of the directory where the script is located (web_demo)
APP_DIR = pathlib.Path(__file__).parent.resolve()
# The project root is one level up (Emilia)
PROJECT_ROOT = APP_DIR.parent

# --- Configuration ---
# 1GB upload limit
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = str(PROJECT_ROOT / 'web_demo' / 'uploads')
app.config['RESULTS_FOLDER'] = str(PROJECT_ROOT / 'web_demo' / 'results')
app.config['LOG_FILE'] = str(PROJECT_ROOT / 'web_demo' / 'web_demo.log')

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(app.config['LOG_FILE']),
        logging.StreamHandler()
    ]
)

# --- Concurrency Control ---
task_queue = Queue()
task_status = {}
status_lock = Lock()

# --- Load Pipeline on Startup ---
pipeline_instance = None

def load_pipeline():
    global pipeline_instance
    logging.info("Loading models into memory... This may take a moment.")
    try:
        # We need to simulate the args that main.py expects
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_path", type=str, default="config.json")
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--compute_type", type=str, default="float16")
        parser.add_argument("--whisper_arch", type=str, default="medium")
        parser.add_argument("--threads", type=int, default=4)
        parser.add_argument("--num_workers", type=int, default=1) # In server mode, GPU pipeline runs on 1 worker
        args, _ = parser.parse_known_args()

        main_cfg = load_cfg(args.config_path)
        pipeline_instance = EmiliaPipeline(main_cfg, args)
        logging.info("Models loaded successfully. ✅")
    except Exception as e:
        logging.error(f"FATAL: Could not load models. The application cannot start. Error: {e}", exc_info=True)
        # In a real production app, you might want to exit or prevent the app from starting
        pipeline_instance = None

# --- Helper Functions ---
def get_task_status(task_id):
    with status_lock:
        return task_status.get(task_id, {}).copy()

def update_task_status(task_id, status, message=None, data=None, progress=None, step=None):
    with status_lock:
        if task_id not in task_status:
            task_status[task_id] = {}
        task_status[task_id]['status'] = status
        if message:
            task_status[task_id]['message'] = message
        if data:
            task_status[task_id]['data'] = data
        if progress is not None:
            task_status[task_id]['progress'] = progress
        if step:
            task_status[task_id]['step'] = step


def parse_retention_from_log(log_output):
    """
    Parses the 'Final Duration' retention percentage from the log output of main.py.
    """
    # Example log: "Final Duration: 46.49s / 51.57s (90.15%)"
    match_duration = re.search(r"Final Duration:.*?\((\d+\.\d+)%\)", log_output)
    if match_duration:
        return f"{match_duration.group(1)}%"
        
    return "N/A"


def find_processed_results(task_id):
    """
    Find and parse the processed results from the _processed directory.
    This version assumes a structure where each .wav file has a corresponding .normalized.txt file.
    """
    results = []
    processed_base_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id + "_processed")

    if not os.path.exists(processed_base_dir):
        logging.warning(f"Results directory not found for task {task_id}: {processed_base_dir}")
        return []

    all_speakers_data = {}

    # Level 1: Iterate through subdirectories named after original audio files
    audio_file_dirs = [d for d in os.listdir(processed_base_dir) if os.path.isdir(os.path.join(processed_base_dir, d))]
    for audio_dir_name in audio_file_dirs:
        audio_dir_path = os.path.join(processed_base_dir, audio_dir_name)

        # Level 2: Find speaker directories inside the audio-specific directory
        speaker_dirs = [d for d in os.listdir(audio_dir_path) if d.startswith('SPK_') and os.path.isdir(os.path.join(audio_dir_path, d))]
        for speaker_id in speaker_dirs:
            if speaker_id not in all_speakers_data:
                all_speakers_data[speaker_id] = {
                    "speaker_id": speaker_id,
                    "segments": []
                }

            speaker_path = os.path.join(audio_dir_path, speaker_id)
            
            # Level 3: Find all wav files and their corresponding text files
            wav_files = [f for f in os.listdir(speaker_path) if f.endswith(('.wav', '.mp3'))]
            
            for wav_filename in wav_files:
                base_name, _ = os.path.splitext(wav_filename)
                txt_filename = f"{base_name}.normalized.txt"
                txt_filepath = os.path.join(speaker_path, txt_filename)
                wav_filepath = os.path.join(speaker_path, wav_filename)

                if os.path.exists(txt_filepath):
                    try:
                        with open(txt_filepath, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        
                        if not text:
                            continue

                        # Copy file to be served statically
                        result_speaker_path = os.path.join(app.config['RESULTS_FOLDER'], task_id, speaker_id)
                        os.makedirs(result_speaker_path, exist_ok=True)
                        shutil.copy(wav_filepath, os.path.join(result_speaker_path, wav_filename))
                        
                        # Store components for URL generation, NOT the URL itself
                        all_speakers_data[speaker_id]["segments"].append({
                            "text": text,
                            "speaker_id": speaker_id,
                            "filename": wav_filename
                        })

                    except Exception as e:
                        logging.error(f"Failed to process file pair '{wav_filename}' and '{txt_filename}': {e}")

    # Apply preview limits (5 speakers, 5 segments)
    limited_results = []
    for i, speaker_id in enumerate(sorted(all_speakers_data.keys())):
        if i >= 5:
            break
        speaker_data = all_speakers_data[speaker_id]
        # Sort segments by filename to ensure consistent order, then take first 5
        speaker_data["segments"] = sorted(speaker_data["segments"], key=lambda x: x['filename'])[:5]
        if speaker_data["segments"]: # only add speaker if they have segments
             limited_results.append(speaker_data)
        
    return limited_results


def process_task(task_id, input_dir):
    """
    The actual worker function that runs the main.py script, with real-time progress tracking.
    This now calls the pipeline class directly.
    """
    if not pipeline_instance:
        logging.error(f"Task {task_id} cannot be processed because pipeline failed to load.")
        update_task_status(task_id, 'failed', '处理失败: 模型服务未启动。')
        return

    logging.info(f"Starting processing for task {task_id} in directory {input_dir}")
    
    # Callback for real-time progress updates
    def progress_callback(progress, step_name):
        update_task_status(
            task_id, 
            'processing',
            message=f"正在进行: {step_name}",
            progress=progress,
            step=step_name
        )

    try:
        # The main logic is now to iterate through files and process them
        audio_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
        all_results = []
        
        # We assume one audio file per task for simplicity in this version.
        if not audio_files:
            raise ValueError("No audio files found in the task directory.")

        # We assume one audio file per task for simplicity in this version.
        audio_path = audio_files[0]
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        # The base output directory (e.g., .../<task_id>_processed)
        output_base_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id + "_processed")
        
        # The final save path for this specific audio (e.g., .../<task_id>_processed/<audio_name>)
        final_save_path = os.path.join(output_base_dir, audio_name)
        
        _, _, stats = pipeline_instance.main_process(
            audio_path=audio_path,
            save_path=final_save_path, # Explicitly set the FINAL save path
            progress_callback=progress_callback
        )
        
        # After processing, parse the results from the directory
        update_task_status(task_id, 'processing', '正在解析结果...', progress=99, step="完成中...")
        
        final_results = find_processed_results(task_id)

        # Calculate retention rate from the returned stats object
        retention_rate_str = "N/A"
        if stats and stats['initial']['duration'] > 0:
            initial_duration = stats['initial']['duration']
            final_duration = stats['final']['duration']
            rate = (final_duration / initial_duration) * 100
            rate = min(100.0, rate) # Cap at 100%
            retention_rate_str = f"{rate:.2f}%"

        final_data = { "results": final_results, "retention_rate": retention_rate_str } 
        update_task_status(task_id, 'completed', '处理完成! ✅', data=final_data, progress=100)

    except Exception as e:
        logging.error(f"An exception occurred while processing task {task_id}: {e}", exc_info=True)
        update_task_status(task_id, 'failed', str(e))


def worker():
    """
    Worker thread to process tasks from the queue one by one.
    """
    while True:
        task_id, input_dir = task_queue.get()
        if task_id is None:
            break
        process_task(task_id, input_dir)
        task_queue.task_done()


# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads."""
    task_id = str(uuid.uuid4())
    task_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    os.makedirs(task_dir, exist_ok=True)
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400

    total_size = 0
    for file in files:
        # Check file size (request.content_length is for the whole request)
        file.seek(0, os.SEEK_END)
        total_size += file.tell()
        file.seek(0)
    
    if total_size > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({"error": f"Total file size exceeds the limit of 1GB."}), 413

    for file in files:
        filename = file.filename
        file.save(os.path.join(task_dir, filename))
        
    logging.info(f"Files uploaded for new task {task_id} in {task_dir}")
    
    # Initialize task status
    update_task_status(task_id, 'uploaded', 'Files uploaded, waiting to be processed.')
    
    return jsonify({"task_id": task_id})


@app.route('/process/<task_id>', methods=['POST'])
def process_files(task_id):
    """Start processing the uploaded files for a given task."""
    input_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    if not os.path.exists(input_dir):
        return jsonify({"error": "Invalid task ID"}), 404

    current_status = get_task_status(task_id)
    if current_status.get('status') == 'processing':
        return jsonify({"message": "Task is already processing."}), 202

    task_queue.put((task_id, input_dir))
    update_task_status(task_id, 'queued', 'Task is queued and will be processed shortly.')
    logging.info(f"Task {task_id} added to the processing queue.")
    
    return jsonify({"message": "Task has been queued for processing."}), 202


@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """Get the current status of a task."""
    status = get_task_status(task_id)
    if not status:
        return jsonify({"error": "Task not found"}), 404

    # If completed, generate URLs now, within the app context
    if status.get('status') == 'completed' and 'data' in status:
        try:
            for speaker_group in status['data'].get('results', []):
                for segment in speaker_group.get('segments', []):
                    segment['audio_url'] = url_for(
                        'get_result_file',
                        task_id=task_id,
                        speaker_id=segment['speaker_id'],
                        filename=segment['filename']
                    )
        except Exception as e:
            logging.error(f"Error generating URLs in get_status: {e}")
            # Potentially corrupt the data to prevent frontend errors
            status['data']['results'] = []
            status['message'] = "Error preparing results for display."


    return jsonify(status)


@app.route('/results/<task_id>/<speaker_id>/<filename>')
def get_result_file(task_id, speaker_id, filename):
    """Serve a single processed audio file."""
    directory = os.path.join(app.config['RESULTS_FOLDER'], task_id, speaker_id)
    return send_from_directory(directory, filename)


@app.route('/download/<task_id>', methods=['GET'])
def download_results(task_id):
    """Compress and download all results for a task."""
    processed_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id + "_processed")
    if not os.path.exists(processed_dir):
        return "Results not found or have been cleaned up.", 404
        
    zip_path = os.path.join(app.config['RESULTS_FOLDER'], f"{task_id}_results")
    shutil.make_archive(zip_path, 'zip', processed_dir)
    
    logging.info(f"Created zip archive for task {task_id} at {zip_path}.zip")

    return send_from_directory(
        app.config['RESULTS_FOLDER'],
        f"{task_id}_results.zip",
        as_attachment=True
    )


if __name__ == '__main__':
    # Load the models first, before starting the server
    load_pipeline()

    # Start the background worker thread
    worker_thread = Thread(target=worker)
    worker_thread.daemon = True
    worker_thread.start()
    
    # Run Flask app
    # Use host='0.0.0.0' to make it accessible on the local network
    app.run(host='0.0.0.0', port=5000, debug=False)
