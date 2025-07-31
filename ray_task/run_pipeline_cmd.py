import os
import subprocess
from pathlib import Path
from ray_task.config import REPORT_PATH

def run_audio_preprocess_pipeline(input_audio_path, output_dir, task_key):
    try:
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)
        
        cmd = [
            "conda", "run", "-n", "AudioPipeline",
            "python", 
            "main.py",
            "--input_audio_path", str(input_audio_path),
            "--output_folder", str(output_dir),
            "--report_path", REPORT_PATH,
            "--threads", "4",
            "--num_workers", "1"
        ]
        
        result = subprocess.run(
            cmd, 
            cwd=project_root,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            status = "SUCCESS"
        else:
            status = "FAILED"
        
        return {
            "task_key": task_key,
            "status": status,
            "return_code": result.returncode,
            "cmd": " ".join(cmd)
        }
    except Exception as e:
        return {
            "task_key": task_key,
            "status": "EXCEPTION",
            "return_code": -1,
            "cmd": " ".join(["python", "main.py", "--input_audio_path", str(input_audio_path), "--output_folder", str(output_dir)]),
        }