import argparse
import json
import logging
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('audio_analyzer')


@lru_cache(maxsize=10000)
def get_file_size_cached(file_path: str) -> int:
    try:
        return os.path.getsize(file_path)
    except (OSError, FileNotFoundError):
        return 0


def parse_path_mappings(mapping_args):
    mappings = []
    mappings_list = mapping_args.split(',')
    for item in mappings_list or []:
        if '=' not in item:
            raise ValueError(f"Invalid path mapping: {item}. Expected 'old=new'")
        old, new = item.split('=', 1)
        mappings.append((Path(old).resolve(), Path(new).resolve()))
    return mappings


def apply_path_mappings(path: Path, mappings):
    for old_prefix, new_prefix in mappings:
        try:
            if path.is_relative_to(old_prefix):
                return new_prefix / path.relative_to(old_prefix)
        except ValueError:
            continue
    return path


def get_audio_files(folder_path: str):
    audio_extensions = ('.mp3', '.wav', '.flac', '.m4a', '.aac', '.mp4', '.ogg')
    skip_dirs = {'_processed', '.temp', 'temp', '.git', '__pycache__'}
    audio_files = []

    for root, dirs, files in os.walk(folder_path, followlinks=False):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]

        with os.scandir(root) as it:
            for entry in it:
                if not entry.is_file():
                    continue

                name = entry.name
                if name.startswith(('.', '~', '._')) or '.temp' in name:
                    continue

                if not entry.path.lower().endswith(audio_extensions):
                    continue

                size = get_file_size_cached(entry.path)
                if size < 1024:
                    continue

                audio_files.append(entry.path)

    return audio_files


def validate_audio_pydub(file_path: str):
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        duration_ms = len(audio)
        if duration_ms <= 0 or duration_ms > 86400000:  # >24小时
            return False, None, "时长异常"
        return True, duration_ms, None
    except Exception as e:
        return False, None, f"解码失败: {str(e)}"


def process_file(file_path: str, base_path: str, base_dir: str,  path_mappings_serialized):
    file_path = Path(file_path)
    base_path = Path(base_path)
    base_dir = Path(base_dir)

    mappings = [(Path(old), Path(new)) for old, new in path_mappings_serialized] if path_mappings_serialized else []
    try:
        relative_dir = file_path.parent.relative_to(base_dir)
        mapped_relative_path = apply_path_mappings(base_dir / relative_dir, mappings)
        if mappings:
            new_base = mappings[0][1]
            final_relative = mapped_relative_path.relative_to(new_base)
        else:
            final_relative = relative_dir
    except Exception as e:
        return {"status": "invalid", "path": str(file_path), "reason": f"路径计算失败: {e}"}
    
    new_file_path = apply_path_mappings(file_path, mappings)

    file_size = get_file_size_cached(str(file_path))
    if file_size < 1024:
        return {"status": "invalid", "path": str(file_path), "reason": "文件过小 (<1KB)"}

    is_valid, duration_ms, error = validate_audio_pydub(str(file_path))
    if not is_valid:
        return {"status": "invalid", "path": str(file_path), "reason": error}

    return {
        "status": "valid",
        "relative_path": str(final_relative).replace('\\', '/'),
        "audio_path": str(new_file_path),
        "audio_duration_second": int(duration_ms // 1000),
        "file_size_mb": round(file_size / (1024 * 1024), 2)
    }


def analyze_audio_files_parallel(base_path: str, base_dir: str, max_workers=None, path_mappings=None):
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 64)

    logger.info(f"扫描音频文件（使用 {max_workers} 个进程）...")
    audio_files = get_audio_files(base_path)

    if not audio_files:
        return {"error": "未找到音频文件"}

    logger.info(f"找到 {len(audio_files)} 个候选文件")
    logger.info(f"开始并行处理（{max_workers} 进程）...")

    path_mappings_serialized = [(str(old), str(new)) for old, new in path_mappings] if path_mappings else None

    valid_files = []
    invalid_files = []
    total_duration = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file, fp, base_path, base_dir, path_mappings_serialized): fp
            for fp in audio_files
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="处理音频文件",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        ):
            try:
                result = future.result()
            except Exception as e:
                fp = futures[future]
                result = {"status": "invalid", "path": fp, "reason": f"进程崩溃: {e}"}

            if result["status"] == "valid":
                valid_files.append(result)
                total_duration += result["audio_duration_second"]
            else:
                invalid_files.append(result)

    valid_files.sort(key=lambda x: x["audio_duration_second"])

    return {
        "audio_duration_second": int(total_duration),
        "audio_duration_hour": round(total_duration / 3600, 2),
        "valid_files_count": len(valid_files),
        "invalid_files_count": len(invalid_files),
        "podcast_data": valid_files
    }


def main():
    parser = argparse.ArgumentParser(description='音频分析工具（基于 pydub）')
    parser.add_argument('--path', type=str,
                       default="/apdcephfs/tts_common/DATA/webdata/audiobooks/有声小说7",
                       help='当前音频数据目录')
    parser.add_argument('--base_dir', type=str,
                       default="/apdcephfs/tts_common/DATA/webdata",
                       help='音频根目录')
    parser.add_argument('--output', type=str, default='/apdcephfs/tts_common/DATA/webdata/audiobooks/有声小说7/data_list.json',
                       help='输出 JSON 文件')
    parser.add_argument('--max-workers', type=int, default=None)
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--path-mapping', type=str, default='/apdcephfs/tts_common/DATA/webdata=/cfs/cfs-czb184s7/DATA/webdata',
                       help='路径映射，如: /old=/new')

    args = parser.parse_args()
    logger.setLevel(args.log_level)

    base_path = Path(args.path).resolve()
    if not base_path.exists():
        logger.error(f"错误: 路径不存在 {base_path}")
        return
    
    base_dir = Path(args.base_dir).resolve()

    # 检查 pydub 和 ffmpeg
    try:
        from pydub import AudioSegment
        logger.info("✅ pydub 已安装")
    except ImportError:
        logger.error("❌ 请安装 pydub: pip install pydub")
        return

    try:
        import subprocess
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logger.info("✅ ffmpeg 可用")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("❌ 请安装 ffmpeg（pydub 依赖）")
        logger.error("Ubuntu: sudo apt install ffmpeg | Windows: https://www.gyan.dev/ffmpeg/builds/")
        return

    try:
        path_mappings = parse_path_mappings(args.path_mapping)
    except ValueError as e:
        logger.error(f"路径映射错误: {e}")
        return

    import time
    start_time = time.time()
    result = analyze_audio_files_parallel(str(base_path), str(base_dir),  args.max_workers, path_mappings)
    end_time = time.time()
    result["processing_time_seconds"] = round(end_time - start_time, 2)

    # 保存结果
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 结果已保存到: {args.output}")
    except Exception as e:
        logger.error(f"保存失败: {e}")

    # 打印摘要
    if "error" not in result:
        logger.info("\n📊 分析摘要:")
        logger.info(f"⏱️  处理时间: {result['processing_time_seconds']} 秒")
        logger.info(f"✅ 有效文件: {result['valid_files_count']}")
        logger.info(f"❌ 无效文件: {result['invalid_files_count']}")
        logger.info(f"🕒 总时长: {result['audio_duration_hour']} 小时")
        if result['valid_files_count'] > 0:
            avg_min = (result['audio_duration_hour'] * 60) / result['valid_files_count']
            logger.info(f"📈 平均时长: {avg_min:.1f} 分钟")


if __name__ == "__main__":
    if os.name == 'nt':
        multiprocessing.freeze_support()
    main()
