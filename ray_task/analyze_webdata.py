import argparse
import json
import logging
import multiprocessing
import os
import sys
from functools import lru_cache
from pathlib import Path
import time
import subprocess 
import json as standard_json 

from tqdm import tqdm


_GLOBAL_CONFIGS = {}
_GLOBAL_MAPPINGS = []

def init_worker(base_dir_str, path_mappings_serialized):
    global _GLOBAL_CONFIGS, _GLOBAL_MAPPINGS
    _GLOBAL_CONFIGS['base_dir'] = Path(base_dir_str)
    _GLOBAL_MAPPINGS = [
        (Path(old), Path(new)) for old, new in path_mappings_serialized
    ]
    

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('audio_analyzer')


@lru_cache(maxsize=100000) 
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

@lru_cache(maxsize=100000)
def apply_path_mappings_cached(path_str: str, mappings_tuple):
    path = Path(path_str)
    mappings = [(Path(old), Path(new)) for old, new in mappings_tuple]
    
    for old_prefix, new_prefix in mappings:
        try:
            if path.is_relative_to(old_prefix):
                return str(new_prefix / path.relative_to(old_prefix))
        except ValueError:
            continue
    return str(path)


def get_audio_files(folder_path: str, skip_dirs: set = None):
    audio_extensions = ('.mp3', '.wav', '.flac', '.m4a', '.aac', '.mp4', '.ogg', '.webm')
    if skip_dirs is None:
        skip_dirs = set()
    audio_files = []
    scanned_dirs = 0

    for root, dirs, files in os.walk(folder_path, followlinks=False):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
        scanned_dirs += 1
        if scanned_dirs % 100 == 0:
            logger.info(f"已扫描 {scanned_dirs} 个目录，找到 {len(audio_files)} 个音频文件...")

        for name in files:
            if name.startswith(('.', '~', '._')) or '.temp' in name:
                continue
            if not name.lower().endswith(audio_extensions):
                continue
            full_path = os.path.join(root, name)
            try:
                size = os.stat(full_path).st_size
            except OSError:
                continue
            if size < 1024:
                continue
            audio_files.append(full_path)

    return audio_files

# --- 性能关键点：音频验证函数 (使用 ffprobe 加速) ---

def validate_audio_pydub_fallback(file_path: str):
    """最后兜底：pydub（全量解码，最慢但兼容性最好）"""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        duration_ms = len(audio)
        if duration_ms <= 0 or duration_ms > 86400000:
            return False, None, "时长异常 (pydub)"
        return True, duration_ms, None
    except Exception as e:
        return False, None, f"解码失败 (pydub): {str(e)}"


def validate_audio_ffprobe(file_path: str):
    """第二级 fallback：ffprobe（可靠但慢）"""
    ffprobe_cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'json', file_path
    ]
    try:
        result = subprocess.run(
            ffprobe_cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, timeout=10
        )
        data = standard_json.loads(result.stdout)
        if 'format' not in data or 'duration' not in data['format']:
            return validate_audio_pydub_fallback(file_path)
        duration_s = float(data['format']['duration'])
        duration_ms = int(duration_s * 1000)
        if duration_ms <= 0 or duration_ms > 86400000:
            return False, None, "时长异常 (ffprobe)"
        return True, duration_ms, None
    except (subprocess.CalledProcessError, FileNotFoundError, standard_json.JSONDecodeError):
        return validate_audio_pydub_fallback(file_path)
    except subprocess.TimeoutExpired:
        return False, None, "ffprobe 超时"
    except Exception as e:
        return False, None, f"ffprobe 失败: {str(e)}"


def validate_audio(file_path: str):
    """
    混合策略：mutagen 优先 + ffprobe fallback + pydub 最终兜底。
    OGG/FLAC/MP4/WAV 等规范容器，mutagen 只读 header 的几 KB，
    耗时约 0.5-2ms，比 ffprobe 快 30-100 倍。
    """
    # 第一级：mutagen
    try:
        from mutagen import File as MutagenFile
        audio = MutagenFile(file_path)
        if audio is not None and audio.info is not None:
            length = getattr(audio.info, 'length', None)
            if length is not None and length > 0:
                duration_ms = int(length * 1000)
                if duration_ms > 86400000:
                    return False, None, "时长异常 (mutagen)"
                return True, duration_ms, None
    except Exception:
        pass

    # 第二级：ffprobe
    return validate_audio_ffprobe(file_path)


def process_file_optimized(file_path: str):
    file_path = Path(file_path)
    
    base_dir = _GLOBAL_CONFIGS.get('base_dir')
    mappings = _GLOBAL_MAPPINGS
    mappings_tuple = tuple(tuple(str(p) for p in m) for m in mappings) 

    if not base_dir:
        return {"status": "invalid", "path": str(file_path), "reason": "进程未正确初始化 (BaseDir缺失)"}

    try:
        relative_dir = file_path.parent.relative_to(base_dir)
        
        mapped_relative_path_str = apply_path_mappings_cached(str(base_dir / relative_dir), mappings_tuple)
        mapped_relative_path = Path(mapped_relative_path_str)
        
        if mappings:
            new_base = mappings[0][1]
            final_relative = mapped_relative_path.relative_to(new_base)
        else:
            final_relative = relative_dir
    except Exception as e:
        return {"status": "invalid", "path": str(file_path), "reason": f"路径计算失败: {e}"}
    
    new_file_path_str = apply_path_mappings_cached(str(file_path), mappings_tuple)

    file_size = os.path.getsize(str(file_path))

    is_valid, duration_ms, error = validate_audio(str(file_path))
    if not is_valid:
        return {"status": "invalid", "path": str(file_path), "reason": error}

    return {
        "status": "valid",
        "relative_path": str(final_relative).replace('\\', '/'),
        "audio_path": new_file_path_str,
        "audio_duration_second": int(duration_ms // 1000),
        "file_size_mb": round(file_size / (1024 * 1024), 2)
    }

def analyze_audio_files_parallel(base_path: str, base_dir: str, max_workers=None, path_mappings=None, skip_dirs: set = None):
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 64)

    logger.info(f"扫描音频文件（使用 {max_workers} 个进程）...")
    scan_start = time.time()
    audio_files = get_audio_files(base_path, skip_dirs)
    logger.info(f"扫描耗时 {time.time() - scan_start:.1f}s")

    if not audio_files:
        return {"error": "未找到音频文件"}

    total = len(audio_files)
    logger.info(f"找到 {total} 个候选文件")
    logger.info(f"开始并行处理（{max_workers} 进程，imap_unordered chunksize=500）...")

    path_mappings_serialized = [(str(old), str(new)) for old, new in path_mappings] if path_mappings else []

    valid_files = []
    invalid_files = []
    total_duration = 0
    process_start = time.time()

    with multiprocessing.Pool(
        processes=max_workers,
        initializer=init_worker,
        initargs=(base_dir, path_mappings_serialized,)
    ) as pool:
        results_iter = pool.imap_unordered(
            process_file_optimized,
            audio_files,
            chunksize=500
        )

        pbar = tqdm(
            results_iter,
            total=total,
            desc="处理音频文件",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            file=sys.stdout,
        )

        for i, result in enumerate(pbar, 1):
            if result["status"] == "valid":
                valid_files.append(result)
                total_duration += result["audio_duration_second"]
            else:
                invalid_files.append(result)

            if i % 10000 == 0:
                elapsed = time.time() - process_start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                logger.info(
                    f"进度 {i}/{total} ({100*i/total:.1f}%), "
                    f"valid={len(valid_files)}, invalid={len(invalid_files)}, "
                    f"速率={rate:.0f} files/s, 已耗时={elapsed:.0f}s, 预计剩余={eta:.0f}s"
                )

    logger.info(f"处理总耗时 {time.time() - process_start:.1f}s")

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
                       default="/apdcephfs/tts_common/DATA/webdata/audiobooks/有声小说1",
                       help='当前音频数据目录')
    parser.add_argument('--base_dir', type=str,
                       default="/apdcephfs/tts_common/DATA/webdata",
                       help='音频根目录')
    parser.add_argument('--output', type=str, default='/apdcephfs/tts_common/DATA/webdata/audiobooks/有声小说1/data_list.json',
                       help='输出 JSON 文件')
    parser.add_argument('--max-workers', type=int, default=None)
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--path-mapping', type=str, default='/apdcephfs/tts_common/DATA/webdata=/cfs/cfs-czb184s7/DATA/webdata',
                       help='路径映射，如: /old=/new')
    parser.add_argument('--skip-dirs', type=str, default='',
                       help='要跳过的目录名，用逗号分隔，如: dir1,dir2,dir3')

    args = parser.parse_args()
    logger.setLevel(args.log_level)

    # 预检输出路径可写
    output_path = Path(args.output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        test_file = output_path.parent / '.write_test'
        test_file.touch()
        test_file.unlink()
        logger.info(f"✅ 输出路径可写: {output_path.parent}")
    except Exception as e:
        logger.error(f"❌ 输出路径不可写: {output_path.parent}, 错误: {e}")
        return

    base_path = Path(args.path).resolve()
    if not base_path.exists():
        logger.error(f"错误: 路径不存在 {base_path}")
        return
    
    base_dir = Path(args.base_dir).resolve()

    try:
        from pydub import AudioSegment
        logger.info("✅ pydub 已安装")
    except ImportError:
        logger.error("❌ 请安装 pydub: pip install pydub")
        return

    try:
        subprocess.run(['ffprobe', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        logger.info("✅ ffmpeg/ffprobe 可用")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("❌ 请安装 ffmpeg/ffprobe（pydub 和加速依赖）")
        return

    try:
        path_mappings = parse_path_mappings(args.path_mapping)
    except ValueError as e:
        logger.error(f"路径映射错误: {e}")
        return

    # 解析要跳过的目录
    skip_dirs = set()
    if args.skip_dirs:
        skip_dirs = {d.strip() for d in args.skip_dirs.split(',') if d.strip()}
        logger.info(f"跳过的目录: {skip_dirs}")

    start_time = time.time()
    result = analyze_audio_files_parallel(str(base_path), str(base_dir), args.max_workers, path_mappings, skip_dirs)
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
