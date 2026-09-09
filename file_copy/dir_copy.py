#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dir_copy.py — 目录全量拷贝工具

功能:
  - 递归拷贝源目录全部内容到目标目录(保留目录结构/权限/时间戳)
  - 默认并发 1(同一时刻只拷贝一个文件), --workers 可调大
  - 拷贝完成后全量校验(--verify size|hash), 不一致自动重拷并二次校验
  - 断点续传: 依据状态文件跳过已成功拷贝且未变化的文件
  - 全部状态文件与日志文件统一保存在执行目录(运行时工作目录)下 log/ 目录

用法:
  python dir_copy.py <src> <dst> [--workers N] [--verify size|hash] [--retries N]

示例:
  # 串行拷贝 + size/mtime 校验(默认)
  python dir_copy.py /data/src /data/dst

  # 4 并发 + MD5 全量校验
  python dir_copy.py /data/src /data/dst --workers 4 --verify hash

  # 中断后再次执行同一条命令即可断点续传
  python dir_copy.py /data/src /data/dst --workers 4 --verify hash

退出码: 0 成功; 1 存在失败文件; 2 参数/环境错误; 130 被用户中断

环境: Python 3.7+ 标准库实现, 零第三方依赖
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CHUNK_SIZE = 1024 * 1024          # 分块拷贝大小 1MB
TMP_SUFFIX = ".dircopy.tmp"       # 临时文件后缀, 原子 rename 前的中间名
MTIME_TOLERANCE = 2.0             # mtime 比对容差(秒), 兼容不同文件系统时间精度
STATE_VERSION = 1
STATE_SAVE_INTERVAL = 1.0         # 状态文件节流落盘间隔(秒)
RETRY_BASE_DELAY = 1.0            # 失败重试基础退避(秒), 指数递增
RETRY_MAX_DELAY = 30.0

logger = logging.getLogger("dir_copy")


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class FileTask:
    """一个待拷贝文件的任务描述(均为源文件信息)."""
    rel: str        # 相对源目录的路径, 使用 '/' 分隔
    src: str        # 源文件绝对路径
    size: int
    mtime: float


# --------------------------------------------------------------------------- #
# 断点续传状态管理
# --------------------------------------------------------------------------- #
class StateStore:
    """JSON 状态文件: 记录已成功拷贝的文件, 支持断点续传.

    结构:
    {
      "version": 1,
      "src": "/abs/src",
      "dst": "/abs/dst",
      "files": {
        "rel/path/file.wav": {"size": 123, "mtime": 1690000000.0, "md5": "..."}
      }
    }
    """

    def __init__(self, path: str, src: str, dst: str):
        self.path = path
        self._lock = threading.Lock()
        self._last_save = 0.0
        self.data = {"version": STATE_VERSION, "src": src, "dst": dst, "files": {}}

    @classmethod
    def load(cls, path: str, src: str, dst: str) -> "StateStore":
        store = cls(path, src, dst)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("files"), dict):
                    store.data["files"] = data["files"]
                    logger.info("加载断点状态文件: %s (已完成 %d 个文件)",
                                path, len(data["files"]))
                else:
                    logger.warning("状态文件结构异常, 忽略并重新开始: %s", path)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("状态文件读取失败(%s), 忽略并重新开始: %s", e, path)
        else:
            logger.info("未发现断点状态文件, 全新任务: %s", path)
        return store

    def is_done(self, task: FileTask) -> bool:
        """源文件与已完成记录一致(size+mtime)则视为已完成, 可跳过."""
        rec = self.data["files"].get(task.rel)
        if not rec:
            return False
        return (rec.get("size") == task.size
                and abs(rec.get("mtime", 0.0) - task.mtime) <= MTIME_TOLERANCE)

    def record(self, rel: str, size: int, mtime: float, md5: Optional[str]) -> None:
        with self._lock:
            rec = {"size": size, "mtime": mtime}
            if md5:
                rec["md5"] = md5
            self.data["files"][rel] = rec

    def maybe_save(self, force: bool = False) -> None:
        """节流落盘(写临时文件+原子 rename, 防状态文件本身损坏)."""
        with self._lock:
            now = time.monotonic()
            if not force and (now - self._last_save) < STATE_SAVE_INTERVAL:
                return
            self._last_save = now
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False)
            os.replace(tmp, self.path)

    @property
    def done_count(self) -> int:
        return len(self.data["files"])


# --------------------------------------------------------------------------- #
# 目录扫描
# --------------------------------------------------------------------------- #
def scan_source(src: str, skip_dirs: List[str]) -> Tuple[List[FileTask], List[str], int]:
    """递归扫描源目录.

    Returns:
        (文件任务清单, 需保证存在的目录相对路径清单, 源文件总字节数)
    """
    tasks: List[FileTask] = []
    dirs: List[str] = []
    total_bytes = 0

    skip_abs = {os.path.abspath(d) for d in skip_dirs}

    def on_walk_error(err: OSError) -> None:
        logger.warning("扫描目录失败: %s (%s)", err.filename, err.strerror)

    for root, subdirs, files in os.walk(src, onerror=on_walk_error):
        root_abs = os.path.abspath(root)
        # 跳过工具自身的 log 目录(仅当执行目录在源目录内时会出现), 避免递归拷贝自身
        if root_abs in skip_abs:
            subdirs[:] = []
            continue
        # 跳过嵌套在源目录内的 log 目录
        subdirs[:] = [d for d in subdirs
                      if os.path.abspath(os.path.join(root, d)) not in skip_abs]

        rel_root = os.path.relpath(root, src)
        if rel_root != ".":
            dirs.append(rel_root.replace(os.sep, "/"))

        for name in files:
            full = os.path.join(root, name)
            if os.path.islink(full) and not os.path.exists(full):
                logger.warning("跳过失效的符号链接: %s", full)
                continue
            if not os.path.isfile(full):
                logger.warning("跳过非常规文件: %s", full)
                continue
            rel = os.path.join(rel_root, name).replace(os.sep, "/") \
                if rel_root != "." else name
            st = os.stat(full)
            tasks.append(FileTask(rel=rel, src=full,
                                  size=st.st_size, mtime=st.st_mtime))
            total_bytes += st.st_size

        for d in subdirs:
            dpath = os.path.join(root, d)
            if os.path.islink(dpath):
                logger.warning("跳过符号链接目录(不跟随): %s", dpath)

    return tasks, dirs, total_bytes


def cleanup_tmp_files(dst: str) -> int:
    """清理目标目录内上次中断残留的 *.dircopy.tmp 临时文件."""
    removed = 0
    for root, _dirs, files in os.walk(dst):
        for name in files:
            if name.endswith(TMP_SUFFIX):
                try:
                    os.remove(os.path.join(root, name))
                    removed += 1
                except OSError as e:
                    logger.warning("清理临时文件失败: %s (%s)", name, e)
    if removed:
        logger.info("已清理上次中断残留的临时文件 %d 个", removed)
    return removed


# --------------------------------------------------------------------------- #
# 拷贝引擎
# --------------------------------------------------------------------------- #
def md5_of_file(path: str) -> str:
    """流式计算文件 MD5, 内存占用恒定."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def copy_file(task: FileTask, dst: str, want_md5: bool) -> Optional[str]:
    """分块拷贝单个文件: 写临时名 -> 保留元数据 -> 原子 rename.

    Returns: want_md5 为 True 时返回源文件 MD5, 否则返回 None.
    """
    dst_path = os.path.join(dst, task.rel)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = dst_path + TMP_SUFFIX

    h = hashlib.md5() if want_md5 else None
    try:
        with open(task.src, "rb") as fsrc, open(tmp_path, "wb") as fdst:
            while True:
                chunk = fsrc.read(CHUNK_SIZE)
                if not chunk:
                    break
                fdst.write(chunk)
                if h is not None:
                    h.update(chunk)
        # 保留权限与时间戳后再原子改名, 保证状态记录的文件一定完整
        shutil.copystat(task.src, tmp_path)
        os.replace(tmp_path, dst_path)
    except BaseException:
        # 任何异常(含中断)都清理临时文件, 避免残留半成品
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise
    return h.hexdigest() if h is not None else None


def copy_with_retry(task: FileTask, dst: str, retries: int,
                    want_md5: bool) -> Tuple[FileTask, Optional[Exception], Optional[str], float]:
    """带指数退避重试的单文件拷贝.

    Returns: (task, 最终异常(None 表示成功), 源文件 MD5(可选), 耗时秒)
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            t0 = time.monotonic()
            md5 = copy_file(task, dst, want_md5)
            return task, None, md5, time.monotonic() - t0
        except Exception as e:  # noqa: BLE001 拷贝阶段需要兜住所有 IO 异常
            last_err = e
            logger.warning("拷贝失败(第 %d/%d 次): %s (%s)",
                           attempt, retries, task.rel, e)
            if attempt < retries:
                delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
                time.sleep(delay)
    return task, last_err, None, 0.0


def run_copy(pending: List[FileTask], dst: str, workers: int, retries: int,
             want_md5: bool, state: StateStore) -> Tuple[List[FileTask], int]:
    """并发执行拷贝任务, 主线程串行更新状态并节流落盘.

    Returns: (失败任务清单, 已拷贝总字节数)
    """
    failed: List[FileTask] = []
    copied_bytes = 0
    done = 0
    total = len(pending)
    t_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(copy_with_retry, t, dst, retries, want_md5): t
                   for t in pending}
        try:
            for fut in as_completed(futures):
                task, err, md5, dur = fut.result()
                done += 1
                if err is None:
                    copied_bytes += task.size
                    state.record(task.rel, task.size, task.mtime, md5)
                    logger.info("[%d/%d] 拷贝成功: %s (%d bytes, %.2fs)",
                                done, total, task.rel, task.size, dur)
                else:
                    failed.append(task)
                    logger.error("[%d/%d] 拷贝最终失败: %s (%s)",
                                 done, total, task.rel, err)
                state.maybe_save()
        except KeyboardInterrupt:
            logger.warning("收到中断信号, 正在保存断点状态...")
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:  # Python < 3.9 无 cancel_futures 参数
                pool.shutdown(wait=False)
            raise
        finally:
            state.maybe_save(force=True)

    if total:
        elapsed = time.monotonic() - t_start
        rate = copied_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0.0
        logger.info("拷贝阶段完成: 计划 %d, 成功 %d, 失败 %d, "
                    "共 %.2f MB, 用时 %.1fs (平均 %.2f MB/s)",
                    total, total - len(failed), len(failed),
                    copied_bytes / 1024 / 1024, elapsed, rate)
    return failed, copied_bytes


# --------------------------------------------------------------------------- #
# 全量校验
# --------------------------------------------------------------------------- #
def verify_all(src: str, dst: str, mode: str,
               skip_dirs: List[str]) -> Tuple[List[str], List[str], int]:
    """拷贝完成后全量比对源与目标.

    Returns: (不一致文件相对路径清单, 目标多出的文件清单, 源文件总数)
    """
    skip_abs = {os.path.abspath(d) for d in skip_dirs}
    src_map: Dict[str, Tuple[int, float, str]] = {}
    dst_map: Dict[str, Tuple[int, float]] = {}

    for base in (src, dst):
        for root, subdirs, files in os.walk(base):
            root_abs = os.path.abspath(root)
            if root_abs in skip_abs:
                subdirs[:] = []
                continue
            rel_root = os.path.relpath(root, base)
            for name in files:
                if name.endswith(TMP_SUFFIX):
                    continue
                full = os.path.join(root, name)
                rel = (os.path.join(rel_root, name) if rel_root != "."
                       else name).replace(os.sep, "/")
                st = os.stat(full)
                if base == src:
                    src_map[rel] = (st.st_size, st.st_mtime, full)
                else:
                    dst_map[rel] = (st.st_size, st.st_mtime)

    mismatch: List[str] = []
    for rel, (size, mtime, sfull) in src_map.items():
        dpath = os.path.join(dst, rel)
        drec = dst_map.get(rel)
        if drec is None or not os.path.isfile(dpath):
            mismatch.append(rel)
            continue
        if drec[0] != size or abs(drec[1] - mtime) > MTIME_TOLERANCE:
            mismatch.append(rel)
            continue
        if mode == "hash":
            if md5_of_file(sfull) != md5_of_file(dpath):
                mismatch.append(rel)

    extra = sorted(set(dst_map) - set(src_map))
    logger.info("全量校验完成: 源文件 %d 个, 不一致 %d 个, 目标多出 %d 个",
                len(src_map), len(mismatch), len(extra))
    for rel in mismatch:
        logger.error("校验不一致: %s", rel)
    for rel in extra:
        logger.warning("目标目录多出(源中不存在, 保留未动): %s", rel)
    return mismatch, extra, len(src_map)


# --------------------------------------------------------------------------- #
# 日志
# --------------------------------------------------------------------------- #
def setup_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, "copy_{}.log".format(time.strftime("%Y%m%d_%H%M%S")))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return log_file


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def paths_overlap(a: str, b: str) -> bool:
    """判断两个目录路径相同或存在包含关系."""
    pa, pb = Path(a).resolve(), Path(b).resolve()
    return pa == pb or pa in pb.parents or pb in pa.parents


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dir_copy.py",
        description="目录全量拷贝工具: 递归拷贝 + 并发控制 + 全量校验 + 断点续传",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("src", help="源目录")
    parser.add_argument("dst", help="目标目录(不存在会自动创建)")
    parser.add_argument("--workers", type=int, default=1,
                        help="并发拷贝的文件数")
    parser.add_argument("--verify", choices=["size", "hash"], default="size",
                        help="全量校验方式: size=大小+mtime, hash=逐文件 MD5")
    parser.add_argument("--retries", type=int, default=3,
                        help="单文件失败重试次数")
    return parser.parse_args(argv)


def validate(args: argparse.Namespace, src: str, dst: str) -> Optional[int]:
    if not os.path.isdir(src):
        logger.error("源目录不存在或不是目录: %s", src)
        return 2
    if paths_overlap(src, dst):
        logger.error("源目录与目标目录相同或存在包含关系, 拒绝执行: %s vs %s", src, dst)
        return 2
    if args.workers < 1:
        logger.error("--workers 必须 >= 1, 当前: %d", args.workers)
        return 2
    if args.retries < 1:
        logger.error("--retries 必须 >= 1, 当前: %d", args.retries)
        return 2
    return None


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)

    # 状态与日志统一保存在执行目录(工作目录)下 log/ 目录
    log_dir = os.path.join(os.getcwd(), "log")
    log_file = setup_logging(log_dir)

    rc = validate(args, src, dst)
    if rc is not None:
        return rc

    logger.info("=" * 64)
    logger.info("目录拷贝任务开始")
    logger.info("源目录: %s", src)
    logger.info("目标目录: %s", dst)
    logger.info("参数: workers=%d, verify=%s, retries=%d",
                args.workers, args.verify, args.retries)
    logger.info("日志文件: %s", log_file)

    t_begin = time.monotonic()

    # 断点状态文件(按 源->目标 签名命名, 同一任务多次运行天然复用/续传)
    sig = hashlib.md5("{}->{}".format(src, dst).encode("utf-8")).hexdigest()[:16]
    state_path = os.path.join(log_dir, "state_{}.json".format(sig))
    state: Optional[StateStore] = None

    try:
        # 1. 加载断点状态
        state = StateStore.load(state_path, src, dst)

        # 2. 预扫描 + 准备目标目录结构 + 清理上次残留临时文件
        tasks, dirs, total_bytes = scan_source(src, skip_dirs=[log_dir])
        logger.info("预扫描完成: 文件 %d 个, 目录 %d 个, 总计 %.2f MB",
                    len(tasks), len(dirs), total_bytes / 1024 / 1024)
        os.makedirs(dst, exist_ok=True)
        for d in dirs:
            os.makedirs(os.path.join(dst, d), exist_ok=True)
        cleanup_tmp_files(dst)

        # 3. 依据状态文件计算待拷贝清单(断点续传)
        pending = [t for t in tasks if not state.is_done(t)]
        skipped = len(tasks) - len(pending)
        if skipped:
            logger.info("断点续传: 跳过已完成且未变化的文件 %d 个, 待拷贝 %d 个",
                        skipped, len(pending))
        else:
            logger.info("待拷贝文件 %d 个", len(pending))

        # 4. 并发拷贝
        want_md5 = args.verify == "hash"
        failed, copied_bytes = run_copy(pending, dst, args.workers,
                                        args.retries, want_md5, state)
        logger.info("断点状态已保存: %s (累计完成 %d 个文件)",
                    state_path, state.done_count)

        # 5. 全量校验, 不一致自动重拷并二次校验
        mismatch, _extra, src_count = verify_all(src, dst, args.verify,
                                                 skip_dirs=[log_dir])
        if mismatch:
            logger.warning("发现 %d 个不一致文件, 自动重拷...", len(mismatch))
            fix_tasks = [t for t in tasks if t.rel in set(mismatch)]
            # 先从状态中移除, 再重拷, 防止状态残留误判
            for t in fix_tasks:
                state.data["files"].pop(t.rel, None)
            failed_fix, _ = run_copy(fix_tasks, dst, args.workers,
                                     args.retries, want_md5, state)
            failed.extend(failed_fix)
            mismatch2, _, _ = verify_all(src, dst, args.verify,
                                         skip_dirs=[log_dir])
            if mismatch2:
                logger.error("二次校验仍有 %d 个不一致: %s",
                             len(mismatch2), ", ".join(mismatch2[:20]))
                failed = failed + [t for t in tasks if t.rel in set(mismatch2)]
            else:
                logger.info("重拷后二次校验通过, 不一致文件已全部修复")
                failed = [t for t in failed if t.rel not in set(mismatch)]

        # 去重失败清单(重拷修复阶段可能与拷贝阶段重复记录同一文件)
        seen_rel = set()
        failed = [t for t in failed
                  if not (t.rel in seen_rel or seen_rel.add(t.rel))]

        # 6. 汇总报告
        elapsed = time.monotonic() - t_begin
        logger.info("=" * 64)
        logger.info("任务汇总: 源文件 %d 个 | 本次拷贝 %d 个 | 续传跳过 %d 个 | "
                    "失败 %d 个 | 拷贝 %.2f MB | 总耗时 %.1fs",
                    src_count, len(pending) - len(failed), skipped, len(failed),
                    copied_bytes / 1024 / 1024, elapsed)
        logger.info("状态文件: %s", state_path)
        logger.info("日志目录: %s", log_dir)
        if failed:
            logger.error("失败文件清单:")
            for t in failed:
                logger.error("  - %s", t.rel)
            logger.error("任务存在失败文件, 可直接重跑同一命令进行断点续传补齐")
            return 1
        if src_count != state.done_count:
            logger.warning("注意: 源文件总数 %d 与状态记录 %d 不一致, "
                           "可能有文件在拷贝期间发生变化", src_count, state.done_count)
        logger.info("目录拷贝任务成功完成")
        return 0

    except KeyboardInterrupt:
        if state is not None:
            state.maybe_save(force=True)
            logger.warning("任务被中断, 断点状态已保存(累计完成 %d 个文件), "
                           "重跑同一命令可续传", state.done_count)
            logger.warning("状态文件: %s", state_path)
        else:
            logger.warning("任务被中断(尚未开始拷贝, 无状态需保存)")
        return 130
    except Exception as e:  # noqa: BLE001 顶层兜底, 保证有日志可查
        logger.exception("任务异常终止: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
