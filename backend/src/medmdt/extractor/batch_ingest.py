# src/medmdt/extractor/batch_ingest.py
"""Batch ingestion from archive files (zip/tar/rar/7z).

Archives are expected to contain top-level folders, each representing
one patient's case. Files at the archive root (not in a folder) are ignored.
Each patient folder is recursively scanned for supported medical files.
"""
import os
import shutil
import subprocess
import tempfile
import zipfile
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm", ".dicom"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".rar", ".7z"}
MAX_EXTRACTED_SIZE = 2 * 1024 * 1024 * 1024  # 2GB bomb protection


@dataclass
class FolderResult:
    folder_name: str
    status: str = "pending"  # pending | completed | failed | skipped
    files_processed: int = 0
    files_failed: int = 0
    message: str = ""
    skipped_files: list[str] = field(default_factory=list)


def extract_archive(archive_path: str, dest_dir: str) -> None:
    """Extract an archive to dest_dir. Supports zip, tar variants, rar, 7z."""
    path = Path(archive_path)
    suffix = path.suffix.lower()
    name_lower = path.name.lower()

    if suffix == ".zip":
        _extract_zip(archive_path, dest_dir)
    elif suffix in (".tar", ".gz", ".tgz", ".bz2") or name_lower.endswith((".tar.gz", ".tar.bz2")):
        _extract_tar(archive_path, dest_dir)
    elif suffix == ".rar":
        _extract_rar(archive_path, dest_dir)
    elif suffix == ".7z":
        _extract_7z(archive_path, dest_dir)
    else:
        raise ValueError(f"不支持的压缩格式: {suffix}")


def _extract_zip(archive_path: str, dest_dir: str) -> None:
    with zipfile.ZipFile(archive_path, "r") as zf:
        total_size = sum(info.file_size for info in zf.infolist())
        if total_size > MAX_EXTRACTED_SIZE:
            raise ValueError(f"压缩包解压后超过2GB限制（{total_size / 1e9:.1f}GB）")
        zf.extractall(dest_dir)


def _extract_tar(archive_path: str, dest_dir: str) -> None:
    with tarfile.open(archive_path, "r:*") as tf:
        members = tf.getmembers()
        total_size = sum(m.size for m in members if m.isfile())
        if total_size > MAX_EXTRACTED_SIZE:
            raise ValueError(f"压缩包解压后超过2GB限制（{total_size / 1e9:.1f}GB）")
        tf.extractall(dest_dir, filter="data")


def _extract_rar(archive_path: str, dest_dir: str) -> None:
    try:
        subprocess.run(
            ["unrar", "x", "-o+", archive_path, dest_dir + "/"],
            check=True,
            capture_output=True,
            timeout=300,
        )
    except FileNotFoundError:
        raise ValueError("系统未安装 unrar，无法处理 .rar 文件。请运行: brew install unrar")
    except subprocess.CalledProcessError as e:
        raise ValueError(f"解压 .rar 失败: {e.stderr.decode()[:200]}")


def _extract_7z(archive_path: str, dest_dir: str) -> None:
    try:
        import py7zr
    except ImportError:
        raise ValueError("未安装 py7zr 库，无法处理 .7z 文件。请运行: pip install py7zr")
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extractall(path=dest_dir)


def _collect_supported_files(folder: Path) -> list[Path]:
    """Recursively collect all supported medical files in a folder."""
    files = []
    for f in folder.rglob("*"):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(f)
    return sorted(files)


def process_archive(
    archive_path: str,
    agent,
    on_folder_done=None,
) -> list[FolderResult]:
    """Process an archive: extract, walk root-level folders, ingest each.

    Args:
        archive_path: Path to the archive file.
        agent: ExtractionAgent instance.
        on_folder_done: Optional callback(folder_index, folder_result) for progress updates.

    Returns:
        List of FolderResult, one per root-level folder.
    """
    tmp_dir = tempfile.mkdtemp(prefix="medmdt_batch_")
    results: list[FolderResult] = []

    try:
        extract_archive(archive_path, tmp_dir)

        root = Path(tmp_dir)
        # Handle case where archive has a single wrapper directory
        entries = [e for e in root.iterdir()]
        if len(entries) == 1 and entries[0].is_dir():
            # Single top-level dir wrapping everything — look inside it
            top_dirs = [e for e in entries[0].iterdir() if e.is_dir()]
            if top_dirs:
                root = entries[0]

        folders = sorted([e for e in root.iterdir() if e.is_dir()])

        for idx, folder in enumerate(folders):
            result = _process_single_folder(folder, agent)
            results.append(result)
            if on_folder_done:
                on_folder_done(idx, result)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


def _process_single_folder(folder: Path, agent) -> FolderResult:
    """Process all supported files in one patient folder."""
    folder_name = folder.name
    supported_files = _collect_supported_files(folder)

    if not supported_files:
        all_files = list(folder.rglob("*"))
        file_list = [f.name for f in all_files if f.is_file()]
        unsupported = [f for f in file_list if Path(f).suffix.lower() not in SUPPORTED_EXTENSIONS]
        msg = "文件夹内无可处理文件"
        if unsupported:
            exts = set(Path(f).suffix.lower() for f in unsupported)
            msg += f"（含不支持的文件类型: {', '.join(sorted(exts))}）"
        return FolderResult(
            folder_name=folder_name,
            status="skipped",
            message=msg,
            skipped_files=unsupported[:10],
        )

    processed = 0
    failed = 0
    for f in supported_files:
        try:
            agent.process_file(str(f))
            processed += 1
        except Exception:
            failed += 1

    status = "completed" if failed == 0 else ("failed" if processed == 0 else "completed")
    msg = f"处理完成: {processed} 个文件成功"
    if failed > 0:
        msg += f", {failed} 个文件失败"

    return FolderResult(
        folder_name=folder_name,
        status=status,
        files_processed=processed,
        files_failed=failed,
        message=msg,
    )
