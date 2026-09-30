"""Helper đọc/ghi file JSON an toàn — dùng chung cho api8018.py và admin_api.py.

Nguyên tắc (đáp ứng yêu cầu P8 tài liệu bàn giao):

1. KHÔNG bao giờ ghi đè file bằng {} khi parse lỗi — chuyển file hỏng sang
   <path>.corrupt.<ts>, tạo marker <path>.corrupt_pending, rồi raise
   JsonCorruptError. Các lần đọc sau (khi file chưa được admin tạo lại) vẫn
   raise, không tự tạo file rỗng.

2. Ghi file NGUYÊN TỬ: ghi file tạm cùng thư mục → flush + fsync → os.replace.
   os.replace là atomic trên POSIX nên tiến trình khác đọc luôn thấy file
   nguyên vẹn; nếu crash giữa chừng, file gốc còn nguyên.

3. Khóa liên tiến trình bằng fcntl.flock trên <path>.lock cho toàn bộ chu
   trình đọc-sửa-ghi. Khác với threading.Lock (chỉ khóa giữa các thread
   trong CÙNG tiến trình) — cần cả hai vì api8018 và admin_api là 2 process.
"""
from __future__ import annotations

import fcntl
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional, Union


class JsonCorruptError(Exception):
    """File tồn tại nhưng không parse được JSON. File đã được đổi tên sang
    *.corrupt.<ts> để admin phục hồi thủ công. KHÔNG ghi đè {}."""


@contextmanager
def _file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _corrupt_marker(path: Path) -> Path:
    return path.with_name(f"{path.name}.corrupt_pending")


def _rename_corrupt(path: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = path.with_name(f"{path.name}.corrupt.{ts}")
    i = 1
    while dest.exists():
        dest = path.with_name(f"{path.name}.corrupt.{ts}.{i}")
        i += 1
    path.rename(dest)
    _corrupt_marker(path).write_text(
        f"corrupt at {time.strftime('%Y-%m-%d %H:%M:%S')}\noriginal -> {dest.name}\n",
        encoding="utf-8",
    )
    return dest


def read_json_safe(path: Union[str, Path], default_factory: Callable[[], Any]) -> Any:
    path = Path(path)
    if not path.exists():
        # Nếu trước đó file đã từng bị hỏng (có marker) mà admin CHƯA tạo lại
        # → vẫn raise, KHÔNG tự tạo mới, tránh vô tình ghi đè dữ liệu rỗng.
        if _corrupt_marker(path).exists():
            raise JsonCorruptError(
                f"{path} đang chờ phục hồi thủ công (marker "
                f"{_corrupt_marker(path).name} tồn tại). Không tự tạo mới."
            )
        return default_factory()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Đọc thành công → xóa marker cũ (nếu có) — admin đã phục hồi.
        m = _corrupt_marker(path)
        if m.exists():
            m.unlink()
        return data
    except Exception as e:
        dest = _rename_corrupt(path)
        print(
            f"[json_store] LỖI parse {path.name} ({e}) — đã chuyển sang "
            f"{dest.name}. Sẽ KHÔNG ghi đè file gốc. Kiểm tra & phục hồi thủ công.",
            file=sys.stderr,
        )
        raise JsonCorruptError(str(path)) from e


def atomic_write_json(path: Union[str, Path], data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def update_json(
    path: Union[str, Path],
    default_factory: Callable[[], Any],
    updater: Callable[[Any], Any],
    thread_lock: Optional[threading.Lock] = None,
) -> Any:
    """Đọc-sửa-ghi JSON an toàn với khóa 2 tầng (thread_lock + flock).

    updater(data) -> data mới. Nếu updater raise, KHÔNG ghi (atomic_write_json
    chưa được gọi).

    Trả về data sau khi update (để caller dùng lại — ví dụ id vừa thêm).
    """
    path = Path(path)
    lock_path = path.with_name(f"{path.name}.lock")

    def _do():
        with _file_lock(lock_path):
            data = read_json_safe(path, default_factory)
            new_data = updater(data)
            if new_data is None:
                new_data = data
            atomic_write_json(path, new_data)
            return new_data

    if thread_lock is not None:
        with thread_lock:
            return _do()
    return _do()