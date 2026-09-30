"""Bộ đọc/ghi general_config.json — cấu hình chung không thuộc registry.json.

Tách khỏi registry.json vì:
- registry.json là cấu hình BẢNG (matcher, /metadata, table-config cùng đọc).
- general_config.json chứa followup_rules, pii_keywords, thresholds — logic
  toàn hệ thống, không phụ thuộc bảng cụ thể.

2 chế độ đọc:
- get_config()    — lúc khởi động / lần gọi bất kỳ: file thiếu → tạo default;
                    file hỏng → TRẢ default cho lần gọi hiện tại nhưng KHÔNG
                    cache, lần sau tự đọc lại (tự phục hồi khi admin sửa file).
- reload_config() — admin gọi qua /admin/reload-index: file thiếu → tạo
                    default; file hỏng → raise JsonCorruptError (endpoint
                    catch và dừng, không nạp tiếp matcher/FAISS).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from json_store import atomic_write_json, JsonCorruptError

BASE = Path(__file__).resolve().parent
CONFIG_PATH = BASE / "general_config.json"

_DEFAULT_CONFIG: dict = {
    "version": 1,
    "followup_rules": {
        "keywords": ["đó", "này", "vậy", "thêm", "chi tiết hơn", "còn gì",
                     "nữa không", "kể thêm", "còn nữa"],
        "pronouns": ["thầy ấy", "cô ấy", "anh ấy", "chị ấy", "người đó",
                     "ngành đó", "khoa đó"],
    },
    "pii_keywords": ["dia chi nha", "dia chi ca nhan", "dia chi rieng",
                     "nha rieng"],
    "thresholds": {"duyet_similarity": 0.88, "lien_quan_similarity": 0.45},
}

_config_cache: Optional[dict] = None


def _default_copy() -> dict:
    return json.loads(json.dumps(_DEFAULT_CONFIG))


def _validate(cfg: dict) -> dict:
    """Bổ sung key còn thiếu từ default. Không từ chối, không crash."""
    if not isinstance(cfg, dict):
        return _default_copy()
    base = _default_copy()
    for k, v in base.items():
        if k not in cfg:
            cfg[k] = v
        elif isinstance(v, dict) and isinstance(cfg[k], dict):
            for kk, vv in v.items():
                cfg[k].setdefault(kk, vv)
    return cfg


def get_config() -> dict:
    """Đọc config. Cache lại nếu đọc THÀNH CÔNG. Nếu file hỏng: trả default
    cho lần gọi hiện tại nhưng KHÔNG cache — lần sau sẽ tự đọc lại file (tự
    phục hồi sau khi admin sửa)."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if not CONFIG_PATH.exists():
        try:
            atomic_write_json(CONFIG_PATH, _DEFAULT_CONFIG)
            print(f"-> Đã tạo {CONFIG_PATH.name} với default (chưa tồn tại).",
                  file=sys.stderr)
            _config_cache = _default_copy()  # file vừa tạo → cache OK
            return _config_cache
        except Exception as e:
            print(f"⚠️  Không tạo được {CONFIG_PATH.name} ({e}) — dùng default "
                  f"nội bộ cho lần gọi này (không cache).", file=sys.stderr)
            return _default_copy()  # KHÔNG cache — lần sau thử tạo lại

    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        _config_cache = _validate(data)
        return _config_cache
    except Exception as e:
        print(f"⚠️  {CONFIG_PATH.name} hỏng ({e}) — trả default cho lần gọi "
              f"này, KHÔNG cache (lần sau tự đọc lại).", file=sys.stderr)
        return _default_copy()  # KHÔNG cache


def reload_config() -> dict:
    """Đọc lại từ file. File thiếu → tạo default. File hỏng → RAISE
    JsonCorruptError (admin chủ động gọi endpoint, cần biết rõ thành/bại)."""
    global _config_cache
    if not CONFIG_PATH.exists():
        atomic_write_json(CONFIG_PATH, _DEFAULT_CONFIG)
        _config_cache = _default_copy()
        return {"reloaded_config": True, "created": True,
                "version": _config_cache["version"]}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        raise JsonCorruptError(
            f"{CONFIG_PATH.name} không parse được: {e}"
        ) from e
    _config_cache = _validate(data)
    return {"reloaded_config": True, "created": False,
            "version": _config_cache["version"]}


def update_config(patch: dict) -> dict:
    """Ghi patch vào file (dùng ở GĐ2). Ghi nguyên tử, cập nhật cache."""
    global _config_cache
    current = get_config()
    merged = {**current, **patch}
    atomic_write_json(CONFIG_PATH, merged)
    _config_cache = _validate(merged)
    return _config_cache