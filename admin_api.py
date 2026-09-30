"""
Admin API (port 8019).
Flow RAG: upload → prepare-md (OCR→Prepare→Generate) → check-conflict (Finalize+Check) → rag-in (rebuild FAISS).
Flow bảng: preview-structured-table (xem cột) → analyze-structured-upload
(job, PHÂN TÍCH không ghi gì) → Admin duyệt từng thay đổi/cảnh báo chéo
trên UI → apply-structured-decisions (job, ghi production) - cùng pattern
analyze-conflict/apply-decisions bên phi cấu trúc.

[MỚI] Hỗ trợ OCR_BACKEND override per-request: Admin chọn "qwen3_5" hoặc
"tesseract" ở UI, admin_api.py truyền qua biến môi trường con cho
pipeline_pdf.py - pipeline tự đọc os.getenv("OCR_BACKEND") như cũ, không cần
sửa pipeline_pdf.py.
"""
import os, sys, json, time, shutil, subprocess, threading, uuid, re
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from json_store import update_json, read_json_safe, atomic_write_json, JsonCorruptError

load_dotenv()

BASE = Path(__file__).resolve().parent
RAW_FOLDERS = {
    "pdf":  BASE / "data" / "raw" / "pdf",
    "docx": BASE / "data" / "raw" / "docx",
    "txt":  BASE / "data" / "raw" / "txt",
    "md":   BASE / "data" / "raw" / "md",
}
PROCESSED         = BASE / "data" / "processed"
TXT_FOLDER        = PROCESSED / "txt"
MD_FOLDER         = PROCESSED / "markdown"
CHUNK_FOLDER      = PROCESSED / "chunks"
MANIFEST_PATH     = PROCESSED / "conflict" / "manifest.json"
STRUCT_UPLOAD_DIR = BASE / "data" / "raw" / "structured_upload"
TEST_CASES_PATH   = BASE / "test_cases.json"

for f in list(RAW_FOLDERS.values()) + [STRUCT_UPLOAD_DIR]:
    f.mkdir(parents=True, exist_ok=True)

ADMIN_TOKEN = (os.getenv("API_AUTH_TOKEN") or "").strip()
if not ADMIN_TOKEN:
    raise SystemExit(
        "\n" + "=" * 70 + "\n"
        "FATAL: API_AUTH_TOKEN chưa được set trong .env\n"
        "\n"
        "admin_api KHÔNG được phép chạy public — service này có quyền\n"
        "ghi/sửa/xóa dữ liệu production. Đây là cấu hình sai tuyệt đối,\n"
        "không phải 'rủi ro': bất kỳ ai trong mạng cũng có thể xóa văn bản,\n"
        "sửa bảng, restore backup, đổi trạng thái chunk.\n"
        "\n"
        "Cách sửa:\n"
        "  1. Mở .env trong thư mục dự án\n"
        "  2. Thêm dòng: API_AUTH_TOKEN=<chuỗi-bí-mật-ngẫu-nhiên>\n"
        "     (gợi ý tạo: python -c \"import secrets; print(secrets.token_urlsafe(32))\")\n"
        "  3. Khởi động lại admin_api\n"
        "\n"
        "Lưu ý: nếu bạn muốn DEV không cần auth, đừng comment dòng này -\n"
        "hãy set 1 token tạm rồi dùng nó khi test (vd qua curl header X-API-Key).\n"
        + "=" * 70
    )
app = Flask(__name__, static_folder=str(BASE / "frontend"), static_url_path="")
CORS(app)


def _check_auth():
    if request.headers.get("X-API-Key") != ADMIN_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401
    return None


# [FIX TESSERACT] Thêm env_overrides để forward OCR_BACKEND (hoặc biến khác)
# xuống subprocess con - chỉ set khi THỰC SỰ có override, tránh thay đổi
# hành vi mặc định (kế thừa nguyên env của process cha).
def run_cmd(args, stdin_text=None, timeout=1800, env_overrides=None):
    kwargs = {"capture_output": True, "text": True, "timeout": timeout,
              "cwd": str(BASE), "encoding": "utf-8", "errors": "replace"}
    if stdin_text is not None: kwargs["input"] = stdin_text
    else: kwargs["stdin"] = subprocess.DEVNULL
    if env_overrides:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.update(env_overrides)
        kwargs["env"] = env
    try:
        r = subprocess.run(args, **kwargs)
        return {"ok": r.returncode == 0, "returncode": r.returncode,
                "stdout": r.stdout or "", "stderr": r.stderr or ""}
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "returncode": -1,
                "stdout": (e.stdout or "") if isinstance(e.stdout, str) else "",
                "stderr": f"TIMEOUT sau {timeout}s"}
    except Exception as e:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": str(e)}


def run_cmd_auto(args, yn_default="y", choice_default="1", timeout=1800):
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, cwd=str(BASE), text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    output, lock, stop = [], threading.Lock(), threading.Event()
    def reader():
        while not stop.is_set():
            try:
                ch = proc.stdout.read(1)
                if not ch: break
                with lock: output.append(ch)
            except Exception: break
    t = threading.Thread(target=reader, daemon=True); t.start()
    responded_at, start = 0, time.time()
    try:
        while proc.poll() is None:
            if time.time() - start > timeout: proc.kill(); break
            time.sleep(0.25)
            with lock:
                current = "".join(output); cur_len = len(current)
            if cur_len == responded_at: continue
            tail = current[-200:].rstrip()
            try:
                if tail.endswith("(y/n):"):
                    proc.stdin.write(yn_default + "\n"); proc.stdin.flush(); responded_at = cur_len
                elif tail.endswith("1/2/3/4:") or tail.endswith("1, 2, 3 hoặc 4:"):
                    proc.stdin.write(choice_default + "\n"); proc.stdin.flush(); responded_at = cur_len
                elif tail.endswith("để trống để hủy):"):
                    proc.stdin.write("\n"); proc.stdin.flush(); responded_at = cur_len
            except Exception: break
    finally:
        stop.set(); t.join(timeout=2)
        try: proc.wait(timeout=5)
        except Exception: proc.kill()
    with lock:
        return {"ok": proc.returncode == 0, "stdout": "".join(output), "stderr": ""}


def _get_manifest():
    if not MANIFEST_PATH.exists(): return {}
    try: return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception: return {}


def _save_manifest(manifest):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_file_status(base_name, file_type):
    manifest = _get_manifest()
    entry = manifest.get(base_name, {})
    return {
        "base_name": base_name, "file_type": file_type,
        "has_txt":    (TXT_FOLDER / f"{base_name}.txt").exists(),
        "has_review": (MD_FOLDER / f"{base_name}_review.json").exists(),
        "has_md":     (MD_FOLDER / f"{base_name}.md").exists(),
        "has_chunk":  (CHUNK_FOLDER / f"{base_name}.json").exists(),
        "in_production": base_name in manifest,
        "production_status": entry.get("trang_thai_van_ban"),
    }


def _find_raw_file(base_name, file_type):
    if file_type == "pdf":
        c = RAW_FOLDERS["pdf"] / f"{base_name}.pdf"; return c if c.exists() else None
    if file_type in ("docx", "doc"):
        for ext in (".docx", ".doc"):
            c = RAW_FOLDERS["docx"] / f"{base_name}{ext}"
            if c.exists(): return c
        return None
    if file_type == "txt":
        c = RAW_FOLDERS["txt"] / f"{base_name}.txt"; return c if c.exists() else None
    if file_type == "md":
        c = RAW_FOLDERS["md"] / f"{base_name}.md"; return c if c.exists() else None
    return None


# ============================================================
# [FIX TESSERACT] Helper validate OCR backend từ request
# ============================================================
_VALID_OCR_BACKENDS = ("qwen3_5", "tesseract")


def _build_ocr_env(ocr_backend_raw) -> dict | None:
    """Trả về {"OCR_BACKEND": "..."} nếu Admin chọn rõ ràng ở UI, hoặc None
    nếu không truyền / truyền giá trị lạ (để pipeline_pdf.py tự dùng giá
    trị mặc định trong .env của nó - hành vi y hệt trước khi có tính năng
    này, không phá vỡ cấu hình cũ)."""
    if not ocr_backend_raw:
        return None
    ocr_backend = str(ocr_backend_raw).strip().lower()
    if ocr_backend not in _VALID_OCR_BACKENDS:
        return None
    return {"OCR_BACKEND": ocr_backend}


# ============================================================
# JOB SYSTEM — chạy subprocess ở background + streaming log
# ============================================================
_jobs = {}
_jobs_lock = threading.Lock()
JOB_TTL_SECONDS = 3600


def _cleanup_old_jobs():
    now = time.time()
    with _jobs_lock:
        to_del = [jid for jid, j in _jobs.items()
                  if now - j.get("started_at", 0) > JOB_TTL_SECONDS
                  and j.get("status") == "done"]   
        for jid in to_del:
            _jobs.pop(jid, None)


# [FIX TESSERACT] Thêm env_overrides - forward OCR_BACKEND khi bước là OCR.
def _run_step_job(job_id, cmd, env_overrides=None):
    """Chạy subprocess background thread, đọc từng dòng stdout ghi vào log.
    Set PYTHONUNBUFFERED=1 để subprocess flush sau mỗi print() → progress real-time."""
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if env_overrides:
        env.update(env_overrides)
    proc = None
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(BASE),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env=env,
        )
        for line in iter(proc.stdout.readline, ""):
            with _jobs_lock:
                _jobs[job_id]["log"].append(line.rstrip("\n"))
        proc.wait(timeout=1800)
        with _jobs_lock:
            _jobs[job_id]["ok"] = (proc.returncode == 0)
            _jobs[job_id]["returncode"] = proc.returncode
            _jobs[job_id]["status"] = "done"
    except subprocess.TimeoutExpired:
        if proc: proc.kill()
        with _jobs_lock:
            _jobs[job_id]["log"].append("[TIMEOUT sau 1800s — đã kill]")
            _jobs[job_id]["ok"] = False
            _jobs[job_id]["returncode"] = -1
            _jobs[job_id]["status"] = "done"
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["log"].append(f"[LỖI HỆ THỐNG] {type(e).__name__}: {e}")
            _jobs[job_id]["ok"] = False
            _jobs[job_id]["returncode"] = -1
            _jobs[job_id]["status"] = "done"


def _resolve_step_cmd(base_name, file_type, step):
    """Trả về [cmd_list] hoặc (http_status, error_msg)."""
    if step == "ocr":
        if file_type != "pdf": return (400, "Chỉ PDF cần OCR")
        pdf = _find_raw_file(base_name, "pdf")
        if not pdf: return (404, "Không tìm thấy PDF")
        return [sys.executable, "pipeline_pdf.py", "ocr", str(pdf)]
    if step == "prepare":
        if file_type == "pdf":
            txt = TXT_FOLDER / f"{base_name}.txt"
            if not txt.exists(): return (400, "Chưa có .txt — chạy OCR trước")
            return [sys.executable, "pipeline_pdf.py", "prepare", str(txt)]
        src = _find_raw_file(base_name, file_type)
        if not src: return (404, "Không tìm thấy file gốc")
        sub = "docx" if src.suffix.lower() in (".docx", ".doc") else file_type
        return [sys.executable, "pipeline_docx_txt.py", sub, str(src)]
    if step == "generate":
        rv = MD_FOLDER / f"{base_name}_review.json"
        if not rv.exists(): return (400, "Chưa có _review.json — chạy Prepare trước")
        return [sys.executable, "pipeline_pdf.py", "generate", str(rv)]
    if step == "finalize":
        rv = MD_FOLDER / f"{base_name}_review.json"
        if not rv.exists(): return (400, "Chưa có _review.json")
        return [sys.executable, "pipeline_pdf.py", "finalize", str(rv)]
    return (400, f"step không hợp lệ cho job: {step}")


@app.route("/admin/start-step", methods=["POST"])
def start_step():
    """Tạo job chạy 1 bước (ocr|prepare|generate|finalize) → trả job_id ngay.

    [FIX TESSERACT] Nhận thêm field 'ocr_backend' (tuỳ chọn) trong body:
    'qwen3_5' | 'tesseract'. Chỉ có tác dụng với step='ocr' - forward qua
    biến môi trường con cho pipeline_pdf.py."""
    err = _check_auth()
    if err: return err
    _cleanup_old_jobs()
    body = request.get_json(silent=True) or {}
    base_name = body.get("base_name")
    file_type = body.get("file_type", "")
    step = body.get("step")
    if not base_name or not step:
        return jsonify({"error": "Thiếu base_name hoặc step"}), 400
    cmd = _resolve_step_cmd(base_name, file_type, step)
    if isinstance(cmd, tuple):
        return jsonify({"error": cmd[1]}), cmd[0]

    # [FIX TESSERACT] Chỉ áp OCR_BACKEND override cho bước OCR.
    env_overrides = _build_ocr_env(body.get("ocr_backend")) if step == "ocr" else None

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued", "log": [], "ok": None, "returncode": None,
            "base_name": base_name, "file_type": file_type, "step": step,
            "started_at": time.time(),
            "ocr_backend": (env_overrides or {}).get("OCR_BACKEND"),
        }
    threading.Thread(target=_run_step_job, args=(job_id, cmd),
                      kwargs={"env_overrides": env_overrides},
                      daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "base_name": base_name, "step": step,
                    "ocr_backend": (env_overrides or {}).get("OCR_BACKEND")})


@app.route("/admin/job/<job_id>", methods=["GET"])
def get_job_status(job_id):
    """Polling log. from=N chỉ lấy dòng từ index N trở đi (delta)."""
    err = _check_auth()
    if err: return err
    from_idx = int(request.args.get("from", "0"))
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job không tồn tại hoặc đã hết hạn"}), 404
        log = job["log"]
        return jsonify({
            "status": job["status"],
            "log": log[from_idx:],
            "log_total": len(log),
            "ok": job["ok"],
            "returncode": job["returncode"],
            "base_name": job.get("base_name"),
            "step": job.get("step"),
            "ocr_backend": job.get("ocr_backend"),
            "result": job.get("result"),
        })


# [FIX TESSERACT] Thêm env_overrides - forward OCR_BACKEND khi cần.
def _run_job_with_autorespond(job_id, cmd, yn_default="y", choice_default="1",
                               timeout=3600, env_overrides=None):
    """Chạy subprocess CÓ prompt input() — vừa STREAMING log theo dòng vừa
    auto-respond:
    - Prompt kết thúc '(y/n):'  → gửi yn_default
    - Prompt '1/2/3/4:'         → gửi choice_default
    - Prompt 'để trống để hủy):' → gửi dòng trống
    Đọc từng ký tự để bắt prompt ngay (input() không thêm newline sau prompt)."""
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if env_overrides:
        env.update(env_overrides)
    proc = None
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(BASE),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
        )
        output, line_buf = [], []
        lock, stop = threading.Lock(), threading.Event()

        def _flush_line():
            with lock:
                if line_buf:
                    with _jobs_lock:
                        _jobs[job_id]["log"].append("".join(line_buf))
                    line_buf.clear()

        def reader():
            while not stop.is_set():
                try:
                    ch = proc.stdout.read(1)
                    if not ch: break
                    with lock:
                        output.append(ch)
                        if ch == "\n":
                            line = "".join(line_buf).rstrip("\r")
                            line_buf.clear()
                            with _jobs_lock:
                                _jobs[job_id]["log"].append(line)
                        else:
                            line_buf.append(ch)
                except Exception: break
            _flush_line()

        rt = threading.Thread(target=reader, daemon=True); rt.start()
        responded_at, start = 0, time.time()
        while proc.poll() is None:
            if time.time() - start > timeout:
                proc.kill(); break
            time.sleep(0.25)
            with lock:
                current = "".join(output); cur_len = len(current)
            if cur_len == responded_at: continue
            tail = current[-200:].rstrip()
            try:
                if tail.endswith("(y/n):"):
                    proc.stdin.write(yn_default + "\n"); proc.stdin.flush()
                    _flush_line(); responded_at = cur_len
                elif tail.endswith("1/2/3/4:") or tail.endswith("1, 2, 3 hoặc 4:"):
                    proc.stdin.write(choice_default + "\n"); proc.stdin.flush()
                    _flush_line(); responded_at = cur_len
                elif tail.endswith("để trống để hủy):"):
                    proc.stdin.write("\n"); proc.stdin.flush()
                    _flush_line(); responded_at = cur_len
            except Exception: break
        stop.set(); rt.join(timeout=2)
        try: proc.wait(timeout=5)
        except Exception: proc.kill()
        with _jobs_lock:
            _jobs[job_id]["ok"] = (proc.returncode == 0)
            _jobs[job_id]["returncode"] = proc.returncode
            _jobs[job_id]["status"] = "done"
    except Exception as e:
        if proc: proc.kill()
        with _jobs_lock:
            _jobs[job_id]["log"].append(f"[LỖI HỆ THỐNG] {type(e).__name__}: {e}")
            _jobs[job_id]["ok"] = False
            _jobs[job_id]["returncode"] = -1
            _jobs[job_id]["status"] = "done"


@app.route("/admin/start-check", methods=["POST"])
def start_check():
    """Tạo job chạy finalize + check xung đột → trả job_id ngay.
    Frontend poll /admin/job/<id> để xem log streaming."""
    err = _check_auth()
    if err: return err
    _cleanup_old_jobs()
    body = request.get_json(silent=True) or {}
    base_name = body.get("base_name")
    mode = body.get("mode", "safe")
    if not base_name:
        return jsonify({"error": "Thiếu base_name"}), 400
    rv = MD_FOLDER / f"{base_name}_review.json"
    if not rv.exists():
        return jsonify({"error": "Chưa có _review.json — chạy Prepare trước"}), 400

    yn = "y" if mode == "replace" else "n"
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued", "log": [], "ok": None, "returncode": None,
            "base_name": base_name, "step": "check", "started_at": time.time(),
        }

    def worker():
        with _jobs_lock:
            _jobs[job_id]["log"].append("▶ [Bước 1/2] Finalize: chunk .md → chunks.json")
        r1 = run_cmd([sys.executable, "pipeline_pdf.py", "finalize", str(rv)], timeout=600)
        for ln in (r1["stdout"] or "").splitlines():
            with _jobs_lock: _jobs[job_id]["log"].append("   " + ln)
        if not r1["ok"]:
            for ln in (r1["stderr"] or "").splitlines():
                with _jobs_lock: _jobs[job_id]["log"].append("   [STDERR] " + ln)
            with _jobs_lock:
                _jobs[job_id]["ok"] = False
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["returncode"] = r1["returncode"]
            return

        with _jobs_lock:
            _jobs[job_id]["log"].append(f"▶ [Bước 2/2] Check xung đột với production (mode={mode})")
        _run_job_with_autorespond(
            job_id,
            [sys.executable, "conflict_detection.py", "check", base_name, f"--auto={mode}"],
            yn_default=yn, choice_default="1", timeout=3600,
        )

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "base_name": base_name, "mode": mode})


@app.route("/admin/analyze-conflict", methods=["POST"])
def analyze_conflict():
    """Chạy finalize + LLM phân tích xung đột KHÔNG tương tác → trả job_id.
    Frontend poll /admin/job/<id>, khi done sẽ nhận `result` JSON."""
    err = _check_auth()
    if err: return err
    _cleanup_old_jobs()
    body = request.get_json(silent=True) or {}
    base_name = body.get("base_name")
    if not base_name:
        return jsonify({"error": "Thiếu base_name"}), 400
    rv = MD_FOLDER / f"{base_name}_review.json"
    if not rv.exists():
        return jsonify({"error": "Chưa có _review.json"}), 400

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued", "log": [], "ok": None, "returncode": None,
            "base_name": base_name, "step": "analyze",
            "started_at": time.time(), "result": None,
        }

    def worker():
        with _jobs_lock:
            _jobs[job_id]["log"].append("▶ [1/2] Finalize .md → chunks.json")
        r1 = run_cmd([sys.executable, "pipeline_pdf.py", "finalize", str(rv)],
                     timeout=600)
        for ln in (r1["stdout"] or "").splitlines():
            with _jobs_lock: _jobs[job_id]["log"].append("   " + ln)
        if not r1["ok"]:
            with _jobs_lock:
                _jobs[job_id]["ok"] = False
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["returncode"] = r1["returncode"]
            return

        with _jobs_lock:
            _jobs[job_id]["log"].append("▶ [2/2] Quick scan: checksum + trùng lặp + vector (CHƯA gọi LLM)")
            _jobs[job_id]["status"] = "running"

        out_file = PROCESSED / "conflict" / f"quickscan_{base_name}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "conflict_detection.py", "quickscan-web",
               base_name, "--out", str(out_file)]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"; env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="replace",
                                bufsize=1, env=env)
        for line in iter(proc.stdout.readline, ""):
            with _jobs_lock:
                _jobs[job_id]["log"].append(line.rstrip("\n"))
        proc.wait(timeout=3600)

        if proc.returncode == 0 and out_file.exists():
            try:
                _jobs[job_id]["result"] = json.loads(
                    out_file.read_text(encoding="utf-8"))
            except Exception as e:
                with _jobs_lock:
                    _jobs[job_id]["log"].append(f"[LỖI parse result] {e}")
        with _jobs_lock:
            _jobs[job_id]["ok"] = (proc.returncode == 0)
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["returncode"] = proc.returncode

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "base_name": base_name})


@app.route("/admin/analyze-chunk-pairs", methods=["POST"])
def analyze_chunk_pairs():
    """Phase 2: gọi LLM cho các cặp đoạn CHỈ của các file user đã chọn
    'xuống cấp đoạn'. Request: {base_name, source_files: [...]}"""
    err = _check_auth()
    if err: return err
    _cleanup_old_jobs()
    body = request.get_json(silent=True) or {}
    base_name = body.get("base_name")
    source_files = body.get("source_files") or []
    if not base_name or not source_files:
        return jsonify({"error": "Thiếu base_name hoặc source_files"}), 400

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued", "log": [], "ok": None, "returncode": None,
            "base_name": base_name, "step": "analyze-pairs",
            "started_at": time.time(), "result": None,
        }

    def worker():
        with _jobs_lock:
            _jobs[job_id]["log"].append(
                f"▶ Gọi LLM cho {len(source_files)} file ứng viên (phase 2)...")
            _jobs[job_id]["status"] = "running"

        out_file = PROCESSED / "conflict" / f"analyze_pairs_{base_name}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        files_json = json.dumps(source_files, ensure_ascii=False)
        cmd = [sys.executable, "conflict_detection.py", "analyze-pairs-web",
               base_name, "--files", files_json, "--out", str(out_file)]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"; env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="replace",
                                bufsize=1, env=env)
        for line in iter(proc.stdout.readline, ""):
            with _jobs_lock:
                _jobs[job_id]["log"].append(line.rstrip("\n"))
        proc.wait(timeout=3600)

        if proc.returncode == 0 and out_file.exists():
            try:
                _jobs[job_id]["result"] = json.loads(
                    out_file.read_text(encoding="utf-8"))
            except Exception as e:
                with _jobs_lock:
                    _jobs[job_id]["log"].append(f"[LỖI parse result] {e}")
        with _jobs_lock:
            _jobs[job_id]["ok"] = (proc.returncode == 0)
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["returncode"] = proc.returncode

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "base_name": base_name})


@app.route("/admin/apply-decisions", methods=["POST"])
def apply_decisions():
    """Áp dụng quyết định xung đột do Admin chọn trên web UI."""
    err = _check_auth()
    if err: return err
    _cleanup_old_jobs()
    body = request.get_json(silent=True) or {}
    base_name = body.get("base_name")
    decisions = body.get("decisions")
    if not base_name or decisions is None:
        return jsonify({"error": "Thiếu base_name hoặc decisions"}), 400

    dec_file = PROCESSED / "conflict" / f"decisions_{base_name}.json"
    dec_file.parent.mkdir(parents=True, exist_ok=True)
    dec_file.write_text(json.dumps(decisions, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued", "log": [], "ok": None, "returncode": None,
            "base_name": base_name, "step": "apply",
            "started_at": time.time(), "result": None,
        }

    def worker():
        with _jobs_lock:
            n_chunk = len(decisions.get("chunk_decisions", []))
            n_dup   = len(decisions.get("exact_duplicates", []))
            _jobs[job_id]["log"].append(
                f"▶ Áp dụng {n_chunk} quyết định chunk + {n_dup} quyết định trùng lặp...")
            _jobs[job_id]["status"] = "running"

        cmd = [sys.executable, "conflict_detection.py", "apply-web",
               base_name, "--decisions", str(dec_file)]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"; env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="replace",
                                bufsize=1, env=env)
        json_buf, in_result = [], False
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip("\n")
            if line == "===APPLY_RESULT_START===": in_result = True; continue
            if line == "===APPLY_RESULT_END===":   in_result = False; continue
            if in_result:
                json_buf.append(line)
            else:
                with _jobs_lock:
                    _jobs[job_id]["log"].append(line)
        if json_buf:
            try:
                _jobs[job_id]["result"] = json.loads("\n".join(json_buf))
            except Exception as e:
                with _jobs_lock:
                    _jobs[job_id]["log"].append(f"[LỖI parse result] {e}")
        proc.wait(timeout=3600)
        with _jobs_lock:
            _jobs[job_id]["ok"] = (proc.returncode == 0)
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["returncode"] = proc.returncode

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "base_name": base_name})


# ============================================================
# STEP-STATUS — trạng thái 6 bước cho 1 file, tính next_step
# ============================================================
@app.route("/admin/step-status", methods=["GET"])
def step_status():
    err = _check_auth()
    if err: return err
    bn = request.args.get("base_name")
    ft = request.args.get("file_type", "pdf")
    if not bn: return jsonify({"error": "Thiếu base_name"}), 400

    steps = {
        "upload":   bool(_find_raw_file(bn, ft)),
        "ocr":      (TXT_FOLDER / f"{bn}.txt").exists(),
        "prepare":  (MD_FOLDER / f"{bn}_review.json").exists(),
        "generate": (MD_FOLDER / f"{bn}.md").exists(),
        "chunk":    (CHUNK_FOLDER / f"{bn}.json").exists(),
        "rag":      bn in _get_manifest(),
    }
    order = ["ocr", "prepare", "generate", "chunk", "rag"] if ft == "pdf" \
            else ["prepare", "generate", "chunk", "rag"]
    next_step = next((s for s in order if not steps[s]), None)
    return jsonify({"base_name": bn, "file_type": ft, "steps": steps, "next_step": next_step})


# ============================================================
# STATIC
# ============================================================
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "admin.html")


# ============================================================
# FILES
# ============================================================
@app.route("/admin/files", methods=["GET"])
def list_files():
    err = _check_auth()
    if err: return err
    files, seen = [], set()
    for ftype, folder in RAW_FOLDERS.items():
        if not folder.exists(): continue
        for f in folder.iterdir():
            if not f.is_file(): continue
            ext = f.suffix.lower().lstrip(".")
            if ext not in ("pdf", "docx", "doc", "txt", "md"): continue
            base = f.stem
            if base in seen: continue
            seen.add(base)
            st = _get_file_status(base, ftype)
            st["filename"] = f.name
            st["size_kb"] = round(f.stat().st_size / 1024, 1)
            st["mtime_ts"] = f.stat().st_mtime
            st["raw_mtime"] = datetime.fromtimestamp(st["mtime_ts"]).strftime("%Y-%m-%d %H:%M")
            files.append(st)
    files.sort(key=lambda x: x.get("mtime_ts", 0), reverse=True)
    return jsonify({"files": files})


@app.route("/admin/upload", methods=["POST"])
def upload():
    err = _check_auth()
    if err: return err
    if "file" not in request.files: return jsonify({"error": "Thiếu file"}), 400
    f = request.files["file"]
    if not f.filename: return jsonify({"error": "Tên file rỗng"}), 400
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ("pdf", "docx", "doc", "txt", "md"):
        return jsonify({"error": f".{ext} không được hỗ trợ"}), 400
    target_dir = (RAW_FOLDERS["pdf"] if ext == "pdf" else
                  RAW_FOLDERS["docx"] if ext in ("docx", "doc") else
                  RAW_FOLDERS["txt"] if ext == "txt" else RAW_FOLDERS["md"])
    target = target_dir / f.filename
    if target.exists():
        target = target_dir / f"{target.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{target.suffix}"
    f.save(str(target))
    return jsonify({"ok": True, "filename": target.name,
                    "saved_to": str(target.relative_to(BASE))})


# ============================================================
# DELETE FILE — xóa hẳn file + chunks + manifest, rebuild FAISS
# ============================================================
@app.route("/admin/delete-file", methods=["POST"])
def delete_file():
    err = _check_auth()
    if err: return err
    body = request.get_json() or {}
    base_name = body.get("base_name")
    file_type = body.get("file_type", "")
    if not base_name: return jsonify({"error": "Thiếu base_name"}), 400

    removed = []
    # 1. File gốc trong data/raw/
    raw = _find_raw_file(base_name, file_type)
    if raw:
        removed.append(str(raw.relative_to(BASE)))
        raw.unlink()
    # 2. .txt (OCR output)
    txt = TXT_FOLDER / f"{base_name}.txt"
    if txt.exists():
        removed.append(str(txt.relative_to(BASE)))
        txt.unlink()
    # 3. _review.json
    rv = MD_FOLDER / f"{base_name}_review.json"
    if rv.exists():
        removed.append(str(rv.relative_to(BASE)))
        rv.unlink()
    # 4. .md
    md = MD_FOLDER / f"{base_name}.md"
    if md.exists():
        removed.append(str(md.relative_to(BASE)))
        md.unlink()
    # 5. Chunks
    ch = CHUNK_FOLDER / f"{base_name}.json"
    if ch.exists():
        removed.append(str(ch.relative_to(BASE)))
        ch.unlink()
    # 6. Manifest
    manifest = _get_manifest()
    was_in_production = base_name in manifest
    if was_in_production:
        del manifest[base_name]
        _save_manifest(manifest)

    # 7. Nếu đã ở production → rebuild FAISS để chunks không còn trong vector
    rebuild_result = None
    if was_in_production:
        r = run_cmd([sys.executable, "conflict_detection.py", "rebuild-index"], timeout=1800)
        rebuild_result = {"ok": r["ok"],
                          "stdout": (r["stdout"] or "")[-20000:],
                          "stderr": (r["stderr"] or "")[-4000:]}

    return jsonify({
        "ok": True,
        "base_name": base_name,
        "removed": removed,
        "was_in_production": was_in_production,
        "rebuild": rebuild_result,
    })


# ============================================================
# PREPARE-MD — OCR → Prepare → Generate
# ============================================================
@app.route("/admin/prepare-md", methods=["POST"])
def prepare_md():
    """Chạy tuần tự OCR → Prepare → Generate (dừng TRƯỚC finalize).
    Các bước đã có output sẽ bị bỏ qua. Mỗi bước chạy xong mới check bước sau.

    [FIX TESSERACT] Nhận thêm field 'ocr_backend' (tuỳ chọn) - chỉ có tác
    dụng cho bước OCR, forward qua biến môi trường con."""
    err = _check_auth()
    if err: return err
    body = request.get_json(silent=True) or {}
    print(f"[prepare-md] body={body!r}")
    base_name = body.get("base_name")
    file_type = body.get("file_type", "")
    include_ocr = bool(body.get("include_ocr", True))
    if not base_name or not file_type:
        return jsonify({"ok": False, "error": "Thiếu base_name hoặc file_type",
                        "received": {"base_name": base_name, "file_type": file_type},
                        "steps_run": [], "logs": []}), 400

    # [FIX TESSERACT] Chuẩn bị env override cho bước OCR (nếu có chọn).
    ocr_env = _build_ocr_env(body.get("ocr_backend"))
    if ocr_env:
        print(f"[prepare-md] OCR_BACKEND override: {ocr_env}")

    logs = []
    steps_run = []

    # [FIX TESSERACT] _run nhận thêm env_overrides để forward cho run_cmd.
    def _run(step_name, cmd, env_overrides=None):
        r = run_cmd(cmd, timeout=1800, env_overrides=env_overrides)
        logs.append({"step": step_name, "ok": r["ok"],
                     "stdout": (r["stdout"] or "")[-20000:],
                     "stderr": (r["stderr"] or "")[-4000:]})
        return r["ok"]

    # ---------------- PDF ----------------
    if file_type == "pdf":
        # Bước 1: OCR (nếu chưa có .txt)
        if not (TXT_FOLDER / f"{base_name}.txt").exists():
            if not include_ocr:
                return jsonify({"ok": False,
                                "error": "Chưa có .txt và include_ocr=False. Bật OCR để chạy.",
                                "steps_run": steps_run, "logs": logs}), 400
            pdf = _find_raw_file(base_name, "pdf")
            if not pdf:
                return jsonify({"ok": False, "error": "Không tìm thấy PDF gốc",
                                "steps_run": steps_run, "logs": logs}), 404
            steps_run.append("ocr")
            # [FIX TESSERACT] Truyền ocr_env vào bước OCR.
            if not _run("ocr", [sys.executable, "pipeline_pdf.py", "ocr", str(pdf)],
                        env_overrides=ocr_env):
                return jsonify({"ok": False, "steps_run": steps_run, "logs": logs})

        # Bước 2: Prepare (nếu chưa có _review.json)
        if not (MD_FOLDER / f"{base_name}_review.json").exists():
            txt = TXT_FOLDER / f"{base_name}.txt"
            if not txt.exists():
                return jsonify({"ok": False,
                                "error": "OCR không tạo được .txt (xem log bước OCR để biết nguyên nhân)",
                                "steps_run": steps_run, "logs": logs}), 500
            steps_run.append("prepare")
            if not _run("prepare", [sys.executable, "pipeline_pdf.py", "prepare", str(txt)]):
                return jsonify({"ok": False, "steps_run": steps_run, "logs": logs})

        # Bước 3: Generate (nếu chưa có .md)
        if (not (MD_FOLDER / f"{base_name}.md").exists()
                and (MD_FOLDER / f"{base_name}_review.json").exists()):
            steps_run.append("generate")
            if not _run("generate", [sys.executable, "pipeline_pdf.py", "generate",
                                      str(MD_FOLDER / f"{base_name}_review.json")]):
                return jsonify({"ok": False, "steps_run": steps_run, "logs": logs})

    # ---------------- DOCX / TXT / MD ----------------
    else:
        if not (MD_FOLDER / f"{base_name}_review.json").exists():
            src = _find_raw_file(base_name, file_type)
            if not src:
                return jsonify({"ok": False, "error": "Không tìm thấy file gốc",
                                "steps_run": steps_run, "logs": logs}), 404
            sub = "docx" if src.suffix.lower() in (".docx", ".doc") else file_type
            steps_run.append("prepare")
            if not _run("prepare", [sys.executable, "pipeline_docx_txt.py", sub, str(src)]):
                return jsonify({"ok": False, "steps_run": steps_run, "logs": logs})

        if (file_type in ("txt", "md")
                and (MD_FOLDER / f"{base_name}_review.json").exists()
                and not (MD_FOLDER / f"{base_name}.md").exists()):
            steps_run.append("generate")
            if not _run("generate", [sys.executable, "pipeline_pdf.py", "generate",
                                      str(MD_FOLDER / f"{base_name}_review.json")]):
                return jsonify({"ok": False, "steps_run": steps_run, "logs": logs})

    return jsonify({"ok": True, "steps_run": steps_run, "logs": logs,
                    "ocr_backend": (ocr_env or {}).get("OCR_BACKEND")})


# ============================================================
# CHECK-CONFLICT — Finalize + Check xung đột
# ============================================================
@app.route("/admin/check-conflict", methods=["POST"])
def check_conflict():
    err = _check_auth()
    if err: return err
    body = request.get_json(silent=True) or {}
    base_name = body.get("base_name")
    mode = body.get("mode", "safe")
    if not base_name: return jsonify({"error": "Thiếu base_name"}), 400
    rv = MD_FOLDER / f"{base_name}_review.json"
    if not rv.exists(): return jsonify({"error": "Chưa có _review.json — chạy Prepare trước"}), 400

    logs = []
    r_fin = run_cmd([sys.executable, "pipeline_pdf.py", "finalize", str(rv)], timeout=300)
    logs.append({"step": "finalize", "ok": r_fin["ok"],
                 "stdout": (r_fin["stdout"] or "")[-8000:],
                 "stderr": (r_fin["stderr"] or "")[-4000:]})
    if not r_fin["ok"]:
        return jsonify({"ok": False, "base_name": base_name, "mode": mode, "logs": logs})

    yn = "y" if mode == "replace" else "n"
    r_check = run_cmd_auto([sys.executable, "conflict_detection.py", "check", base_name, f"--auto={mode}"],
                           yn_default=yn, choice_default="1", timeout=3600)
    logs.append({"step": "check", "ok": r_check["ok"],
                 "stdout": (r_check["stdout"] or "")[-40000:],
                 "stderr": (r_check["stderr"] or "")[-4000:]})
    return jsonify({"ok": all(l["ok"] for l in logs),
                    "base_name": base_name, "mode": mode, "logs": logs})


# ============================================================
# RAG-IN — rebuild FAISS
# ============================================================
@app.route("/admin/rag-in", methods=["POST"])
def rag_in():
    err = _check_auth()
    if err: return err
    r = run_cmd([sys.executable, "conflict_detection.py", "rebuild-index"], timeout=3600)
    def _cut(s, n=20000): return s if len(s) <= n else s[-n:]
    return jsonify({"ok": r["ok"], "stdout": _cut(r["stdout"]), "stderr": _cut(r["stderr"], 8000)})


# ============================================================
# Aliases
# ============================================================
@app.route("/admin/process", methods=["POST"])
def process():
    """[FIX TESSERACT] Endpoint alias này cũng nhận ocr_backend - cùng cơ chế
    như /admin/start-step, giữ đồng bộ 2 endpoint."""
    err = _check_auth()
    if err: return err
    body = request.get_json(silent=True) or {}
    base_name = body.get("base_name"); file_type = body.get("file_type", ""); step = body.get("step", "")
    if not base_name or not step: return jsonify({"error": "Thiếu base_name hoặc step"}), 400
    ocr_env = _build_ocr_env(body.get("ocr_backend")) if step == "ocr" else None
    if step == "ocr":
        if file_type != "pdf": return jsonify({"error": "Chỉ PDF mới cần OCR"}), 400
        pdf = _find_raw_file(base_name, "pdf")
        if not pdf: return jsonify({"error": "Không tìm thấy file PDF"}), 404
        r = run_cmd([sys.executable, "pipeline_pdf.py", "ocr", str(pdf)], env_overrides=ocr_env)
    elif step == "prepare":
        if file_type == "pdf":
            txt = TXT_FOLDER / f"{base_name}.txt"
            if not txt.exists(): return jsonify({"error": "Chưa có .txt"}), 400
            r = run_cmd([sys.executable, "pipeline_pdf.py", "prepare", str(txt)])
        else:
            src = _find_raw_file(base_name, file_type)
            if not src: return jsonify({"error": "Không tìm thấy file gốc"}), 404
            sub = "docx" if src.suffix.lower() in (".docx", ".doc") else file_type
            r = run_cmd([sys.executable, "pipeline_docx_txt.py", sub, str(src)])
    elif step == "generate":
        rv = MD_FOLDER / f"{base_name}_review.json"
        if not rv.exists(): return jsonify({"error": "Chưa có _review.json"}), 400
        r = run_cmd([sys.executable, "pipeline_pdf.py", "generate", str(rv)])
    elif step == "finalize":
        rv = MD_FOLDER / f"{base_name}_review.json"
        if not rv.exists(): return jsonify({"error": "Chưa có _review.json"}), 400
        r = run_cmd([sys.executable, "pipeline_pdf.py", "finalize", str(rv)])
    else:
        return jsonify({"error": f"step không hợp lệ"}), 400
    def _cut(s, n): return s if len(s) <= n else s[-n:]
    return jsonify({"ok": r["ok"], "step": step, "base_name": base_name,
                    "stdout": _cut(r["stdout"], 30000), "stderr": _cut(r["stderr"], 8000)})


@app.route("/admin/check", methods=["POST"])
def check_alias(): return check_conflict()

@app.route("/admin/rebuild-index", methods=["POST"])
def rebuild_index_alias(): return rag_in()


# ============================================================
# MARKDOWN / CHUNKS / META
# ============================================================
@app.route("/admin/markdown", methods=["GET", "POST"])
def markdown_io():
    err = _check_auth()
    if err: return err
    if request.method == "GET":
        base_name = request.args.get("base_name"); stage = request.args.get("stage", "md")
        if not base_name: return jsonify({"error": "Thiếu base_name"}), 400
        path = (MD_FOLDER / f"{base_name}.md") if stage == "md" else (TXT_FOLDER / f"{base_name}.txt")
        if not path.exists(): return jsonify({"error": f"Không tìm thấy {path.name}"}), 404
        return jsonify({"content": path.read_text(encoding="utf-8"),
                        "path": str(path.relative_to(BASE)), "stage": stage, "size": path.stat().st_size})
    body = request.get_json(silent=True) or {}
    base_name = body.get("base_name"); stage = body.get("stage", "md"); content = body.get("content", "")
    if not base_name: return jsonify({"error": "Thiếu base_name"}), 400
    path = (MD_FOLDER / f"{base_name}.md") if stage == "md" else (TXT_FOLDER / f"{base_name}.txt")
    if not path.exists(): return jsonify({"error": f"Không tìm thấy {path.name}"}), 404
    path.write_text(content, encoding="utf-8")

    result = {"ok": True, "cascade": []}
    if stage == "md":
        # ... giữ nguyên toàn bộ nhánh md như cũ
        rv = MD_FOLDER / f"{base_name}_review.json"
        if rv.exists():
            r_fin = run_cmd([sys.executable, "pipeline_pdf.py", "finalize", str(rv)], timeout=600)
            result["cascade"].append({"step": "finalize", "ok": r_fin["ok"],
                                      "stdout": (r_fin["stdout"] or "")[-8000:],
                                      "stderr": (r_fin["stderr"] or "")[-4000:]})
            was_in_production = base_name in _get_manifest()
            result["was_in_production"] = was_in_production
            if r_fin["ok"] and was_in_production:
                r_rag = run_cmd([sys.executable, "conflict_detection.py", "rebuild-index"], timeout=3600)
                result["cascade"].append({"step": "rebuild-index", "ok": r_rag["ok"],
                                          "stdout": (r_rag["stdout"] or "")[-8000:],
                                          "stderr": (r_rag["stderr"] or "")[-4000:]})
        else:
            result["warning"] = "Chưa có _review.json - chỉ lưu .md, KHÔNG có chunk để cập nhật."

    elif stage == "txt":
        # [FIX #1] Đồng bộ nội dung .txt vừa lưu vào _review.json["ocr_text"]
        # nếu file đó đã tồn tại. Lý do: prepare_for_review() chụp SNAPSHOT
        # ocr_text vào _review.json 1 lần duy nhất, và generate_markdown_for_review()
        # đọc TỪ review.json["ocr_text"] (không đọc lại file .txt). Nếu không
        # sync, mọi sửa OCR trong tab .txt sẽ bị bỏ qua âm thầm khi Generate -
        # đúng hiện tượng "sửa .txt rồi Lưu, tưởng đã cập nhật, nhưng .md
        # vẫn theo bản OCR cũ".
        rv = MD_FOLDER / f"{base_name}_review.json"
        if rv.exists():
            try:
                data = json.loads(rv.read_text(encoding="utf-8"))
                data["ocr_text"] = content
                rv.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                result["cascade"].append({"step": "sync_review_json_ocr_text", "ok": True})
                result["warning"] = (
                    "Đã đồng bộ .txt mới vào _review.json['ocr_text']. "
                    "LƯU Ý: metadata_goi_y (loai_van_ban/tieu_de/so_hieu...) sinh từ bản OCR CŨ "
                    "nên có thể không còn chính xác. Nếu sửa OCR đáng kể, cân nhắc chạy 'Prepare lại' "
                    "(tab Tổng quan) TRƯỚC KHI bấm 'Generate' để sinh lại .md từ nội dung mới."
                )
            except Exception as e:
                result["ok"] = False
                result["error"] = (f"Đã ghi .txt thành công nhưng KHÔNG sync được _review.json "
                                   f"(Generate tiếp theo sẽ vẫn dùng bản OCR cũ): {e}")
        else:
            # Chưa có review.json - chưa qua Prepare, không cần sync gì.
            # Ghi .txt xong, admin chạy Prepare như bình thường.
            result["cascade"].append({"step": "chỉ_ghi_txt", "ok": True})

    return jsonify(result)


@app.route("/admin/chunks", methods=["GET"])
def get_chunks():
    err = _check_auth()
    if err: return err
    base_name = request.args.get("base_name")
    if not base_name: return jsonify({"error": "Thiếu base_name"}), 400
    path = CHUNK_FOLDER / f"{base_name}.json"
    if not path.exists():
        return jsonify({"error": "Chưa có chunks", "chunks": [], "n": 0}), 404
    try:
        chunks = json.loads(path.read_text(encoding="utf-8"))
        rows = [{"chunk_id": c.get("metadata",{}).get("chunk_id"),
                 "trang_thai": c.get("metadata",{}).get("trang_thai"),
                 "content_type": c.get("metadata",{}).get("content_type"),
                 "trang": c.get("metadata",{}).get("trang"),
                 "chuong_hoac_muc": c.get("metadata",{}).get("chuong_hoac_muc"),
                 "dieu": c.get("metadata",{}).get("dieu"),
                 "loai_van_ban": c.get("metadata",{}).get("loai_van_ban"),
                 "valid_from": c.get("metadata",{}).get("valid_from"),
                 "valid_to": c.get("metadata",{}).get("valid_to"),
                 "nhan_phan_biet": c.get("metadata",{}).get("nhan_phan_biet"),
                 "content": c.get("content", "")} for c in chunks]
        return jsonify({"chunks": rows, "n": len(rows), "path": str(path.relative_to(BASE))})
    except Exception as e: return jsonify({"error": str(e)}), 500


_CHUNK_EDITABLE_META_FIELDS = ("trang_thai", "valid_from", "valid_to", "nhan_phan_biet")


@app.route("/admin/chunks", methods=["POST"])
def update_chunks():
    err = _check_auth()
    if err: return err
    body = request.get_json(silent=True) or {}
    base_name = body.get("base_name")
    rows = body.get("chunks")
    if not base_name or rows is None:
        return jsonify({"error": "Thiếu base_name hoặc chunks"}), 400
    path = CHUNK_FOLDER / f"{base_name}.json"
    if not path.exists():
        return jsonify({"error": "Chưa có chunks cho văn bản này"}), 404

    try:
        chunks = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return jsonify({"error": f"Không đọc được chunk file hiện tại: {e}"}), 500

    by_id = {c.get("metadata", {}).get("chunk_id"): c for c in chunks}
    updated, not_found = [], []
    for row in rows:
        cid = row.get("chunk_id")
        c = by_id.get(cid)
        if c is None:
            not_found.append(cid)
            continue
        if "content" in row and row["content"] != c.get("content"):
            c["content"] = row["content"]
            c.setdefault("metadata", {})["da_sua_tay_luc"] = datetime.now().isoformat()
        for field in _CHUNK_EDITABLE_META_FIELDS:
            if field in row:
                c.setdefault("metadata", {})[field] = row[field]
        updated.append(cid)

    chunk_backup_dir = CHUNK_FOLDER / "_backups"
    chunk_backup_dir.mkdir(parents=True, exist_ok=True)
    bak = chunk_backup_dir / f"{path.stem}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy2(path, bak)
    path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    rebuild_result = None
    was_in_production = base_name in _get_manifest()
    if was_in_production:
        r = run_cmd([sys.executable, "conflict_detection.py", "rebuild-index"], timeout=3600)
        rebuild_result = {"ok": r["ok"], "stdout": (r["stdout"] or "")[-8000:],
                          "stderr": (r["stderr"] or "")[-4000:]}

    return jsonify({"ok": True, "updated": updated, "not_found": not_found,
                    "was_in_production": was_in_production, "rebuild": rebuild_result,
                    "backup": str(bak.relative_to(BASE))})


@app.route("/admin/review-meta", methods=["GET", "POST"])
def review_meta():
    err = _check_auth()
    if err: return err
    if request.method == "GET":
        base_name = request.args.get("base_name")
        if not base_name: return jsonify({"error": "Thiếu base_name"}), 400
        rv = MD_FOLDER / f"{base_name}_review.json"
        if not rv.exists(): return jsonify({"error": "Chưa có _review.json"}), 404
        try:
            data = json.loads(rv.read_text(encoding="utf-8"))
            return jsonify({"metadata_xac_nhan": data.get("metadata_xac_nhan", {}),
                            "metadata_goi_y": data.get("metadata_goi_y", {}),
                            "loai_van_ban_options": data.get("loai_van_ban_options", [])})
        except Exception as e: return jsonify({"error": str(e)}), 500
    body = request.get_json(silent=True) or {}
    base_name = body.get("base_name"); new_meta = body.get("metadata_xac_nhan")
    regenerate_md = bool(body.get("regenerate_md", False))
    if not base_name or new_meta is None: return jsonify({"error": "Thiếu input"}), 400
    rv = MD_FOLDER / f"{base_name}_review.json"
    if not rv.exists(): return jsonify({"error": "Chưa có _review.json"}), 404
    try:
        data = json.loads(rv.read_text(encoding="utf-8"))
        data["metadata_xac_nhan"] = {**data.get("metadata_xac_nhan", {}), **new_meta}
        rv.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e: return jsonify({"error": str(e)}), 500

    result = {"ok": True, "metadata_xac_nhan": data["metadata_xac_nhan"]}
    md_path = MD_FOLDER / f"{base_name}.md"
    result["md_exists"] = md_path.exists()
    if md_path.exists() and not regenerate_md:
        result["warning"] = ("Đã lưu metadata, nhưng .md hiện có vẫn theo metadata CŨ. "
                             "Gọi lại /admin/review-meta với regenerate_md=true (hoặc bấm "
                             "'Sinh lại markdown' trên UI) nếu muốn áp dụng - sẽ GHI ĐÈ "
                             "mọi sửa tay đang có trong .md hiện tại.")
    elif regenerate_md:
        # [FIX #5] Cascade ĐẦY ĐỦ giống saveMd stage=md: generate -> finalize
        # -> rebuild-index (nếu đang production).
        # [FIX C1] Thêm guard checksum: chỉ finalize nếu .md THỰC SỰ thay đổi.
        import hashlib
        md_path = MD_FOLDER / f"{base_name}.md"
        md_hash_before = hashlib.sha256(md_path.read_bytes()).hexdigest() if md_path.exists() else None

        r_gen = run_cmd([sys.executable, "pipeline_pdf.py", "generate", str(rv)], timeout=600)
        result["cascade"] = [{"step": "generate", "ok": r_gen["ok"],
                              "stdout": (r_gen["stdout"] or "")[-8000:],
                              "stderr": (r_gen["stderr"] or "")[-4000:]}]

        if not r_gen["ok"]:
            result["ok"] = False
            return jsonify(result)

        # GUARD: kiểm tra .md thực sự thay đổi
        if not md_path.exists():
            result["ok"] = False
            result["cascade"].append({
                "step": "guard_after_generate", "ok": False,
                "error": ".md không tồn tại sau generate"
            })
            return jsonify(result)

        # review_meta: KHÔNG chặn nếu .md không đổi — metadata có thể không ảnh
        # hưởng cấu trúc .md (vd so_hieu/ngay_ban_hanh khi markdown_da_chinh_xac=True).
        # Chỉ ghi note nhẹ để debug.
        md_hash_after = hashlib.sha256(md_path.read_bytes()).hexdigest()
        if md_hash_after == md_hash_before:
            result["cascade"].append({
                "step": "guard_after_generate", "ok": True,
                "note": "Nội dung .md không đổi sau generate — metadata vừa sửa "
                        "không ảnh hưởng cấu trúc .md. Đây là hành vi hợp lệ."
            })

        if r_gen["ok"]:
            md_path_after_gen = MD_FOLDER / f"{base_name}.md"
            if md_path_after_gen.exists():
                r_fin = run_cmd([sys.executable, "pipeline_pdf.py", "finalize", str(rv)], timeout=600)
                result["cascade"].append({"step": "finalize", "ok": r_fin["ok"],
                                          "stdout": (r_fin["stdout"] or "")[-8000:],
                                          "stderr": (r_fin["stderr"] or "")[-4000:]})
                was_in_production = base_name in _get_manifest()
                result["was_in_production"] = was_in_production
                if r_fin["ok"] and was_in_production:
                    r_rag = run_cmd([sys.executable, "conflict_detection.py", "rebuild-index"],
                                    timeout=3600)
                    result["cascade"].append({"step": "rebuild-index", "ok": r_rag["ok"],
                                              "stdout": (r_rag["stdout"] or "")[-8000:],
                                              "stderr": (r_rag["stderr"] or "")[-4000:]})
                result["note"] = ("Đã sinh lại .md theo metadata mới + chunk lại"
                                  + (" + rebuild-index FAISS." if was_in_production else "."))
    return jsonify(result)

# ============================================================
# [FIX #7] SAVE-ALL — gộp TXT + METADATA + GENERATE + FINALIZE + REBUILD
# vào 1 lần gọi. Dùng cho nút "Lưu tất cả" trong modal File Detail, khi
# admin sửa nhiều tab trong 1 phiên (vd sửa .txt fix OCR + đổi loai_van_ban
# + regenerate .md) và muốn 1 nút thay vì phải bấm tuần tự đúng thứ tự.
#
# KHÔNG bao gồm chunks — chunks có flow riêng (bulk status, backup, ...),
# admin vẫn bấm "Lưu chunks" riêng khi cần.
# ============================================================
@app.route("/admin/save-all", methods=["POST"])
def save_all():
    err = _check_auth()
    if err: return err
    body = request.get_json(silent=True) or {}
    base_name = body.get("base_name")
    if not base_name:
        return jsonify({"error": "Thiếu base_name"}), 400

    txt_content = body.get("txt_content")            # optional
    new_meta    = body.get("metadata_xac_nhan")      # optional
    # Mặc định True: sau khi sync txt+metadata thì LUÔN regen .md (vì cả 2
    # nguồn này đều feed vào .md), để đảm bảo .md phản ánh đúng cấu trúc mới.
    force_regen = body.get("force_regen", True)

    if not txt_content and new_meta is None:
        return jsonify({"error": "Không có gì để lưu (thiếu cả txt_content lẫn metadata_xac_nhan)"}), 400

    rv = MD_FOLDER / f"{base_name}_review.json"
    if not rv.exists():
        return jsonify({"error": "Chưa có _review.json — chạy Prepare trước"}), 404

    result = {"ok": True, "cascade": [], "steps_done": []}

    # ---- Bước 1: sync .txt vào review.json["ocr_text"] ----
    if txt_content is not None:
        txt_path = TXT_FOLDER / f"{base_name}.txt"
        if txt_path.exists():
            txt_path.write_text(txt_content, encoding="utf-8")
            result["steps_done"].append("ghi_txt")
        try:
            data = json.loads(rv.read_text(encoding="utf-8"))
            data["ocr_text"] = txt_content
            rv.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            result["steps_done"].append("sync_ocr_text")
        except Exception as e:
            result["ok"] = False
            result["cascade"].append({"step": "sync_ocr_text", "ok": False, "error": str(e)})
            return jsonify(result)

    # ---- Bước 2: sync metadata_xac_nhan ----
    if new_meta is not None:
        try:
            data = json.loads(rv.read_text(encoding="utf-8"))
            data["metadata_xac_nhan"] = {**data.get("metadata_xac_nhan", {}), **new_meta}
            rv.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            result["steps_done"].append("sync_metadata")
        except Exception as e:
            result["ok"] = False
            result["cascade"].append({"step": "sync_metadata", "ok": False, "error": str(e)})
            return jsonify(result)

    # ---- Bước 3: generate (luôn chạy nếu force_regen + có gì đã sync) ----
    if force_regen:
        import hashlib

        # Lấy hash trước khi generate để ghi note (không chặn)
        md_path = MD_FOLDER / f"{base_name}.md"
        md_hash_before = hashlib.sha256(md_path.read_bytes()).hexdigest() if md_path.exists() else None

        r_gen = run_cmd([sys.executable, "pipeline_pdf.py", "generate", str(rv)], timeout=600)
        result["cascade"].append({"step": "generate", "ok": r_gen["ok"],
                                  "stdout": (r_gen["stdout"] or "")[-8000:],
                                  "stderr": (r_gen["stderr"] or "")[-4000:]})
        if not r_gen["ok"]:
            result["ok"] = False
            return jsonify(result)

        # ---- GUARD (đơn giản hoá sau khi xác nhận guard hash gây false positive) ----
        # Lý do bỏ guard hash:
        #   1. _atomic_write_text() đã đảm bảo .md hoặc ghi thành công hoàn toàn,
        #      hoặc không ghi gì và raise exception → subprocess exit != 0 → r_gen["ok"]=False.
        #   2. Nếu generate chạy tới đây với r_gen["ok"]=True → .md đã được ghi
        #      atomic thành công → không thể có chuyện "generate OK nhưng .md cũ".
        #   3. Hash không đổi có thể do:
        #      (a) input không đổi thực sự, HOẶC
        #      (b) input đổi nhưng field đổi không ảnh hưởng .md
        #          (vd so_hieu khi markdown_da_chinh_xac=True, vì generate
        #           dùng thẳng ocr_text, KHÔNG chạy markup_headers()).
        #      Cả (a) và (b) đều HỢP LỆ → không được chặn finalize.
        #   4. Chỉ còn check md_path.exists() để bắt case kỳ lạ hiếm gặp
        #      (file bị xóa giữa generate và finalize).
        if not md_path.exists():
            result["ok"] = False
            result["cascade"].append({
                "step": "guard_after_generate", "ok": False,
                "error": "generate báo OK nhưng .md không tồn tại sau khi chạy"
            })
            return jsonify(result)

        # Ghi note nhẹ nếu hash không đổi (debug, không chặn)
        md_hash_after = hashlib.sha256(md_path.read_bytes()).hexdigest()
        if md_hash_after == md_hash_before:
            result["cascade"].append({
                "step": "guard_after_generate", "ok": True,
                "note": "Nội dung .md không đổi sau generate — có thể do input không "
                        "thay đổi, hoặc metadata sửa không ảnh hưởng cấu trúc .md "
                        "(vd so_hieu khi markdown_da_chinh_xac=True). Đây là hành vi hợp lệ."
            })

        # ---- Bước 4: finalize ----
        r_fin = run_cmd([sys.executable, "pipeline_pdf.py", "finalize", str(rv)], timeout=600)
        result["cascade"].append({"step": "finalize", "ok": r_fin["ok"],
                                  "stdout": (r_fin["stdout"] or "")[-8000:],
                                  "stderr": (r_fin["stderr"] or "")[-4000:]})
        if not r_fin["ok"]:
            result["ok"] = False
            return jsonify(result)

        # ---- Bước 5: rebuild-index nếu đang production ----
        was_in_production = base_name in _get_manifest()
        result["was_in_production"] = was_in_production
        if was_in_production:
            r_rag = run_cmd([sys.executable, "conflict_detection.py", "rebuild-index"],
                            timeout=3600)
            result["cascade"].append({"step": "rebuild-index", "ok": r_rag["ok"],
                                      "stdout": (r_rag["stdout"] or "")[-8000:],
                                      "stderr": (r_rag["stderr"] or "")[-4000:]})
            if not r_rag["ok"]:
                result["ok"] = False

    return jsonify(result)

# ============================================================
# Status / Pending
# ============================================================
@app.route("/admin/pending", methods=["GET"])
def pending():
    err = _check_auth()
    if err: return err
    r = run_cmd([sys.executable, "conflict_detection.py", "list-pending"], timeout=60)
    return jsonify({"ok": r["ok"], "stdout": r["stdout"], "stderr": r["stderr"]})


@app.route("/admin/status", methods=["GET"])
def status():
    err = _check_auth()
    if err: return err
    r = run_cmd([sys.executable, "conflict_detection.py", "status"], timeout=60)
    return jsonify({"ok": r["ok"], "stdout": r["stdout"], "stderr": r["stderr"]})


# ============================================================
# TABLES
# ============================================================
@app.route("/admin/tables", methods=["GET"])
def list_tables():
    err = _check_auth()
    if err: return err
    try: registry = json.loads((BASE / "registry.json").read_text(encoding="utf-8"))
    except Exception as e: return jsonify({"error": str(e)}), 500
    tables = []
    for name, cfg in registry.items():
        p = BASE / cfg["path"]; n_rows = 0
        if p.exists():
            try:
                with open(p, encoding="utf-8") as fp:
                    n_rows = max(0, sum(1 for _ in fp) - 1)
            except Exception: pass
        tables.append({"name": name, "path": cfg["path"],
                       "primary_key": cfg.get("primary_key"),
                       "description": cfg.get("description", ""), "n_rows": n_rows})
    return jsonify({"tables": tables})


@app.route("/admin/table-preview", methods=["GET"])
def table_preview():
    err = _check_auth()
    if err: return err
    name = request.args.get("name")
    if not name: return jsonify({"error": "Thiếu name"}), 400
    try:
        registry = json.loads((BASE / "registry.json").read_text(encoding="utf-8"))
        if name not in registry: return jsonify({"error": f"Bảng '{name}' không tồn tại"}), 404
        path = BASE / registry[name]["path"]
        if not path.exists():
            return jsonify({"columns": [], "rows": [], "n_total": 0, "path": registry[name]["path"]})
        df = pd.read_csv(path)
        cols = list(df.columns)

        # ── Tham số all=1 → trả TOÀN BỘ dòng, KHÔNG cắt ──
        show_all = (request.args.get("all") or "").lower() in ("1", "true", "yes")
        if show_all:
            rows = df.fillna("").astype(str).to_dict("records")   # TẤT CẢ dòng
        else:
            rows = df.head(300).fillna("").astype(str).to_dict("records")

        pii_cols    = set(registry[name].get("pii_columns") or [])
        hidden_cols = set(registry[name].get("hidden_columns") or [])
        column_flags = {c: {"pii": c in pii_cols, "hidden": c in hidden_cols} for c in cols}
        return jsonify({
            "columns": cols, "rows": rows, "n_total": len(df),
            "n_loaded": len(rows),
            "showing_all": show_all,
            "path": registry[name]["path"],
            "primary_key": registry[name].get("primary_key"),
            "column_flags": column_flags,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/table-update", methods=["POST"])
def table_update():
    err = _check_auth()
    if err: return err
    body = request.get_json(silent=True) or {}
    name = body.get("name"); columns = body.get("columns"); rows = body.get("rows")
    if not name or not columns or rows is None: return jsonify({"error": "Thiếu input"}), 400
    try:
        registry = json.loads((BASE / "registry.json").read_text(encoding="utf-8"))
        if name not in registry: return jsonify({"error": f"Bảng '{name}' không tồn tại"}), 404
        path = BASE / registry[name]["path"]
        if path.exists():
            bak = path.with_name(f"{path.stem}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
            shutil.copy2(path, bak)
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(path, index=False)
        reload_status = "skipped"
        try:
            import urllib.request
            api_token = os.getenv("API_AUTH_TOKEN")
            api_base  = os.getenv("API8018_BASE_URL", "http://127.0.0.1:8018")
            if api_token:
                req = urllib.request.Request(f"{api_base}/admin/reload-index",
                                             method="POST", headers={"X-API-Key": api_token})
                urllib.request.urlopen(req, timeout=15)
                reload_status = "ok"
        except Exception as e: reload_status = f"failed: {e}"
        return jsonify({"ok": True, "n_rows": len(df), "reload": reload_status})
    except Exception as e: return jsonify({"error": str(e)}), 500


@app.route("/admin/table-config", methods=["GET"])
def table_config():
    err = _check_auth()
    if err: return err
    name = request.args.get("name")
    if not name: return jsonify({"error": "Thiếu name"}), 400
    out_file = PROCESSED / "conflict" / f"getconfig_{name}_{uuid.uuid4().hex[:8]}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    r = run_cmd([sys.executable, "structured_data_pipeline.py", "get-config-web", name,
                 "--out", str(out_file)], timeout=60)
    if out_file.exists():
        try:
            result = json.loads(out_file.read_text(encoding="utf-8"))
            out_file.unlink(missing_ok=True)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": r.get("stderr") or r.get("stdout") or "Không đọc được cấu hình"}), 500

@app.route("/admin/apply-table-dtypes", methods=["POST"])
def apply_table_dtypes():
    """Ép lại dtype cho file CSV của 1 bảng ĐÃ CÓ theo column_dtypes trong
    registry. Dùng sau khi admin đổi dtype trong panel 'Cấu hình cho chatbot'
    — nếu không ép lại, file .csv vẫn lưu dạng cũ dù registry đã khai mới."""
    err = _check_auth()
    if err: return err
    _cleanup_old_jobs()
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Thiếu name"}), 400

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "log": [], "ok": None, "returncode": None,
                          "table_name": name, "step": "apply-dtypes",
                          "started_at": time.time(), "result": None}

    def worker():
        with _jobs_lock:
            _jobs[job_id]["log"].append(f"▶ Ép lại dtype các cột cho bảng '{name}'...")
            _jobs[job_id]["status"] = "running"
        out_file = PROCESSED / "conflict" / f"applydtypes_{name}_{job_id[:8]}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "structured_data_pipeline.py", "apply-dtypes",
               name, "--out", str(out_file)]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"; env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="replace",
                                bufsize=1, env=env)
        for line in iter(proc.stdout.readline, ""):
            with _jobs_lock: _jobs[job_id]["log"].append(line.rstrip("\n"))
        proc.wait(timeout=300)
        if out_file.exists():
            try:
                _jobs[job_id]["result"] = json.loads(out_file.read_text(encoding="utf-8"))
            except Exception as e:
                with _jobs_lock: _jobs[job_id]["log"].append(f"[LỖI parse result] {e}")
        with _jobs_lock:
            _jobs[job_id]["ok"] = (proc.returncode == 0)
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["returncode"] = proc.returncode

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/admin/update-table-config", methods=["POST"])
def update_table_config():
    err = _check_auth()
    if err: return err
    _cleanup_old_jobs()
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    metadata = body.get("metadata")
    if not name or metadata is None:
        return jsonify({"error": "Thiếu name hoặc metadata"}), 400

    meta_file = PROCESSED / "conflict" / f"newconfig_{name}_{uuid.uuid4().hex[:8]}.json"
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "log": [], "ok": None, "returncode": None,
                          "table_name": name, "step": "update-table-config",
                          "started_at": time.time(), "result": None}

    def worker():
        with _jobs_lock:
            _jobs[job_id]["log"].append(f"▶ Cập nhật cấu hình bảng '{name}'...")
            _jobs[job_id]["status"] = "running"
        out_file = PROCESSED / "conflict" / f"updateconfig_{name}_{job_id[:8]}.json"
        cmd = [sys.executable, "structured_data_pipeline.py", "update-config-web", name,
               "--metadata", str(meta_file), "--out", str(out_file)]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"; env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="replace",
                                bufsize=1, env=env)
        for line in iter(proc.stdout.readline, ""):
            with _jobs_lock: _jobs[job_id]["log"].append(line.rstrip("\n"))
        proc.wait(timeout=60)

        # Đọc result JSON (nếu có) TRƯỚC khi set done.
        result_obj = {}
        if out_file.exists():
            try:
                parsed = json.loads(out_file.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    result_obj = parsed
                else:
                    result_obj = {"subprocess_result": parsed}
            except Exception as e:
                with _jobs_lock:
                    _jobs[job_id]["log"].append(f"[LỖI parse result] {e}")

        # [P10] Reload api8018 để matcher đọc lại registry.json + CSV mới.
        # CHỈ khi subprocess returncode == 0. Tính TRƯỚC khi set status="done"
        # để admin polling không thấy "done" mà thiếu result.reload_status.
        rc = proc.returncode
        reload_status = "skipped: returncode != 0" if rc != 0 else "skipped"
        if rc == 0:
            try:
                import urllib.request
                api_token = os.getenv("API_AUTH_TOKEN")
                api_base  = os.getenv("API8018_BASE_URL", "http://127.0.0.1:8018")
                if not api_token:
                    reload_status = "failed: API_AUTH_TOKEN không có trong .env"
                else:
                    req = urllib.request.Request(
                        f"{api_base}/admin/reload-index",
                        method="POST", headers={"X-API-Key": api_token},
                    )
                    urllib.request.urlopen(req, timeout=15)
                    reload_status = "ok"
            except Exception as e:
                reload_status = f"failed: {e}"

        # Ghi result + status="done" TRONG CÙNG block _jobs_lock.
        with _jobs_lock:
            result_obj["reload_status"] = reload_status
            _jobs[job_id]["result"] = result_obj
            _jobs[job_id]["ok"] = (rc == 0)
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["returncode"] = rc

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


FAQ_LOG_PATH = PROCESSED / "conflict" / "faq_log.json"
FEEDBACK_LOG_PATH = PROCESSED / "conflict" / "feedback_log.json"


def _read_faq() -> dict:
    """Đọc FAQ log KHÔNG khóa — dùng cho GET list_faq (chỉ đọc).
    Nếu file hỏng: vẫn raise (đã rename .corrupt.<ts>) — endpoint GET
    tương ứng bắt exception và trả 500."""
    return read_json_safe(FAQ_LOG_PATH, dict)

def _read_feedback() -> list:
    return read_json_safe(FEEDBACK_LOG_PATH, list)

def _update_faq(updater):
    return update_json(FAQ_LOG_PATH, dict, updater)

def _update_feedback(updater):
    return update_json(FEEDBACK_LOG_PATH, list, updater)

def _faq_has_unresolved_warning(item: dict) -> bool:
    """Warning còn hiệu lực nếu:
      - Có ít nhất 1 warning trong entry, VÀ
      - Chưa được admin đánh dấu xử lý (warning_resolved_at), HOẶC
        có warning MỚI xuất hiện SAU mốc đã xử lý.
    """
    warns = item.get("answer_warnings") or []
    if not warns:
        return False
    resolved = item.get("warning_resolved_at")
    if not resolved:
        return True
    last_ask = item.get("last_asked_at") or ""
    return last_ask > resolved


@app.route("/admin/faq", methods=["GET"])
def list_faq():
    err = _check_auth()
    if err: return err
    q = (request.args.get("q") or "").strip().lower()
    status = request.args.get("status") or "all"
    warning_filter = request.args.get("warning") or "all"
    page = max(1, int(request.args.get("page", 1)))
    page_size = min(100, max(1, int(request.args.get("page_size", 20))))
    try:
        data = _read_faq()
    except JsonCorruptError as e:
        return jsonify({"error": f"FAQ log bị hỏng, đã chuyển sang *.corrupt.<ts>. "
                                  f"Phục hồi rồi thử lại. Chi tiết: {e}"}), 500
    for v in data.values():
        v["has_unresolved_warning"] = _faq_has_unresolved_warning(v)

    items = [{"key": k, **v} for k, v in data.items()]
    if q:
        items = [it for it in items if q in it.get("question_display", "").lower()
                 or q in (it.get("last_asker_name") or "").lower()
                 or q in (it.get("last_asker_email") or "").lower()
                 or q in (it.get("last_answer") or "").lower()]
    total_chua_sua = sum(1 for it in data.values() if it.get("status", "chua_sua") == "chua_sua")
    total_da_sua = sum(1 for it in data.values() if it.get("status") == "da_sua")
    total_da_phu_hop = sum(1 for it in data.values() if it.get("status") == "da_phu_hop")
    total_with_warnings = sum(1 for it in data.values() if it.get("has_unresolved_warning"))

    if status != "all":
        items = [it for it in items if it.get("status", "chua_sua") == status]
    if warning_filter == "yes":
        items = [it for it in items if it.get("has_unresolved_warning")]
    elif warning_filter == "no":
        items = [it for it in items if not it.get("has_unresolved_warning")]

    items.sort(key=lambda it: (
        0 if it.get("has_unresolved_warning") else 1,
        -it.get("count", 0)
    ))

    total = len(items)
    start = (page - 1) * page_size
    page_items = items[start:start + page_size]
    return jsonify({
        "items": page_items, "total": total, "page": page, "page_size": page_size,
        "total_all": len(data), "total_chua_sua": total_chua_sua,
        "total_da_sua": total_da_sua, "total_da_phu_hop": total_da_phu_hop,
        "total_with_warnings": total_with_warnings,
    })

@app.route("/admin/faq/status", methods=["POST"])
def update_faq_status():
    err = _check_auth()
    if err: return err
    body = request.get_json(silent=True) or {}
    key = body.get("key")
    status = body.get("status")
    note = body.get("note")
    if not key or status not in ("chua_sua", "da_sua", "da_phu_hop"):
        return jsonify({"error": "Thiếu key hoặc status không hợp lệ"}), 400

    def _apply(data):
        if key not in data:
            raise KeyError(key)
        data[key]["status"] = status
        if note is not None:
            data[key]["note"] = note
        # Ghi mốc đã xử lý để badge ⚠️ không còn hiện nữa — warning cũ vẫn
        # giữ trong entry để audit.
        if status in ("da_sua", "da_phu_hop"):
            data[key]["warning_resolved_at"] = datetime.now(timezone.utc).isoformat()
        elif status == "chua_sua":
            data[key].pop("warning_resolved_at", None)
        return data

    try:
        data = _update_faq(_apply)
    except KeyError:
        return jsonify({"error": "Không tìm thấy câu hỏi này"}), 404
    except JsonCorruptError as e:
        return jsonify({"error": f"FAQ log hỏng: {e}"}), 500
    return jsonify({"ok": True, "key": key, "item": data[key]})

@app.route("/admin/faq/push-to-rag", methods=["POST"])
def push_faq_to_rag():
    """Admin sửa câu trả lời → đưa nội dung đã chỉnh vào RAG để lần sau
    chatbot không sai lại. Dùng CÙNG cơ chế với /admin/feedback/push-to-rag
    (tạo file .md tạm → check-file → chunk hóa → rebuild FAISS).
    """
    err = _check_auth()
    if err: return err
    _cleanup_old_jobs()
    body = request.get_json(silent=True) or {}
    faq_key = (body.get("key") or "").strip()
    title   = (body.get("title") or "").strip()
    content = (body.get("content") or "").strip()
    if not faq_key or not title or not content:
        return jsonify({"error": "Thiếu key, title hoặc content"}), 400

    # Xác nhận entry FAQ tồn tại
    faq_data = _load_faq()
    if faq_key not in faq_data:
        return jsonify({"error": "Không tìm thấy câu hỏi này trong FAQ log"}), 404

    base_name = re.sub(r"[/\\\"']", "", title)
    base_name = re.sub(r"\s+", " ", base_name).strip()
    tmp_dir = RAW_FOLDERS["md"] / "_faq_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_md = tmp_dir / f"{base_name}.md"
    tmp_md.write_text(content, encoding="utf-8")

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued", "log": [], "ok": None, "returncode": None,
            "base_name": base_name, "step": "push-faq-to-rag",
            "started_at": time.time(), "result": None,
        }

    def worker():
        with _jobs_lock:
            _jobs[job_id]["log"].append(
                f"▶ Đưa nội dung đã sửa từ FAQ '{faq_key}' vào RAG, "
                f"tên văn bản: '{base_name}'...")
            _jobs[job_id]["status"] = "running"
        cmd = [sys.executable, "conflict_detection.py",
               "check-file", str(tmp_md), base_name, "--auto=safe"]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"; env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="replace",
                                bufsize=1, env=env)
        for line in iter(proc.stdout.readline, ""):
            with _jobs_lock: _jobs[job_id]["log"].append(line.rstrip("\n"))
        proc.wait(timeout=1800)
        ok = proc.returncode == 0

        # Sau khi đẩy vào RAG thành công → cập nhật FAQ entry:
        #   - đánh dấu đã sửa + clear warnings + ghi mốc
        if ok:
            def _apply(data):
                if faq_key not in data:
                    return data
                item = data[faq_key]
                item["status"] = "da_sua"
                item["warning_resolved_at"] = datetime.now(timezone.utc).isoformat()
                item["pushed_base_name"] = base_name
                item["edited_title"] = title
                item["edited_content"] = content
                man = (f"[AUTO {datetime.now().strftime('%Y-%m-%d %H:%M')}] "
                        f"Đã đưa vào RAG: '{base_name}'")
                item["note"] = ((item.get("note") or "").strip() + "\n" + man).strip()
                return data
            try:
                _update_faq(_apply)
            except JsonCorruptError:
                pass  # đã log trong json_store; job vẫn thành công (RAG đã rebuild)
        with _jobs_lock:
            _jobs[job_id]["ok"] = ok
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["returncode"] = proc.returncode

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "base_name": base_name})

@app.route("/admin/feedback", methods=["GET"])
def list_feedback():
    err = _check_auth()
    if err: return err
    q = (request.args.get("q") or "").strip().lower()
    status = request.args.get("status") or "all"
    try:
        page = max(1, int(request.args.get("page", 1)))
        page_size = min(100, max(1, int(request.args.get("page_size", 20))))
    except (TypeError, ValueError):
        return jsonify({"error": "page/page_size không hợp lệ"}), 400

    try:
        items = _read_feedback()
    except JsonCorruptError as e:
        return jsonify({"error": f"Feedback log bị hỏng, đã chuyển sang *.corrupt.<ts>. "
                                  f"Phục hồi rồi thử lại. Chi tiết: {e}"}), 500

    total_chua_xu_ly = sum(1 for it in items if it.get("status", "chua_xu_ly") == "chua_xu_ly")
    total_da_xu_ly   = sum(1 for it in items if it.get("status") == "da_xu_ly")

    if q:
        items = [it for it in items if q in (it.get("question") or "").lower()
                 or q in (it.get("reason") or "").lower()
                 or q in (it.get("asker_name") or "").lower()
                 or q in (it.get("asker_email") or "").lower()]
    if status != "all":
        items = [it for it in items if it.get("status", "chua_xu_ly") == status]

    total = len(items)
    start = (page - 1) * page_size
    page_items = items[start:start + page_size]
    return jsonify({
        "items": page_items, "total": total, "page": page, "page_size": page_size,
        "total_chua_xu_ly": total_chua_xu_ly, "total_da_xu_ly": total_da_xu_ly,
    })


@app.route("/admin/feedback/status", methods=["POST"])
def update_feedback_status():
    err = _check_auth()
    if err: return err
    body = request.get_json(silent=True) or {}
    fid = body.get("id")
    status = body.get("status")
    if not fid or status not in ("chua_xu_ly", "da_xu_ly"):
        return jsonify({"error": "Thiếu id hoặc status không hợp lệ"}), 400

    def _apply(items):
        target = None
        for it in items:
            if it.get("id") == fid:
                target = it
                break
        if target is None:
            raise KeyError(fid)
        target["status"] = status
        return items
    try:
        items = _update_feedback(_apply)
    except KeyError:
        return jsonify({"error": "Không tìm thấy phản hồi này"}), 404
    except JsonCorruptError as e:
        return jsonify({"error": f"Feedback log hỏng: {e}"}), 500
    item = next(it for it in items if it.get("id") == fid)
    return jsonify({"ok": True, "item": item})


@app.route("/admin/feedback/push-to-rag", methods=["POST"])
def push_feedback_to_rag():
    err = _check_auth()
    if err: return err
    _cleanup_old_jobs()
    body = request.get_json(silent=True) or {}
    fid = body.get("id")
    title = (body.get("title") or "").strip()
    content = (body.get("content") or "").strip()
    if not fid or not title or not content:
        return jsonify({"error": "Thiếu id, title hoặc content"}), 400
    items = _load_feedback()
    item = next((it for it in items if it.get("id") == fid), None)
    if not item:
        return jsonify({"error": "Không tìm thấy phản hồi này"}), 404

    base_name = re.sub(r"[/\\\"']", "", title)
    base_name = re.sub(r"\s+", " ", base_name).strip()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_dir = RAW_FOLDERS["md"] / "_feedback_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_md = tmp_dir / f"{base_name}.md"
    tmp_md.write_text(content, encoding="utf-8")

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "log": [], "ok": None, "returncode": None,
                          "base_name": base_name, "step": "push-feedback-to-rag",
                          "started_at": time.time(), "result": None}

    def worker():
        with _jobs_lock:
            _jobs[job_id]["log"].append(f"▶ Đưa nội dung đã sửa từ phản hồi '{fid}' vào RAG, tên văn bản: "
                                          f"'{base_name}'...")
            _jobs[job_id]["status"] = "running"
        cmd = [sys.executable, "conflict_detection.py", "check-file", str(tmp_md), base_name, "--auto=safe"]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"; env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="replace",
                                bufsize=1, env=env)
        for line in iter(proc.stdout.readline, ""):
            with _jobs_lock: _jobs[job_id]["log"].append(line.rstrip("\n"))
        proc.wait(timeout=1800)
        ok = proc.returncode == 0
        if ok:
            def _apply(items):
                for it in items:
                    if it.get("id") == fid:
                        it["status"] = "da_xu_ly"
                        it["edited_title"] = title
                        it["edited_content"] = content
                        it["pushed_base_name"] = base_name
                        break
                return items
            try:
                _update_feedback(_apply)
            except JsonCorruptError as e:
                with _jobs_lock:
                    _jobs[job_id]["log"].append(
                        f"[CẢNH BÁO] Đã đẩy vào RAG thành công nhưng KHÔNG ghi được "
                        f"trạng thái 'da_xu_ly' vào feedback log: {e}"
                    )
        with _jobs_lock:
            _jobs[job_id]["ok"] = ok
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["returncode"] = proc.returncode

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "base_name": base_name})


@app.route("/admin/pending-cross-conflicts", methods=["GET"])
def pending_cross_conflicts():
    err = _check_auth()
    if err: return err
    out_file = PROCESSED / "conflict" / f"pendingcross_{uuid.uuid4().hex[:8]}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    r = run_cmd([sys.executable, "conflict_detection.py", "list-pending-cross-web",
                 "--out", str(out_file)], timeout=60)
    if out_file.exists():
        try:
            result = json.loads(out_file.read_text(encoding="utf-8"))
            out_file.unlink(missing_ok=True)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": r.get("stderr") or r.get("stdout") or "Không đọc được danh sách"}), 500


@app.route("/admin/apply-cross-conflict-decisions", methods=["POST"])
def apply_cross_conflict_decisions():
    err = _check_auth()
    if err: return err
    _cleanup_old_jobs()
    body = request.get_json(silent=True) or {}
    base_name = (body.get("base_name") or "").strip()
    decisions = body.get("decisions")
    if not base_name or not decisions:
        return jsonify({"error": "Thiếu base_name hoặc decisions"}), 400

    dec_file = PROCESSED / "conflict" / f"crossdecisions_{base_name}_{uuid.uuid4().hex[:8]}.json"
    dec_file.parent.mkdir(parents=True, exist_ok=True)
    dec_file.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "log": [], "ok": None, "returncode": None,
                          "base_name": base_name, "step": "apply-cross-conflict",
                          "started_at": time.time(), "result": None}

    def worker():
        with _jobs_lock:
            _jobs[job_id]["log"].append(f"▶ Áp dụng {len(decisions)} quyết định cảnh báo chéo cho '{base_name}'...")
            _jobs[job_id]["status"] = "running"
        out_file = PROCESSED / "conflict" / f"applycross_{base_name}_{job_id[:8]}.json"
        cmd = [sys.executable, "conflict_detection.py", "apply-pending-cross-web", base_name,
               "--decisions", str(dec_file), "--out", str(out_file)]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"; env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="replace",
                                bufsize=1, env=env)
        for line in iter(proc.stdout.readline, ""):
            with _jobs_lock: _jobs[job_id]["log"].append(line.rstrip("\n"))
        proc.wait(timeout=120)
        if out_file.exists():
            try:
                _jobs[job_id]["result"] = json.loads(out_file.read_text(encoding="utf-8"))
            except Exception as e:
                with _jobs_lock: _jobs[job_id]["log"].append(f"[LỖI parse result] {e}")
        with _jobs_lock:
            _jobs[job_id]["ok"] = (proc.returncode == 0)
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["returncode"] = proc.returncode

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/admin/table-backups", methods=["GET"])
def table_backups():
    err = _check_auth()
    if err: return err
    name = request.args.get("name")
    if not name: return jsonify({"error": "Thiếu name"}), 400
    try:
        registry = json.loads((BASE / "registry.json").read_text(encoding="utf-8"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    if name not in registry:
        return jsonify({"error": f"Bảng '{name}' không tồn tại"}), 404
    path = BASE / registry[name]["path"]
    backups = sorted(path.parent.glob(f"{path.stem}.bak_*"), key=lambda p: p.stat().st_mtime) \
        if path.parent.exists() else []
    return jsonify({"table_name": name, "current_path": registry[name]["path"], "backups": [
        {"filename": b.name, "mtime": datetime.fromtimestamp(b.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
         "size_kb": round(b.stat().st_size / 1024, 1)}
        for b in backups
    ]})


@app.route("/admin/restore-table-backup", methods=["POST"])
def restore_table_backup():
    err = _check_auth()
    if err: return err
    _cleanup_old_jobs()
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    backup_filename = (body.get("backup_filename") or "").strip()
    if not name or not backup_filename:
        return jsonify({"error": "Thiếu name hoặc backup_filename"}), 400
    if "/" in backup_filename or "\\" in backup_filename:
        return jsonify({"error": "backup_filename không hợp lệ"}), 400

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "log": [], "ok": None, "returncode": None,
                          "table_name": name, "step": "restore-backup", "started_at": time.time()}
    cmd = [sys.executable, "structured_data_pipeline.py", "restore-backup", name, backup_filename]
    threading.Thread(target=_run_job_with_autorespond, args=(job_id, cmd),
                      kwargs={"yn_default": "y", "choice_default": "1", "timeout": 300},
                      daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/admin/remove-structured-table", methods=["POST"])
def remove_structured_table():
    err = _check_auth()
    if err: return err
    _cleanup_old_jobs()
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Thiếu name"}), 400

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "log": [], "ok": None, "returncode": None,
                          "table_name": name, "step": "remove-table", "started_at": time.time()}
    cmd = [sys.executable, "structured_data_pipeline.py", "remove-table", name]
    threading.Thread(target=_run_job_with_autorespond, args=(job_id, cmd),
                      kwargs={"yn_default": "y", "choice_default": "1", "timeout": 120},
                      daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


def _read_structured_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Không hỗ trợ định dạng '{suffix}'")


_PREVIEW_TTL_SECONDS = 2 * 3600


def _cleanup_old_previews():
    now = time.time()
    for f in STRUCT_UPLOAD_DIR.glob("_preview_*"):
        try:
            if now - f.stat().st_mtime > _PREVIEW_TTL_SECONDS:
                f.unlink()
        except Exception:
            pass


@app.route("/admin/preview-structured-table", methods=["POST"])
def preview_structured_table():
    err = _check_auth()
    if err: return err
    _cleanup_old_previews()
    if "file" not in request.files: return jsonify({"error": "Thiếu file"}), 400
    f = request.files["file"]
    table_name = (request.form.get("table_name") or "").strip()
    if not f.filename: return jsonify({"error": "Thiếu file"}), 400
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ("csv", "xlsx", "xls"): return jsonify({"error": "Chỉ nhận .csv/.xlsx/.xls"}), 400

    token = f"_preview_{uuid.uuid4().hex}.{ext}"
    saved = STRUCT_UPLOAD_DIR / token
    f.save(str(saved))

    try:
        df = _read_structured_file(saved)
    except Exception as e:
        saved.unlink(missing_ok=True)
        return jsonify({"error": f"Không đọc được file: {e}"}), 400
    # Phát hiện dtype + cảnh báo cột nguy hiểm (mobile/CCCD/mã)
    from structured_conflict_detection import phat_hien_dtype
    dtype_info = phat_hien_dtype(df)
    columns = [str(c).strip() for c in df.columns]
    n_rows = len(df)
    sample = df.head(10).fillna("").astype(str)
    sample.columns = columns
    sample_rows = sample.to_dict("records")

    try:
        registry = json.loads((BASE / "registry.json").read_text(encoding="utf-8"))
    except Exception:
        registry = {}

    existing_table = table_name in registry
    existing_primary_key = registry[table_name].get("primary_key") if existing_table else None
    suggested_table = None
    if not existing_table:
        new_cols = set(columns)
        best = None
        for name, cfg in registry.items():
            p = BASE / cfg.get("path", "")
            if not p.exists():
                continue
            try:
                old_cols = set(str(c).strip() for c in pd.read_csv(p, nrows=0).columns)
            except Exception:
                continue
            if not old_cols:
                continue
            ratio = len(new_cols & old_cols) / len(new_cols | old_cols)
            if ratio >= 0.7 and (best is None or ratio > best[1]):
                best = (name, ratio)
        if best:
            suggested_table = {"name": best[0], "ratio": round(best[1], 2)}

    existing_dtypes = {}
    if existing_table:
        existing_dtypes = registry[table_name].get("column_dtypes") or {}

    return jsonify({
        "ok": True, "preview_token": token, "columns": columns, "n_rows": n_rows,
        "sample_rows": sample_rows, "existing_table": existing_table,
        "existing_primary_key": existing_primary_key, "suggested_table": suggested_table,
        "dtype_info": dtype_info,                    # ← MỚI
        "existing_column_dtypes": existing_dtypes,   # ← MỚI
    })


@app.route("/admin/analyze-structured-upload", methods=["POST"])
def analyze_structured_upload():
    err = _check_auth()
    if err: return err
    _cleanup_old_jobs()
    body = request.get_json(silent=True) or {}
    preview_token = (body.get("preview_token") or "").strip()
    table_name = (body.get("table_name") or "").strip()
    key_cols = body.get("key_cols") or []
    force_new = bool(body.get("force_new"))
    dtype_overrides = body.get("dtype_overrides") or {}
    if not preview_token or not table_name:
        return jsonify({"error": "Thiếu preview_token hoặc table_name"}), 400
    if not preview_token.startswith("_preview_") or "/" in preview_token or "\\" in preview_token:
        return jsonify({"error": "preview_token không hợp lệ"}), 400
    saved = STRUCT_UPLOAD_DIR / preview_token
    if not saved.exists():
        return jsonify({"error": "File preview không tồn tại hoặc đã hết hạn - chọn lại file."}), 400

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued", "log": [], "ok": None, "returncode": None,
            "table_name": table_name, "step": "analyze-table",
            "started_at": time.time(), "result": None,
        }

    def worker():
        with _jobs_lock:
            _jobs[job_id]["log"].append(
                f"▶ Phân tích bảng '{table_name}' - tiền xử lý + so khớp khóa + đối chiếu "
                f"chéo với RAG (có gọi LLM cho các dòng mới/thay đổi, có thể mất vài phút)...")
            _jobs[job_id]["status"] = "running"
        out_file = PROCESSED / "conflict" / f"analyze_table_{table_name}_{job_id[:8]}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "structured_data_pipeline.py", "analyze-web",
               str(saved), table_name, "--out", str(out_file)]
        if key_cols:
            cmd.append("--key-cols=" + ",".join(key_cols))
        if force_new:
            cmd.append("--force-new")
        if dtype_overrides:
            cmd.append("--dtype-overrides=" + json.dumps(dtype_overrides, ensure_ascii=False))
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"; env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="replace",
                                bufsize=1, env=env)
        for line in iter(proc.stdout.readline, ""):
            with _jobs_lock: _jobs[job_id]["log"].append(line.rstrip("\n"))
        proc.wait(timeout=3600)
        if out_file.exists():
            try:
                _jobs[job_id]["result"] = json.loads(out_file.read_text(encoding="utf-8"))
            except Exception as e:
                with _jobs_lock: _jobs[job_id]["log"].append(f"[LỖI parse result] {e}")
        with _jobs_lock:
            _jobs[job_id]["ok"] = (proc.returncode == 0)
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["returncode"] = proc.returncode

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "table_name": table_name})


@app.route("/admin/recheck-cross-rag", methods=["POST"])
def recheck_cross_rag():
    err = _check_auth()
    if err: return err
    _cleanup_old_jobs()
    body = request.get_json(silent=True) or {}
    table_name = (body.get("table_name") or "").strip()
    staging_token = (body.get("staging_token") or "").strip()
    name_columns = body.get("name_columns") or []
    role_columns = body.get("role_columns") or []
    group_columns = body.get("group_columns") or []
    if not table_name or not staging_token or not name_columns:
        return jsonify({"error": "Thiếu table_name, staging_token hoặc name_columns"}), 400

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "log": [], "ok": None, "returncode": None,
                          "table_name": table_name, "step": "recheck-cross-rag",
                          "started_at": time.time(), "result": None}

    def worker():
        with _jobs_lock:
            _jobs[job_id]["log"].append(f"▶ Đối chiếu chéo với RAG cho bảng mới '{table_name}' "
                                          f"(dùng cột tên: {name_columns}, gọi LLM - có thể mất vài phút)...")
            _jobs[job_id]["status"] = "running"
        out_file = PROCESSED / "conflict" / f"recheck_cross_{table_name}_{job_id[:8]}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "structured_data_pipeline.py", "recheck-cross-web",
               table_name, staging_token, "--name-cols=" + ",".join(name_columns), "--out", str(out_file)]
        if role_columns:
            cmd.append("--role-cols=" + ",".join(role_columns))
        if group_columns:
            cmd.append("--group-cols=" + ",".join(group_columns))
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"; env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="replace",
                                bufsize=1, env=env)
        for line in iter(proc.stdout.readline, ""):
            with _jobs_lock: _jobs[job_id]["log"].append(line.rstrip("\n"))
        proc.wait(timeout=1800)
        if out_file.exists():
            try:
                _jobs[job_id]["result"] = json.loads(out_file.read_text(encoding="utf-8"))
            except Exception as e:
                with _jobs_lock: _jobs[job_id]["log"].append(f"[LỖI parse result] {e}")
        with _jobs_lock:
            _jobs[job_id]["ok"] = (proc.returncode == 0)
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["returncode"] = proc.returncode

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/admin/apply-structured-decisions", methods=["POST"])
def apply_structured_decisions():
    err = _check_auth()
    if err: return err
    _cleanup_old_jobs()
    body = request.get_json(silent=True) or {}
    table_name = (body.get("table_name") or "").strip()
    decisions = body.get("decisions")
    if not table_name or decisions is None:
        return jsonify({"error": "Thiếu table_name hoặc decisions"}), 400

    dec_file = PROCESSED / "conflict" / f"decisions_table_{table_name}_{uuid.uuid4().hex[:8]}.json"
    dec_file.parent.mkdir(parents=True, exist_ok=True)
    dec_file.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued", "log": [], "ok": None, "returncode": None,
            "table_name": table_name, "step": "apply-table",
            "started_at": time.time(), "result": None,
        }

    def worker():
        with _jobs_lock:
            n_thay_doi = len(decisions.get("thay_doi_decisions", []))
            n_cheo = len(decisions.get("canh_bao_cheo_decisions", []))
            _jobs[job_id]["log"].append(
                f"▶ Áp dụng {n_thay_doi} quyết định 'thay đổi' + {n_cheo} quyết định "
                f"'cảnh báo chéo' vào bảng '{table_name}'...")
            _jobs[job_id]["status"] = "running"
        out_file = PROCESSED / "conflict" / f"apply_table_{table_name}_{job_id[:8]}.json"
        cmd = [sys.executable, "structured_data_pipeline.py", "apply-web",
               table_name, "--decisions", str(dec_file), "--out", str(out_file)]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"; env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                text=True, encoding="utf-8", errors="replace",
                                bufsize=1, env=env)
        for line in iter(proc.stdout.readline, ""):
            with _jobs_lock: _jobs[job_id]["log"].append(line.rstrip("\n"))
        proc.wait(timeout=1800)
        if out_file.exists():
            try:
                _jobs[job_id]["result"] = json.loads(out_file.read_text(encoding="utf-8"))
            except Exception as e:
                with _jobs_lock: _jobs[job_id]["log"].append(f"[LỖI parse result] {e}")
        with _jobs_lock:
            _jobs[job_id]["ok"] = (proc.returncode == 0)
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["returncode"] = proc.returncode

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "table_name": table_name})


# ============================================================
# TEST CASES
# ============================================================
@app.route("/admin/test-cases", methods=["GET"])
def list_test_cases():
    err = _check_auth()
    if err: return err
    if not TEST_CASES_PATH.exists(): return jsonify({"cases": [], "note": "Chưa có test_cases.json"})
    try: return jsonify({"cases": json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))})
    except Exception as e: return jsonify({"error": str(e)}), 500


@app.route("/admin/run-test-case", methods=["POST"])
def run_test_case():
    """Trả job_id NGAY, chạy nền từng bước, stream log qua /admin/job/<id>."""
    err = _check_auth()
    if err: return err
    _cleanup_old_jobs()
    body = request.get_json(silent=True) or {}
    case_id = body.get("id")
    if not case_id:
        return jsonify({"error": "Thiếu id"}), 400
    try:
        cases = json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        return jsonify({"error": f"Không load test_cases.json: {e}"}), 500
    case = next((c for c in cases if c["id"] == case_id), None)
    if not case:
        return jsonify({"error": f"Test case '{case_id}' không tồn tại"}), 404

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued", "log": [], "ok": None, "returncode": None,
            "step": "run-test-case", "case_id": case_id,
            "case_name": case.get("name"), "started_at": time.time(), "result": None,
        }
    threading.Thread(target=_run_test_case_job, args=(job_id, case), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "case_id": case_id,
                    "case_name": case.get("name")})


def _run_test_case_job(job_id, case):
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"

    def _log(msg):
        with _jobs_lock:
            _jobs[job_id]["log"].append(msg)

    steps = case.get("steps", [])
    _log(f"▶ Test case {case.get('id')}: {case.get('name')}")
    _log(f"   {len(steps)} bước — KHÔNG tự rebuild giữa các bước, dồn 1 lần cuối.")
    _log("")

    tmp_dir = RAW_FOLDERS["md"] / "_test_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["CONFLICT_SKIP_AUTO_REBUILD"] = "1"

    step_logs = []
    for i, step in enumerate(steps, 1):
        base_name = step["base_name"]
        _log("")
        _log("=" * 60)
        _log(f"▶ [Bước {i}/{len(steps)}] base_name = {base_name}")
        _log("=" * 60)

        tmp_md = tmp_dir / f"{base_name}.md"
        tmp_md.write_text(step.get("md", ""), encoding="utf-8")
        cmd = [sys.executable, "conflict_detection.py",
               "check-file", str(tmp_md), base_name]

        proc = None
        step_lines = []
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(BASE),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
            )
            if step.get("stdin"):
                try:
                    proc.stdin.write(step["stdin"])
                    proc.stdin.flush()
                    proc.stdin.close()
                except Exception:
                    pass

            for line in iter(proc.stdout.readline, ""):
                line = line.rstrip("\n")
                _log("   " + line)
                step_lines.append(line)

            proc.wait(timeout=1800)
            step_logs.append({
                "base_name": base_name,
                "returncode": proc.returncode,
                "stdout": "\n".join(step_lines),
                "stderr": "",
            })
        except Exception as e:
            _log(f"[LỖI] {type(e).__name__}: {e}")
            step_logs.append({"base_name": base_name, "returncode": -1,
                              "stdout": "\n".join(step_lines), "stderr": str(e)})
            if proc: proc.kill()

    _log("")
    _log("=" * 60)
    _log("▶ Rebuild FAISS 1 lần duy nhất ở cuối (dồn mọi thay đổi)...")
    _log("=" * 60)
    r = run_cmd([sys.executable, "conflict_detection.py", "rebuild-index"], timeout=1800)
    for ln in (r.get("stdout") or "").splitlines(): _log("   " + ln)
    for ln in (r.get("stderr") or "").splitlines(): _log("   [STDERR] " + ln)

    with _jobs_lock:
        _jobs[job_id]["result"] = {
            "case_id": case.get("id"),
            "case_name": case.get("name"),
            "logs": step_logs,
            "verify_question": case.get("verify_question"),
            "expected_answer": case.get("expected_answer"),
            "rebuild_ok": r["ok"],
        }
        _jobs[job_id]["ok"] = r["ok"] and all(l["returncode"] == 0 for l in step_logs)
        _jobs[job_id]["status"] = "done"


@app.route("/admin/reset-test", methods=["POST"])
def reset_test():
    """Trả job_id NGAY, chạy nền, frontend poll /admin/job/<id> để xem log."""
    err = _check_auth()
    if err: return err
    _cleanup_old_jobs()
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued", "log": [], "ok": None, "returncode": None,
            "step": "reset-test", "started_at": time.time(), "result": None,
        }
    threading.Thread(target=_run_reset_test_job, args=(job_id,), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


def _run_reset_test_job(job_id):
    """Chạy nền, từng bước ghi log vào _jobs[job_id]['log']."""
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"

    def _log(msg):
        with _jobs_lock:
            _jobs[job_id]["log"].append(msg)

    def _dump(prefix, r):
        for ln in (r.get("stdout") or "").splitlines():
            _log(prefix + ln)
        for ln in (r.get("stderr") or "").splitlines():
            _log(prefix + "[STDERR] " + ln)

    _log("▶ Reset toàn bộ test data (có thể mất 5-15 phút)...")

    latest_r = run_cmd([sys.executable, "conflict_detection.py", "latest-snapshot"], timeout=30)
    latest = _parse_snapshot_name(latest_r.get("stdout"))
    if latest:
        _log(f"→ Tìm thấy snapshot gần nhất: '{latest}' - khôi phục TOÀN BỘ (chunks + manifest + "
             f"bảng cấu trúc + registry.json) từ đây...")
        r0 = run_cmd([sys.executable, "conflict_detection.py", "restore-snapshot", latest], timeout=300)
        _dump("   ", r0)
        _log("")
        _log("── Rebuild-index: nạp lại FAISS production sau khi khôi phục snapshot")
        r3 = run_cmd([sys.executable, "conflict_detection.py", "rebuild-index"], timeout=1800)
        _dump("   ", r3)
        removed = []
        tmp_dir = RAW_FOLDERS["md"] / "_test_tmp"
        if tmp_dir.exists():
            for f in tmp_dir.glob("*.md"):
                removed.append(f"raw/md/_test_tmp/{f.name}"); f.unlink()
        staging = BASE / "data" / "processed" / "tables_staging"
        if staging.exists():
            for f in staging.glob("test_*"):
                removed.append(f"tables_staging/{f.name}"); f.unlink()
        _log("")
        _log(f"✅ Reset xong (đã khôi phục từ snapshot '{latest}'). Dọn thêm {len(removed)} file tạm.")
        with _jobs_lock:
            _jobs[job_id]["result"] = {"cleanup_ok": r0["ok"], "prune_ok": True, "rebuild_ok": r3["ok"],
                                        "removed_files": removed, "restored_from_snapshot": latest}
            _jobs[job_id]["ok"] = r0["ok"] and r3["ok"]
            _jobs[job_id]["status"] = "done"
        return

    _log("⚠️  KHÔNG tìm thấy snapshot nào (chưa bấm Snapshot trước khi test) - CHỈ dọn được dữ liệu "
         "phía văn bản (test_*), KHÔNG hoàn tác được bất kỳ thay đổi nào đã ghi vào bảng cấu trúc "
         "(.csv). Nếu có test file .csv/.xlsx, kiểm tra lại bảng đó bằng tay hoặc dùng "
         "'Backup / Khôi phục' ở từng bảng trong tab Dữ liệu bảng.")

    _log("")
    _log("── [1/3] cleanup-test: xóa test_* khỏi markdown/, chunks/, manifest.json")
    r1 = run_cmd([sys.executable, "conflict_detection.py",
                  "cleanup-test", "test_", "--no-rebuild"], timeout=900)
    _dump("   ", r1)

    _log("")
    _log("── [2/3] prune: dọn manifest.json các mục mồ côi")
    r2 = run_cmd([sys.executable, "conflict_detection.py", "prune"], timeout=180)
    _dump("   ", r2)

    _log("")
    _log("── [3/3] rebuild-index: nạp lại FAISS production (bước chậm nhất)")
    r3 = run_cmd([sys.executable, "conflict_detection.py", "rebuild-index"], timeout=1800)
    _dump("   ", r3)

    tmp_dir = RAW_FOLDERS["md"] / "_test_tmp"
    removed = []
    if tmp_dir.exists():
        for f in tmp_dir.glob("*.md"):
            removed.append(f"raw/md/_test_tmp/{f.name}"); f.unlink()
    staging = BASE / "data" / "processed" / "tables_staging"
    if staging.exists():
        for f in staging.glob("test_*"):
            removed.append(f"tables_staging/{f.name}"); f.unlink()

    _log("")
    _log(f"✅ Reset xong. Dọn thêm {len(removed)} file tạm.")

    with _jobs_lock:
        _jobs[job_id]["result"] = {
            "cleanup_ok": r1["ok"], "prune_ok": r2["ok"], "rebuild_ok": r3["ok"],
            "removed_files": removed,
        }
        _jobs[job_id]["ok"] = r1["ok"] and r2["ok"] and r3["ok"]
        _jobs[job_id]["status"] = "done"


def _parse_snapshot_name(stdout):
    """Lấy tên snapshot từ stdout của 'conflict_detection.py latest-snapshot'.
    Chỉ nhận dòng đúng dạng YYYYMMDD_HHMMSS[_nhan]; bỏ qua cảnh báo/log lạc vào stdout."""
    import re as _re
    for ln in reversed((stdout or "").splitlines()):
        ln = ln.strip()
        if _re.match(r"^\d{8}_\d{6}(_\S+)?$", ln):
            return ln
    return ""


@app.route("/admin/latest-snapshot", methods=["GET"])
def latest_snapshot_info():
    """[FIX RESET] Cho UI biết Reset test SẼ khôi phục snapshot nào (chỉ đọc, không đổi gì)."""
    err = _check_auth()
    if err: return err
    r = run_cmd([sys.executable, "conflict_detection.py", "latest-snapshot"], timeout=30)
    name = _parse_snapshot_name(r.get("stdout")) if r["ok"] else ""
    created = None
    if name:
        try:
            created = datetime.strptime(name[:15], "%Y%m%d_%H%M%S").isoformat()
        except ValueError:
            created = None
    return jsonify({"ok": r["ok"], "name": name or None, "created": created})


@app.route("/admin/snapshot", methods=["POST"])
def take_snapshot():
    err = _check_auth()
    if err: return err
    body = request.get_json(silent=True) or {}
    label = body.get("label") or "before_test"
    r = run_cmd([sys.executable, "conflict_detection.py", "snapshot", label], timeout=180)
    return jsonify({"ok": r["ok"], "stdout": r["stdout"], "stderr": r["stderr"]})


if __name__ == "__main__":
    port = int(os.getenv("ADMIN_API_PORT", "8924"))
    print(f"→ Admin API: http://0.0.0.0:{port}")
    print(f"→ Frontend:  http://127.0.0.1:{port}/")
    app.run(host="0.0.0.0", port=port, debug=False)