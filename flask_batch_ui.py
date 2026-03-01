import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request


ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))


TASKS = {
    "setup": {
        "label": "Base Setup",
        "description": "Run setup.bat with optional python reinstall choice.",
        "kind": "bat_with_answers",
        "script": "setup.bat",
        "fields": [
            {
                "name": "reinstall_python",
                "label": "Reinstall python_embeded",
                "type": "select",
                "options": [
                    {"value": "n", "label": "No (default)"},
                    {"value": "y", "label": "Yes"},
                ],
                "default": "n",
            }
        ],
    },
    "install_trt_opt": {
        "label": "Install TensorRT (Optional)",
        "description": "Install TensorRT Python packages for TRT backend.",
        "kind": "bat",
        "script": "install_tensorrt_optional.bat",
        "fields": [],
    },
    "install_ffmpeg_opt": {
        "label": "Install FFmpeg (Optional)",
        "description": "Install FFmpeg using winget and update .env paths.",
        "kind": "bat",
        "script": "install_ffmpeg_optional.bat",
        "fields": [],
    },
    "download_convert_hf": {
        "label": "Download HF -> ONNX -> TRT",
        "description": "Download safetensors model from HF and convert to ONNX/TRT.",
        "kind": "python",
        "script": "download_convert_hf_to_trt.py",
        "fields": [
            {"name": "repo_id", "label": "Repo ID", "type": "text", "default": "ZhengPeng7/BiRefNet"},
            {"name": "width", "label": "Width", "type": "number", "default": "1024"},
            {"name": "height", "label": "Height", "type": "number", "default": "1024"},
            {"name": "batch", "label": "Batch", "type": "number", "default": "1"},
            {"name": "workspace_gb", "label": "TRT Workspace (GB)", "type": "number", "default": "6"},
            {"name": "download_workers", "label": "Download Workers", "type": "number", "default": "16"},
        ],
    },
    "torch_launcher": {
        "label": "Torch Launcher",
        "description": "Run torch launcher over files in workfolder.",
        "kind": "bat_with_answers",
        "script": "torch_launcher.bat",
        "fields": [
            {"name": "model_choice", "label": "Model #", "type": "number", "default": "1"},
            {"name": "width", "label": "Width", "type": "number", "default": "1024"},
            {"name": "height", "label": "Height", "type": "number", "default": "1024"},
            {
                "name": "mask_mode",
                "label": "Mask Mode",
                "type": "select",
                "options": [
                    {"value": "b", "label": "Binary"},
                    {"value": "s", "label": "Soft"},
                ],
                "default": "b",
            },
            {"name": "threshold", "label": "Binary Threshold", "type": "text", "default": "0.65"},
            {
                "name": "device",
                "label": "Device",
                "type": "select",
                "options": [
                    {"value": "auto", "label": "auto"},
                    {"value": "cuda", "label": "cuda"},
                    {"value": "cpu", "label": "cpu"},
                ],
                "default": "auto",
            },
        ],
    },
    "onnx_launcher": {
        "label": "ONNX Launcher",
        "description": "Run ONNX launcher over files in workfolder.",
        "kind": "bat_with_answers",
        "script": "onnx_launcher.bat",
        "fields": [
            {"name": "model_choice", "label": "ONNX #", "type": "number", "default": "1"},
            {
                "name": "mask_mode",
                "label": "Mask Mode",
                "type": "select",
                "options": [
                    {"value": "b", "label": "Binary"},
                    {"value": "s", "label": "Soft"},
                ],
                "default": "b",
            },
            {"name": "threshold", "label": "Binary Threshold", "type": "text", "default": "0.65"},
            {
                "name": "providers",
                "label": "Providers",
                "type": "select",
                "options": [
                    {"value": "auto", "label": "auto"},
                    {"value": "cuda", "label": "cuda"},
                    {"value": "cpu", "label": "cpu"},
                ],
                "default": "auto",
            },
        ],
    },
    "trt_launcher": {
        "label": "TensorRT Launcher",
        "description": "Run TRT launcher over files in workfolder.",
        "kind": "bat_with_answers",
        "script": "birefnet_trt_launcher.bat",
        "fields": [
            {"name": "model_choice", "label": "Engine #", "type": "number", "default": "1"},
            {
                "name": "mask_mode",
                "label": "Mask Mode",
                "type": "select",
                "options": [
                    {"value": "b", "label": "Binary"},
                    {"value": "s", "label": "Soft"},
                ],
                "default": "b",
            },
            {"name": "threshold", "label": "Binary Threshold", "type": "text", "default": "0.65"},
        ],
    },
}


JOBS = {}
JOBS_LOCK = threading.Lock()
MAX_LOG_LINES = 5000


def _new_job(task_id: str, command: list[str], input_blob: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "task_id": task_id,
            "command": command,
            "status": "running",
            "created_at": time.time(),
            "started_at": time.time(),
            "ended_at": None,
            "returncode": None,
            "logs": [],
            "input_blob": input_blob,
            "pid": None,
        }
    return job_id


def _append_log(job_id: str, line: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["logs"].append(line.rstrip("\n"))
        if len(job["logs"]) > MAX_LOG_LINES:
            job["logs"] = job["logs"][-MAX_LOG_LINES:]


def _set_job_done(job_id: str, returncode: int) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        if job["status"] != "stopped":
            job["status"] = "done" if returncode == 0 else "failed"
        job["returncode"] = returncode
        job["ended_at"] = time.time()


def _set_job_pid(job_id: str, pid: int) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            job["pid"] = pid


def _run_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        command = job["command"]
        input_blob = job["input_blob"]

    try:
        proc = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        _set_job_pid(job_id, proc.pid)

        if input_blob and proc.stdin:
            proc.stdin.write(input_blob)
            proc.stdin.flush()
            proc.stdin.close()

        if proc.stdout:
            for line in proc.stdout:
                _append_log(job_id, line)

        ret = proc.wait()
        _set_job_done(job_id, ret)
    except Exception as exc:
        _append_log(job_id, f"[runner-error] {exc}")
        _set_job_done(job_id, 1)


def _build_command_and_input(task_id: str, payload: dict) -> tuple[list[str], str]:
    task = TASKS[task_id]
    kind = task["kind"]

    if kind == "bat":
        return ["cmd", "/c", task["script"]], ""

    if kind == "python":
        python_exe = str(ROOT / "python_embeded" / "python.exe")
        cmd = [python_exe, task["script"]]
        for f in task["fields"]:
            key = f["name"]
            val = str(payload.get(key, f.get("default", ""))).strip()
            if not val:
                continue
            cmd.extend([f"--{key.replace('_', '-')}", val])
        return cmd, ""

    if kind == "bat_with_answers":
        answers = []
        if task_id == "setup":
            answers.append(str(payload.get("reinstall_python", "n")))
        elif task_id == "torch_launcher":
            answers.extend(
                [
                    str(payload.get("model_choice", "1")),
                    str(payload.get("width", "1024")),
                    str(payload.get("height", "1024")),
                    str(payload.get("mask_mode", "b")),
                ]
            )
            if str(payload.get("mask_mode", "b")).lower() != "s":
                answers.append(str(payload.get("threshold", "0.65")))
            answers.append(str(payload.get("device", "auto")))
        elif task_id == "onnx_launcher":
            answers.extend(
                [
                    str(payload.get("model_choice", "1")),
                    str(payload.get("mask_mode", "b")),
                ]
            )
            if str(payload.get("mask_mode", "b")).lower() != "s":
                answers.append(str(payload.get("threshold", "0.65")))
            answers.append(str(payload.get("providers", "auto")))
        elif task_id == "trt_launcher":
            answers.extend(
                [
                    str(payload.get("model_choice", "1")),
                    str(payload.get("mask_mode", "b")),
                ]
            )
            if str(payload.get("mask_mode", "b")).lower() != "s":
                answers.append(str(payload.get("threshold", "0.65")))
        else:
            raise ValueError(f"Unsupported bat_with_answers task: {task_id}")

        input_blob = "\n".join(answers) + "\n"
        return ["cmd", "/c", task["script"]], input_blob

    raise ValueError(f"Unsupported task kind: {kind}")


@app.get("/")
def index():
    return render_template("index.html", tasks=TASKS)


@app.get("/api/tasks")
def api_tasks():
    return jsonify(TASKS)


@app.post("/api/run")
def api_run():
    data = request.get_json(force=True, silent=True) or {}
    task_id = data.get("task_id", "").strip()
    if task_id not in TASKS:
        return jsonify({"ok": False, "error": "Unknown task_id"}), 400

    try:
        command, input_blob = _build_command_and_input(task_id, data.get("params", {}))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    job_id = _new_job(task_id, command, input_blob)
    t = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    t.start()
    return jsonify({"ok": True, "job_id": job_id})


@app.get("/api/jobs")
def api_jobs():
    with JOBS_LOCK:
        jobs = sorted(JOBS.values(), key=lambda x: x["created_at"], reverse=True)
    return jsonify(jobs)


@app.get("/api/jobs/<job_id>")
def api_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    return jsonify(job)


@app.post("/api/jobs/<job_id>/stop")
def api_stop_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "job not found"}), 404
        pid = job.get("pid")
        job["status"] = "stopped"
        job["ended_at"] = time.time()

    if pid:
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)
        except Exception:
            pass
    return jsonify({"ok": True})


if __name__ == "__main__":
    host = os.getenv("FLASK_BATCH_UI_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_BATCH_UI_PORT", "7860"))
    debug = os.getenv("FLASK_BATCH_UI_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)

