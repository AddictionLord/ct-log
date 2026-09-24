#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["requests", "websocket-client"]
# ///
"""euler — bash na vzdáleném Jupyter serveru přes REST API + kernel websocket.

Konfigurace (env):
  EULER_URL    https://euler.mendelu.cz/jupyterlab (samostatný Jupyter Server, žádný Hub)
  EULER_TOKEN  token serveru (sdílený, spravuje admin; nikdy ho nevypisuj)

Veškerý výstup prochází _scrub(): EULER_TOKEN a MLFLOW_TRACKING_PASSWORD (i URL-encoded)
se nahradí [REDACTED]. Jupyter vkládá token do HTML stránek, proto se HTML těla nevypisují.

Příkazy:
  euler exec "<cmd>" [--timeout S]          synchronně, vrací stdout/stderr/exit code
  euler bg <name> "<cmd>" [--min-free-mb N] detached job -> ~/jobs/<name>/{run.sh,log,exit,pid}
  euler put <local> <remote>                malé soubory (kód, configy)
  euler get <remote> <local>
  euler close                               smaže kernel (úklid po práci)
"""

import argparse
import base64
import json
import os
import re
import sys
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
import websocket

URL = os.environ.get("EULER_URL", "").rstrip("/")
TOKEN = os.environ.get("EULER_TOKEN", "")
HDR = {"Authorization": f"token {TOKEN}"}
STATE = Path.home() / ".euler_kernel"
MARK = "__EULER__"
SECRET_KV = re.compile(r"((?:token|password|secret)[\w.-]*\s*[=:]\s*)[^\s'\"]+", re.IGNORECASE)


def _scrub(text):
    text = SECRET_KV.sub(r"\1[REDACTED]", text)
    for secret in (TOKEN, os.environ.get("MLFLOW_TRACKING_PASSWORD", "")):
        for form in {secret, urllib.parse.quote(secret, safe=""), urllib.parse.quote_plus(secret)}:
            for n in range(len(form), 7, -1):
                text = text.replace(form[:n], "[REDACTED]")
    return text


def die(msg, rc=2):
    print(_scrub(f"euler: {msg}"), file=sys.stderr)
    sys.exit(rc)


def api(method, path, **kw):
    try:
        r = requests.request(method, f"{URL}/api/{path}", headers=HDR, timeout=60, **kw)
    except requests.RequestException as e:
        die(f"server nedostupný ({e.__class__.__name__}) — běží tvůj Jupyter server na Hubu?")
    if r.status_code >= 400:
        body = "<html body suppressed>" if "html" in r.headers.get("content-type", "") else r.text[:300]
        die(f"{method} {path} -> {r.status_code}: {body}")
    return r.json() if r.content else None


def kernel_id():
    """Reuse one kernel between calls; create a new one if it was culled."""
    if STATE.exists():
        kid = STATE.read_text().strip()
        try:
            r = requests.get(f"{URL}/api/kernels/{kid}", headers=HDR, timeout=30)
        except requests.RequestException:
            r = None
        if r is not None and r.ok:
            return kid
    kid = api("POST", "kernels", json={"name": "python3"})["id"]
    STATE.write_text(kid)
    return kid


def run_py(code, timeout):
    """Execute Python in the remote kernel, return printed stdout."""
    kid = kernel_id()
    ws_url = URL.replace("http", "ws", 1) + f"/api/kernels/{kid}/channels"
    ws = websocket.create_connection(ws_url, header=[f"Authorization: token {TOKEN}"], timeout=timeout)
    msg_id = uuid.uuid4().hex
    ws.send(
        json.dumps(
            {
                "header": {
                    "msg_id": msg_id,
                    "username": "euler",
                    "session": uuid.uuid4().hex,
                    "msg_type": "execute_request",
                    "version": "5.3",
                    "date": datetime.now(timezone.utc).isoformat(),
                },
                "parent_header": {},
                "metadata": {},
                "buffers": [],
                "channel": "shell",
                "content": {
                    "code": code,
                    "silent": False,
                    "store_history": False,
                    "user_expressions": {},
                    "allow_stdin": False,
                    "stop_on_error": True,
                },
            }
        )
    )
    out, err = [], None
    try:
        while True:
            m = json.loads(ws.recv())
            if m.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            t = m["msg_type"]
            if t == "stream":
                out.append(m["content"]["text"])
            elif t == "error":
                err = "\n".join(m["content"]["traceback"])
            elif t == "status" and m["content"]["execution_state"] == "idle":
                break
    except websocket.WebSocketTimeoutException:
        die(f"timeout po {timeout}s (příkaz na serveru mohl dál běžet; pro dlouhé věci použij bg)")
    finally:
        ws.close()
    if err:
        die(f"chyba v kernelu:\n{err}")
    return "".join(out)


def b64(s):
    return base64.b64encode(s.encode()).decode()


def cmd_exec(cmd, timeout):
    code = f"""
import subprocess, json, base64, os
_c = base64.b64decode("{b64(cmd)}").decode()
try:
    _r = subprocess.run(_c, shell=True, executable="/bin/bash", capture_output=True,
                        timeout={timeout}, cwd=os.path.expanduser("~"))
    _res = dict(rc=_r.returncode, o=base64.b64encode(_r.stdout).decode(), e=base64.b64encode(_r.stderr).decode())
except subprocess.TimeoutExpired as _x:
    _res = dict(rc=124, o=base64.b64encode(_x.stdout or b"").decode(), e=base64.b64encode(b"timeout\\n").decode())
print("{MARK}" + json.dumps(_res))
"""
    raw = run_py(code, timeout + 30)
    line = next((l for l in raw.splitlines() if l.startswith(MARK)), None)
    if line is None:
        die(f"nečekaný výstup:\n{raw}")
    res = json.loads(line[len(MARK) :])
    sys.stdout.write(_scrub(base64.b64decode(res["o"]).decode(errors="replace")))
    sys.stderr.write(_scrub(base64.b64decode(res["e"]).decode(errors="replace")))
    sys.exit(res["rc"])


def cmd_bg(name, cmd, min_free_mb):
    guard = f"""
if {min_free_mb} > 0:
    try:
        _free = max(int(x) for x in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).split())
    except Exception as _x:
        _free = -1
    if _free < {min_free_mb}:
        _abort = f"GPU guard: volných {{_free}} MB < {min_free_mb} MB, nespouštím"
"""
    code = f"""
import subprocess, base64, os, json, pathlib
_abort = None
{guard}
_d = pathlib.Path.home() / "jobs" / "{name}"
if _abort is None and (_d / "pid").exists() and not (_d / "exit").exists():
    _abort = "job '{name}' už běží (nebo skončil bez exit souboru)"
if _abort is None:
    _d.mkdir(parents=True, exist_ok=True)
    for _f in ("exit", "log"):
        (_d / _f).unlink(missing_ok=True)
    (_d / "run.sh").write_text(base64.b64decode("{b64(cmd)}").decode() + "\\n")
    _w = f'cd ~ && bash {{_d}}/run.sh; echo $? > {{_d}}/exit'
    _p = subprocess.Popen(["bash", "-c", _w], stdout=open(_d / "log", "wb"), stderr=subprocess.STDOUT,
                          stdin=subprocess.DEVNULL, start_new_session=True)
    (_d / "pid").write_text(str(_p.pid))
print("{MARK}" + json.dumps(dict(abort=_abort, dir=str(_d))))
"""
    raw = run_py(code, 60)
    res = json.loads(next(l for l in raw.splitlines() if l.startswith(MARK))[len(MARK) :])
    if res["abort"]:
        die(res["abort"], 1)
    print(f"spuštěno: {res['dir']}  (log: tail {res['dir']}/log, konec: {res['dir']}/exit)")


def cmd_put(local, remote):
    data = base64.b64encode(Path(local).read_bytes()).decode()
    api(
        "PUT",
        f"contents/{remote}",
        json={"type": "file", "format": "base64", "content": data},
    )
    print(f"nahráno: {remote}")


def cmd_get(remote, local):
    r = api(
        "GET",
        f"contents/{remote}",
        params={"content": 1, "format": "base64", "type": "file"},
    )
    Path(local).write_bytes(base64.b64decode(r["content"]))
    print(f"staženo: {local}")


def cmd_close():
    if not STATE.exists():
        print("žádný kernel")
        return
    kid = STATE.read_text().strip()
    r = requests.delete(f"{URL}/api/kernels/{kid}", headers=HDR, timeout=30)
    STATE.unlink()
    print(f"kernel smazán ({r.status_code})")


def main():
    if not URL or not TOKEN:
        die("nastav EULER_URL a EULER_TOKEN")
    p = argparse.ArgumentParser(prog="euler")
    sp = p.add_subparsers(dest="c", required=True)
    e = sp.add_parser("exec")
    e.add_argument("cmd")
    e.add_argument("--timeout", type=int, default=600)
    b = sp.add_parser("bg")
    b.add_argument("name")
    b.add_argument("cmd")
    b.add_argument("--min-free-mb", type=int, default=0)
    u = sp.add_parser("put")
    u.add_argument("local")
    u.add_argument("remote")
    g = sp.add_parser("get")
    g.add_argument("remote")
    g.add_argument("local")
    sp.add_parser("close")
    a = p.parse_args()
    if a.c == "exec":
        cmd_exec(a.cmd, a.timeout)
    elif a.c == "bg":
        cmd_bg(a.name, a.cmd, a.min_free_mb)
    elif a.c == "put":
        cmd_put(a.local, a.remote)
    elif a.c == "get":
        cmd_get(a.remote, a.local)
    elif a.c == "close":
        cmd_close()


if __name__ == "__main__":
    main()
