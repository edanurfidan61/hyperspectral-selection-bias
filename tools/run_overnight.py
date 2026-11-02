"""Gözetimsiz gece zinciri — revision-1 E3/E4/E5/E6 tam koşuları (sırayla, tek ağır koşu).

Adımlar (sıra sabit):

  1 e3_grouping   E3 yalnız plot gruplaması (R3-9; öncelikli)
  2 e6_G0_G1      E6 G0 (makale bölmesi) + G1 (tekrarlı bölme)
  3 e3_all        E3 tüm eksenler (grouping hücreleri checkpoint'ten atlanır)
  4 e6_G2_G3      E6 G2 (kardinalite-sabit) + G3 (kırpmasız)
  5 e5_rest       E5, ryckewaert/lasso HARİÇ (B = 200)
  6 e4            E4, önceden belirlenmiş kural: S-null lasso+MI 100 tekrar (DEVIATIONS.md D1)
  7 e5_ryck_lasso E5 yalnız ryckewaert/lasso, B = 50, free + within_plot (DEVIATIONS.md D2)

Adım durumları: PENDING → RUNNING → DONE | FAILED; ya da SKIPPED (CONTROL ile).
Her döngüde sıradaki İLK PENDING adım koşar. FAILED ve SKIPPED kalıcıdır (orkestratör
yeniden başlasa da tekrar koşmaz); yalnız CONTROL ``retry`` ile PENDING'e döner.

Adım çalıştırma
---------------
* DONE dosyası (``outputs_rev1/logs/<...>.DONE.json``) varsa koşulmaz (DONE, ``skipped``).
  Yoksa deney YENİ bir python sürecinde başlar ve kendi checkpoint'lerinden devam eder
  (kod düzeltmeleri bir sonraki denemede geçerlidir).
* Çıkış ≠ 0 (ya da çıkış 0 ama DONE dosyası yok) → bir kez yeniden deneme → FAILED.
* Her adımdan sonra (atlansa da) ``--check`` koşulur; sonuç durum dosyasına yazılır.

Kontrol dosyası — ``outputs_rev1/logs/overnight.CONTROL.json``
---------------------------------------------------------------
Adım sınırlarında, koşu sırasında ve boştayken ``control_s`` (60 sn) aralıkla okunur.
Yazan taraf dosyayı tek seferde yazmalı (son 3 sn içinde değişmiş dosya bir sonraki tura kalır).

  {"retry": ["e4", ...]}          FAILED/DONE/SKIPPED/PENDING adım → PENDING; DONE dosyası
                                  önce ``<ad>.bak`` (varsa ``.bak-<zaman>``) olarak taşınır;
                                  deneme sayacı sıfırlanır. Koşan adıma uygulanmaz.
  {"skip": ["e5_ryck_lasso"]}     PENDING/FAILED adım → SKIPPED (DONE ve koşan adıma uygulanmaz)
  {"stop_after_current": true}    koşan adım bitince YENİ adım başlatılmaz: zincir DURAKLAR
                                  (CPU kullanmaz, CONTROL'ü dinlemeye devam eder)
  {"resume": true}                duraklatmayı kaldır
  {"restart_orchestrator": true}  adım sınırında çık (kod 5) → denetleyici (run_overnight.ps1)
                                  orkestratörü YENİ süreçte başlatır (bu dosyadaki düzeltmeler için)

Uygulanan dosya ``overnight.CONTROL.applied-<zaman>.json``, geçersiz olan (bozuk JSON,
bilinmeyen anahtar, yanlış tip) ``overnight.CONTROL.rejected-<zaman>.json`` olarak yeniden
adlandırılır; sonuç ``overnight_status.json`` → ``control_history`` / ``last_control``.

Bekleme modu
------------
Koşulacak PENDING adım kalmadığında: hepsi DONE → çıkış 0 (``ALL_DONE: true``). FAILED /
SKIPPED varsa ya da zincir duraklatılmışsa çıkılmaz: ``--wait-until``'e (varsayılan
2026-11-02 03:00) kadar CONTROL dinlenir; süre dolunca çıkış 4.

Diğer
-----
* Kilit ``overnight.lock`` (PID). İkinci kopya hemen çıkar (kod 2); sahibi ölmüş/python
  değilse kilit temizlenir.
* ``overnight_status.json`` adım başı/sonunda ve ``heartbeat_s`` (60 sn) aralıkla yazılır:
  ``heartbeat`` = zaman, durum, koşan adım, deney logundaki son ``[k/N]`` satırı.
* Ana log ``overnight.log``; adım çıktıları ``overnight_steps/``.
* Yerel durdurma dosyaları (KALICI — silinene kadar zincir başlamaz):
  ``overnight.STOP`` (hemen; koşan adımın süreç ağacı sonlandırılır, biten görevlerin
  checkpoint'leri korunur) ve ``overnight.STOP_AFTER`` (koşan adım bitince). Çıkış 3.
* Çıkış kodları: 0 hepsi DONE | 2 kilit başkasında | 3 STOP | 4 bekleme süresi doldu |
  5 yeniden başlatma istendi | diğer = anormal (denetleyici 60 sn sonra yeniden başlatır).

Çalıştırma (proje kökünden; normalde ``tools\\run_overnight.ps1`` denetleyicisi üzerinden)::

    .venv\\Scripts\\python.exe tools\\run_overnight.py --dry-run
    .venv\\Scripts\\python.exe tools\\run_overnight.py --simulate --logs-dir <boş dizin> --simulate-fail e6_G2_G3
    .venv\\Scripts\\python.exe tools\\run_overnight.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs_rev1" / "logs"
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
MAX_ATTEMPTS = 2                      # ilk deneme + bir yeniden deneme
WAIT_UNTIL_DEFAULT = "2026-11-02T03:00:00"
CHECK_TIMEOUT_S = 1800
CONTROL_SETTLE_S = 3.0
ERR_RE = re.compile(r"(Traceback|Error|Exception|HATA|FAILED|Killed)", re.IGNORECASE)
PROGRESS_RE = re.compile(r"\[\d+/\d+\]")
CONTROL_KEYS = {"retry", "skip", "stop_after_current", "resume", "restart_orchestrator"}

EXIT_ALL_DONE, EXIT_LOCKED, EXIT_STOP, EXIT_DEADLINE, EXIT_RESTART = 0, 2, 3, 4, 5


# ---------------------------------------------------------------------------
# Adımlar
# ---------------------------------------------------------------------------
def build_steps(py: str) -> list[dict]:
    m = lambda mod, *a: [py, "-m", f"src.m07_revision.{mod}", *a]  # noqa: E731
    return [
        {"name": "e3_grouping", "cmd": m("e3_sensitivity", "--axes", "grouping"),
         "check": m("e3_sensitivity", "--check", "--axes", "grouping"),
         "done": "E3_sensitivity.axes-grouping.DONE.json", "exp_log": "E3_sensitivity.log"},
        {"name": "e6_G0_G1", "cmd": m("e6_ga", "--variants", "G0,G1"),
         "check": m("e6_ga", "--check", "--variants", "G0,G1"),
         "done": "E6_ga.G0-G1.DONE.json", "exp_log": "E6_ga.log"},
        {"name": "e3_all", "cmd": m("e3_sensitivity"),
         "check": m("e3_sensitivity", "--check"),
         "done": "E3_sensitivity.DONE.json", "exp_log": "E3_sensitivity.log"},
        {"name": "e6_G2_G3", "cmd": m("e6_ga", "--variants", "G2,G3"),
         "check": m("e6_ga", "--check", "--variants", "G2,G3"),
         "done": "E6_ga.G2-G3.DONE.json", "exp_log": "E6_ga.log"},
        {"name": "e5_rest", "cmd": m("e5_permutation", "--exclude", "ryckewaert/lasso"),
         "check": m("e5_permutation", "--check", "--exclude", "ryckewaert/lasso"),
         "done": "E5_permutation.excl-ryckewaert_lasso.DONE.json", "exp_log": "E5_permutation.log"},
        {"name": "e4", "cmd": m("e4_simulation", "--reduced-repeats", "lasso,mutual_info"),
         "check": m("e4_simulation", "--check"),
         "done": "E4_simulation.DONE.json", "exp_log": "E4_simulation.log"},
        {"name": "e5_ryck_lasso",
         "cmd": m("e5_permutation", "--only", "ryckewaert/lasso", "--n-perm", "50"),
         "check": m("e5_permutation", "--check", "--only", "ryckewaert/lasso", "--n-perm", "50"),
         "done": "E5_permutation.only-ryckewaert_lasso.B50.DONE.json", "exp_log": "E5_permutation.log"},
    ]


#: --simulate: gerçek deney yerine sahte adım (deney loguna [k/N] yazar; DONE'u sandbox'a yazar)
_FAKE_STEP = r"""
import sys, time, pathlib
logs, exp_log, done, fail, name, n = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == "1", sys.argv[5], int(sys.argv[6])
fixed = (logs / f"{name}.FIXED").exists()
with open(logs / exp_log, "a", encoding="utf-8") as fh:
    for i in range(1, n + 1):
        fh.write(f"SIM | [{i}/{n}] bitti {name} gorev {i}\n"); fh.flush()
        print(f"[{i}/{n}] {name}", flush=True); time.sleep(0.3)
if fail and not fixed:
    print(f"RuntimeError: simulated failure in {name}", flush=True); sys.exit(1)
(logs / done).write_text("{}", encoding="utf-8"); print(f"ALL DONE {name}")
"""


def simulated_steps(py: str, logs: Path, fail: set[str], n_tasks: int = 3) -> list[dict]:
    out = []
    for s in build_steps(py):
        out.append({**s, "cmd": [py, "-c", _FAKE_STEP, str(logs), s["exp_log"], s["done"],
                                 "1" if s["name"] in fail else "0", s["name"], str(n_tasks)],
                    "check": [py, "-c", f"print('SIM check {s['name']}: TAMAM')"]})
    return out


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


class Log:
    def __init__(self, path: Path, echo: bool = True):
        self.path, self.echo = path, echo
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self.echo:
            try:
                print(line, flush=True)
            except Exception:
                pass


def write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for _ in range(40):                      # Windows: okuyucu dosyayı açık tutuyorsa kısa bekle
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.25)
    tmp.replace(path)


def rel(path: Path) -> str:
    """Kök altındaysa göreli yol, değilse mutlak (test dizinleri)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def pid_alive_python(pid: int) -> bool:
    """PID yaşıyor VE bir python süreci mi? (PID yeniden kullanımına karşı ad kontrolü)."""
    if not pid or pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    h = k32.OpenProcess(0x1000, False, int(pid))           # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ctypes.get_last_error() == 5                  # erişim reddi → süreç var
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value != 259:   # STILL_ACTIVE
            return False
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return "python" in Path(buf.value).name.lower()
        return True
    finally:
        k32.CloseHandle(h)


def acquire_lock(lock: Path, log: Log) -> bool:
    """O_EXCL ile kilit. Sahibi ölü/python değilse kilit temizlenip bir kez daha denenir."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"pid": os.getpid(), "started": now(), "argv": sys.argv}, fh)
            return True
        except FileExistsError:
            try:
                info = json.loads(lock.read_text(encoding="utf-8"))
                pid = int(info.get("pid", 0))
            except Exception:
                info, pid = {}, 0
                if time.time() - lock.stat().st_mtime < 30:      # yazımı süren bir kopya olabilir
                    return False
            if pid and pid != os.getpid() and pid_alive_python(pid):
                log(f"KİLİT: başka bir kopya çalışıyor (PID {pid}, başlangıç {info.get('started')}) — çıkılıyor")
                return False
            log(f"KİLİT: bayat kilit temizlendi (PID {pid} yaşamıyor)")
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
    return False


def release_lock(lock: Path) -> None:
    try:
        if int(json.loads(lock.read_text(encoding="utf-8")).get("pid", 0)) == os.getpid():
            lock.unlink()
    except Exception:
        pass


def kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(pid, 9)
        except Exception:
            pass


def _read_tail_text(path: Path, max_bytes: int = 65536) -> list[str]:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read()
    except FileNotFoundError:
        return []
    return [x.rstrip() for x in data.decode("utf-8", errors="replace").splitlines()]


def last_error_line(path: Path) -> str | None:
    lines = [x for x in _read_tail_text(path) if x.strip()]
    errs = [x for x in lines if ERR_RE.search(x)]
    return errs[-1] if errs else (lines[-1] if lines else None)


def last_progress_line(path: Path) -> str | None:
    lines = [x for x in _read_tail_text(path) if PROGRESS_RE.search(x)]
    return lines[-1] if lines else None


def tail(path: Path, n: int = 15) -> list[str]:
    return _read_tail_text(path)[-n:]


def parse_control(raw: str) -> dict:
    """CONTROL içeriğini doğrula. Geçersizse ValueError."""
    obj = json.loads(raw)
    if not isinstance(obj, dict) or not obj:
        raise ValueError("CONTROL boş olmayan bir JSON nesnesi olmalı")
    unknown = set(obj) - CONTROL_KEYS
    if unknown:
        raise ValueError(f"bilinmeyen anahtar(lar): {sorted(unknown)}; geçerli: {sorted(CONTROL_KEYS)}")
    for k in ("retry", "skip"):
        if k in obj and not (isinstance(obj[k], list) and all(isinstance(x, str) for x in obj[k])):
            raise ValueError(f"'{k}' adım adlarından oluşan bir liste olmalı")
    for k in ("stop_after_current", "resume", "restart_orchestrator"):
        if k in obj and not isinstance(obj[k], bool):
            raise ValueError(f"'{k}' true/false olmalı")
    return obj


# ---------------------------------------------------------------------------
# Orkestratör
# ---------------------------------------------------------------------------
class Chain:
    def __init__(self, steps: list[dict], logs: Path, *, poll: float = 10.0, heartbeat_s: float = 60.0,
                 control_s: float = 60.0, wait_until: datetime | None = None,
                 max_attempts: int = MAX_ATTEMPTS, echo: bool = True):
        self.steps = steps
        self.by_name = {s["name"]: s for s in steps}
        self.logs, self.poll, self.max_attempts = logs, poll, max_attempts
        self.heartbeat_s, self.control_s = heartbeat_s, control_s
        self.wait_until = wait_until or datetime.fromisoformat(WAIT_UNTIL_DEFAULT)
        self.status_path = logs / "overnight_status.json"
        self.lock = logs / "overnight.lock"
        self.stop_now = logs / "overnight.STOP"
        self.stop_after = logs / "overnight.STOP_AFTER"
        self.control = logs / "overnight.CONTROL.json"
        self.step_dir = logs / "overnight_steps"
        self.log = Log(logs / "overnight.log", echo=echo)
        self.restart_requested = False
        self._last_hb = 0.0
        self._last_ctl = 0.0
        self.status = self._load_status()

    # -- durum -----------------------------------------------------------------
    def _load_status(self) -> dict:
        old = {}
        if self.status_path.exists():
            try:
                old = json.loads(self.status_path.read_text(encoding="utf-8"))
            except Exception:
                old = {}
        prev = {s["name"]: s for s in old.get("steps", [])}
        steps = []
        for s in self.steps:
            p = prev.get(s["name"], {})
            status = p.get("status", "PENDING")
            steps.append({"name": s["name"], "command": "(simulated step)" if s["cmd"][1] == "-c"
                          else " ".join(s["cmd"][1:]),
                          "check_command": " ".join(s["check"][1:]), "done_file": s["done"],
                          "experiment_log": s.get("exp_log"),
                          "status": "PENDING" if status == "RUNNING" else status,
                          "skipped": p.get("skipped", False), "attempts": p.get("attempts", 0),
                          "exit_codes": p.get("exit_codes", []), "started": p.get("started"),
                          "finished": p.get("finished"), "check": p.get("check"),
                          "last_error_line": p.get("last_error_line"), "step_log": p.get("step_log")})
        return {"chain": "revision-1 overnight", "root": str(ROOT), "pid": os.getpid(),
                "orchestrator_started": now(),
                "orchestrator_starts": int(old.get("orchestrator_starts", 0)) + 1,
                "supervisor": {"restarts": os.environ.get("REV1_SUPERVISOR_RESTARTS"),
                               "last_exit_code": os.environ.get("REV1_SUPERVISOR_LAST_EXIT")},
                "wait_until": self.wait_until.isoformat(timespec="seconds"),
                "state": "starting", "paused": bool(old.get("paused", False)),
                "last_update": now(), "current_step": None, "heartbeat": None,
                "CHAIN_FINISHED": False, "ALL_DONE": False, "ALL_CHECKS_OK": False,
                "control_file": str(self.control),
                "control_history": old.get("control_history", [])[-50:],
                "last_control": old.get("last_control"),
                "stop_files": {"now": str(self.stop_now), "after_step": str(self.stop_after)},
                "steps": steps}

    def save(self) -> None:
        sts = self.status["steps"]
        self.status["n_done"] = sum(x["status"] == "DONE" for x in sts)
        self.status["n_failed"] = sum(x["status"] == "FAILED" for x in sts)
        self.status["n_skipped"] = sum(x["status"] == "SKIPPED" for x in sts)
        self.status["n_pending"] = sum(x["status"] in ("PENDING", "RUNNING") for x in sts)
        self.status["CHAIN_FINISHED"] = self.status["n_pending"] == 0
        self.status["ALL_DONE"] = self.status["n_done"] == len(sts)
        self.status["ALL_CHECKS_OK"] = all((x.get("check") or {}).get("ok") for x in sts)
        self.status["last_update"] = now()
        write_json_atomic(self.status_path, self.status)

    def _st(self, name: str) -> dict:
        return next(s for s in self.status["steps"] if s["name"] == name)

    def reconcile(self) -> None:
        """DONE dosyası ↔ durum uyumu (dosya esas)."""
        for st in self.status["steps"]:
            has = (self.logs / st["done_file"]).exists()
            if has and st["status"] != "DONE":
                if st["status"] == "SKIPPED":
                    continue                        # kullanıcı atlattı; dokunma
                st.update(status="DONE", skipped=True, finished=st.get("finished") or now())
            elif not has and st["status"] == "DONE":
                self.log(f"[{st['name']}] durum DONE ama DONE dosyası yok → PENDING")
                st.update(status="PENDING", skipped=False)

    # -- heartbeat ---------------------------------------------------------------
    def heartbeat(self, force: bool = False, step: dict | None = None, t0: float | None = None) -> None:
        if not force and time.time() - self._last_hb < self.heartbeat_s:
            return
        self._last_hb = time.time()
        hb = {"at": now(), "pid": os.getpid(), "state": self.status["state"],
              "paused": self.status["paused"], "current_step": step["name"] if step else None}
        if step:
            exp = self.logs / step["exp_log"] if step.get("exp_log") else None
            hb.update({"step_elapsed_s": None if t0 is None else round(time.time() - t0),
                       "experiment_log": None if exp is None else rel(exp),
                       "last_progress_line": None if exp is None else last_progress_line(exp)})
        self.status["heartbeat"] = hb
        self.save()

    # -- kontrol dosyası ----------------------------------------------------------
    def poll_control(self, running: str | None = None, force: bool = False) -> None:
        if not force and time.time() - self._last_ctl < self.control_s:
            return
        self._last_ctl = time.time()
        self.apply_control(running)

    def apply_control(self, running: str | None = None) -> dict | None:
        p = self.control
        if not p.exists():
            return None
        try:
            # NTFS mtime, time.time()'dan ~1e-7 sn ileride olabiliyor → yaş negatif çıkmasın
            if max(0.0, time.time() - p.stat().st_mtime) < CONTROL_SETTLE_S:
                return None                          # yazımı sürüyor olabilir → bir sonraki tur
            raw = p.read_text(encoding="utf-8-sig")
        except (FileNotFoundError, PermissionError):
            return None
        ts = stamp()
        try:
            obj = parse_control(raw)
        except Exception as e:                       # bozuk JSON / geçersiz içerik → reddet
            dest = p.with_name(f"overnight.CONTROL.rejected-{ts}.json")
            p.replace(dest)
            rec = {"at": now(), "file": dest.name, "result": "rejected", "error": f"{type(e).__name__}: {e}"}
            self.status["control_history"].append(rec)
            self.status["last_control"] = rec
            self.log(f"CONTROL REDDEDİLDİ → {dest.name}: {rec['error']}")
            self.save()
            return rec
        actions = []
        for name in obj.get("retry", []):
            actions.append({"action": "retry", "step": name, "result": self._retry(name, running)})
        for name in obj.get("skip", []):
            actions.append({"action": "skip", "step": name, "result": self._skip(name, running)})
        if obj.get("stop_after_current"):
            self.status["paused"] = True
            actions.append({"action": "stop_after_current",
                            "result": "duraklatıldı: yeni adım başlatılmayacak (resume ile devam)"})
        if obj.get("resume"):
            self.status["paused"] = False
            actions.append({"action": "resume", "result": "duraklatma kaldırıldı"})
        if obj.get("restart_orchestrator"):
            self.restart_requested = True
            actions.append({"action": "restart_orchestrator",
                            "result": "adım sınırında çıkılacak (kod 5); denetleyici yeniden başlatır"})
        dest = p.with_name(f"overnight.CONTROL.applied-{ts}.json")
        p.replace(dest)
        rec = {"at": now(), "file": dest.name, "result": "applied", "request": obj, "actions": actions}
        self.status["control_history"].append(rec)
        self.status["control_history"] = self.status["control_history"][-50:]
        self.status["last_control"] = rec
        for a in actions:
            self.log(f"CONTROL {a['action']} {a.get('step', '')}: {a['result']}")
        self.save()
        return rec

    def _retry(self, name: str, running: str | None) -> str:
        if name not in self.by_name:
            return "yok sayıldı: bilinmeyen adım"
        if name == running:
            return "yok sayıldı: adım şu an koşuyor"
        st = self._st(name)
        old = st["status"]
        done = self.logs / st["done_file"]
        moved = None
        if done.exists():
            bak = done.with_name(done.name + ".bak")
            if bak.exists():
                bak = done.with_name(done.name + f".bak-{stamp()}")
            done.replace(bak)
            moved = bak.name
        st.update(status="PENDING", skipped=False, attempts=0, exit_codes=[], started=None,
                  finished=None, check=None, last_error_line=None)
        return f"{old} → PENDING" + (f" (DONE dosyası → {moved})" if moved else "")

    def _skip(self, name: str, running: str | None) -> str:
        if name not in self.by_name:
            return "yok sayıldı: bilinmeyen adım"
        if name == running:
            return "yok sayıldı: adım şu an koşuyor"
        st = self._st(name)
        if st["status"] == "DONE":
            return "yok sayıldı: adım zaten DONE"
        old = st["status"]
        st.update(status="SKIPPED", finished=now())
        return f"{old} → SKIPPED"

    # -- adım çalıştırma ----------------------------------------------------------
    def _run_proc(self, cmd: list[str], out: Path, *, step: dict | None = None,
                  timeout: float | None = None) -> tuple[int, bool]:
        """Alt süreç (her çağrı YENİ python süreci); çıktı dosyaya. Dönüş (kod, STOP ile kesildi mi)."""
        out.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"\n===== {now()} | {' '.join(cmd[:4])}{' …' if len(cmd) > 4 else ''}\n")
            fh.flush()
            # Aynı konsol grubu: Ctrl+C alt sürece de ulaşır (sahipsiz deney süreci kalmaz)
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
            t0 = time.time()
            while True:
                try:
                    return proc.wait(timeout=self.poll), False
                except subprocess.TimeoutExpired:
                    pass
                except KeyboardInterrupt:
                    kill_tree(proc.pid)
                    raise
                if self.stop_now.exists():
                    self.log(f"STOP dosyası görüldü → süreç ağacı sonlandırılıyor (PID {proc.pid})")
                    kill_tree(proc.pid)
                    proc.wait()
                    return proc.returncode, True
                if timeout and time.time() - t0 > timeout:
                    kill_tree(proc.pid)
                    proc.wait()
                    return 124, False
                if step is not None:
                    self.heartbeat(step=step, t0=t0)
                    self.poll_control(running=step["name"])

    def _check(self, s: dict) -> None:
        st = self._st(s["name"])
        out = self.step_dir / f"{s['name']}.check.log"
        t0 = time.time()
        code, _ = self._run_proc(s["check"], out, timeout=CHECK_TIMEOUT_S)
        st["check"] = {"exit_code": code, "ok": code == 0, "at": now(),
                       "seconds": round(time.time() - t0, 1), "log": rel(out), "tail": tail(out, 15)}
        self.log(f"[{s['name']}] --check → çıkış {code} ({'TAMAM' if code == 0 else 'SORUN'})")
        self.save()

    def run_step(self, s: dict) -> str:
        st = self._st(s["name"])
        done = self.logs / s["done"]
        if done.exists():
            st.update(status="DONE", skipped=True, finished=st.get("finished") or now())
            self.log(f"[{s['name']}] DONE dosyası var ({s['done']}) → atlandı")
            self.save()
            if not (st.get("check") or {}).get("ok"):
                self._check(s)
            return "DONE"
        st.update(status="RUNNING", skipped=False, started=now(), finished=None, attempts=0,
                  exit_codes=[], last_error_line=None)
        self.status.update(state="running", current_step=s["name"])
        self.save()
        out = self.step_dir / f"{s['name']}.log"
        st["step_log"] = rel(out)
        for attempt in range(1, self.max_attempts + 1):
            st["attempts"] = attempt
            self.save()
            self.log(f"[{s['name']}] başladı (deneme {attempt}/{self.max_attempts}, yeni süreç)")
            t0 = time.time()
            self.heartbeat(force=True, step=s, t0=t0)
            code, stopped = self._run_proc(s["cmd"], out, step=s)
            st["exit_codes"].append(code)
            el = time.time() - t0
            if stopped:
                st.update(status="PENDING", finished=None, last_error_line="kullanıcı STOP ile durdurdu")
                self.save()
                return "STOPPED"
            if code == 0 and done.exists():
                st.update(status="DONE", finished=now(), last_error_line=None)
                self.log(f"[{s['name']}] bitti (çıkış 0, {el / 3600:.2f} sa)")
                self.save()
                self._check(s)
                return "DONE"
            st["last_error_line"] = last_error_line(out) if code != 0 else \
                f"çıkış 0 ama DONE dosyası yok: {s['done']}"
            self.log(f"[{s['name']}] BAŞARISIZ (çıkış {code}, {el / 60:.1f} dk): {st['last_error_line']}")
            self.save()
        st.update(status="FAILED", finished=now())
        self.log(f"[{s['name']}] FAILED — {self.max_attempts} deneme; sonraki adıma geçiliyor")
        self.save()
        self._check(s)
        return "FAILED"

    def _next_pending(self) -> dict | None:
        for st in self.status["steps"]:
            if st["status"] == "PENDING":
                return self.by_name[st["name"]]
        return None

    def _stop_file(self) -> bool:
        return self.stop_now.exists() or self.stop_after.exists()

    def run(self) -> int:
        if self._stop_file():
            f = self.stop_now if self.stop_now.exists() else self.stop_after
            self.log(f"STOP dosyası mevcut ({f.name}) — zincir başlatılmadı. Devam için dosyayı silin.")
            return EXIT_STOP
        if not acquire_lock(self.lock, self.log):
            return EXIT_LOCKED
        try:
            self.log("=" * 70)
            self.log(f"ORKESTRATÖR başladı | PID {os.getpid()} | başlatma #{self.status['orchestrator_starts']} | "
                     f"denetleyici yeniden başlatma: {self.status['supervisor']['restarts']} | "
                     f"bekleme sınırı {self.status['wait_until']}")
            self.reconcile()
            self.save()
            waiting_logged = False
            while True:
                self.apply_control(running=None)                        # adım sınırı
                if self.restart_requested:
                    self.status.update(state="restarting", current_step=None)
                    self.save()
                    self.log("CONTROL restart_orchestrator → çıkış 5 (denetleyici yeni süreçte başlatır)")
                    return EXIT_RESTART
                if self._stop_file():
                    self.status.update(state="stopped", current_step=None)
                    self.save()
                    self.log("STOP dosyası → zincir adım sınırında durduruldu")
                    return EXIT_STOP
                nxt = None if self.status["paused"] else self._next_pending()
                if nxt is not None:
                    waiting_logged = False
                    if self.run_step(nxt) == "STOPPED":
                        self.status.update(state="stopped", current_step=None)
                        self.save()
                        self.log("Zincir STOP ile durduruldu (checkpoint'ler korunur; aynı komutla devam)")
                        return EXIT_STOP
                    self.status["current_step"] = None
                    self.save()
                    continue
                # Koşulacak adım yok
                if self.status["n_done"] == len(self.status["steps"]) or all(
                        x["status"] == "DONE" for x in self.status["steps"]):
                    self.status.update(state="finished", current_step=None)
                    self.save()
                    self.log(f"ZİNCİR bitti | ALL_DONE=True | ALL_CHECKS_OK={self.status['ALL_CHECKS_OK']}")
                    self.heartbeat(force=True)
                    return EXIT_ALL_DONE
                if datetime.now() >= self.wait_until:
                    self.status.update(state="deadline", current_step=None)
                    self.save()
                    self.log(f"Bekleme süresi doldu ({self.status['wait_until']}) | DONE={self.status['n_done']} "
                             f"FAILED={self.status['n_failed']} SKIPPED={self.status['n_skipped']} → çıkış 4")
                    self.heartbeat(force=True)
                    return EXIT_DEADLINE
                state = "paused" if self.status["paused"] else "waiting"
                if not waiting_logged:
                    self.status.update(state=state, current_step=None)
                    self.save()
                    self.log(f"{'DURAKLATILDI' if state == 'paused' else 'BEKLEME MODU'}: DONE={self.status['n_done']} "
                             f"FAILED={self.status['n_failed']} SKIPPED={self.status['n_skipped']} "
                             f"PENDING={self.status['n_pending']} | {self.control.name} dinleniyor "
                             f"({self.control_s:.0f} sn) — sınır {self.status['wait_until']}")
                    waiting_logged = True
                self.status["state"] = state
                self.heartbeat(force=True)
                t_end = time.time() + self.control_s                    # boşta: control_s aralık
                while time.time() < t_end and not self._stop_file():
                    time.sleep(min(1.0, self.control_s))
        except KeyboardInterrupt:
            cur = self.status.get("current_step")
            if cur:
                self._st(cur).update(status="PENDING", last_error_line="Ctrl+C ile durduruldu")
            self.status["state"] = "stopped"
            self.save()
            self.log("Ctrl+C → zincir durduruldu (checkpoint'ler korunur; aynı komutla devam)")
            return 130
        finally:
            release_lock(self.lock)


# ---------------------------------------------------------------------------
# Kuru koşu (dosya YAZMAZ)
# ---------------------------------------------------------------------------
def preview_control(control: Path, steps: list[dict], status: dict | None) -> list[str]:
    """CONTROL dosyası olsaydı ne olurdu? (yeniden adlandırma / durum değişikliği YOK)."""
    if not control.exists():
        return [f"  CONTROL: {control.name} yok"]
    names = {s["name"] for s in steps}
    st = {s["name"]: s for s in (status or {}).get("steps", [])}
    try:
        obj = parse_control(control.read_text(encoding="utf-8-sig"))
    except Exception as e:
        return [f"  CONTROL: GEÇERSİZ → overnight.CONTROL.rejected-<zaman>.json olarak adlandırılır; "
                f"zincir etkilenmez ({type(e).__name__}: {e})"]
    out = [f"  CONTROL: geçerli → uygulanır, sonra overnight.CONTROL.applied-<zaman>.json: {json.dumps(obj)}"]
    cur_of = {n: st.get(n, {}).get("status", "PENDING") for n in names}   # apply_control ile aynı sıra
    for n in obj.get("retry", []):
        cur = cur_of.get(n)
        out.append(f"    retry {n}: " + ("bilinmeyen adım → yok sayılır" if n not in names else
                                         f"{cur} → PENDING (DONE dosyası varsa .bak'a taşınır; deneme sayacı 0)"))
        if n in names:
            cur_of[n] = "PENDING"
    for n in obj.get("skip", []):
        cur = cur_of.get(n, "PENDING")
        out.append(f"    skip {n}: " + ("bilinmeyen adım → yok sayılır" if n not in names else
                                        "DONE → yok sayılır" if cur == "DONE" else f"{cur} → SKIPPED"))
    for k in ("stop_after_current", "resume", "restart_orchestrator"):
        if obj.get(k):
            out.append(f"    {k}: true")
    return out


def dry_run(steps: list[dict], logs: Path) -> int:
    print(f"KURU KOŞU (hiçbir dosya yazılmaz/taşınmaz) | logs = {logs}")
    lock = logs / "overnight.lock"
    if lock.exists():
        try:
            pid = int(json.loads(lock.read_text(encoding="utf-8")).get("pid", 0))
        except Exception:
            pid = 0
        print(f"  kilit: PID {pid} → {'CANLI (ikinci kopya çıkar)' if pid_alive_python(pid) else 'bayat (temizlenir)'}")
    else:
        print("  kilit: yok")
    for f in ("overnight.STOP", "overnight.STOP_AFTER"):
        if (logs / f).exists():
            print(f"  {f} MEVCUT → zincir başlamaz (silinmeli)")
    status = None
    sp = logs / "overnight_status.json"
    if sp.exists():
        try:
            status = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            status = None
    prev = {s["name"]: s.get("status") for s in (status or {}).get("steps", [])}
    for i, s in enumerate(steps, 1):
        done = (logs / s["done"]).exists()
        p = prev.get(s["name"], "PENDING")
        act = ("ATLA (DONE dosyası var)" if done and p != "SKIPPED" else
               f"BEKLE ({p}; yalnız CONTROL retry ile koşar)" if p in ("FAILED", "SKIPPED") else "KOŞ")
        print(f"  {i}. {s['name']:<14} durum={p:<8} → {act}")
        print(f"       komut : {' '.join(s['cmd'][1:])}")
        print(f"       DONE  : {s['done']} | check: {' '.join(s['check'][1:])}")
    for line in preview_control(logs / "overnight.CONTROL.json", steps, status):
        print(line)
    print(f"  hata → {MAX_ATTEMPTS - 1} yeniden deneme → FAILED; her adım sonrası --check; "
          f"FAILED/SKIPPED varsa {WAIT_UNTIL_DEFAULT}'e kadar CONTROL dinlenir")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="revision-1 gözetimsiz gece zinciri")
    ap.add_argument("--dry-run", action="store_true", help="gerçek koşu yok; sıra, atlama ve CONTROL önizlemesi")
    ap.add_argument("--simulate", action="store_true",
                    help="tam zincir mantığı, deneyler yerine sahte adımlar (yalnız --logs-dir sandbox'ında)")
    ap.add_argument("--simulate-fail", type=str, default="",
                    help="--simulate: <logs>/<ad>.FIXED yoksa başarısız olacak adımlar (virgülle)")
    ap.add_argument("--logs-dir", type=str, default=str(LOGS), help="DONE/durum/kilit/CONTROL dizini")
    ap.add_argument("--wait-until", type=str, default=WAIT_UNTIL_DEFAULT, help="bekleme modu sınırı (ISO)")
    ap.add_argument("--poll", type=float, default=10.0, help="alt süreç yoklama aralığı (sn)")
    ap.add_argument("--heartbeat", type=float, default=60.0, help="heartbeat aralığı (sn)")
    ap.add_argument("--control-interval", type=float, default=60.0, help="CONTROL okuma aralığı (sn)")
    args = ap.parse_args(argv)
    py = str(VENV_PY if VENV_PY.exists() else Path(sys.executable))
    logs = Path(args.logs_dir)
    steps = build_steps(py)
    if args.simulate:
        if logs.resolve() == LOGS.resolve():
            ap.error("--simulate gerçek logs dizininde çalışmaz; --logs-dir ile boş bir sandbox verin")
        logs.mkdir(parents=True, exist_ok=True)
        steps = simulated_steps(py, logs, {x.strip() for x in args.simulate_fail.split(",") if x.strip()})
    if args.dry_run:
        return dry_run(steps, logs)
    return Chain(steps, logs, poll=args.poll, heartbeat_s=args.heartbeat, control_s=args.control_interval,
                 wait_until=datetime.fromisoformat(args.wait_until)).run()


if __name__ == "__main__":
    sys.exit(main())
