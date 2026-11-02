"""tools/run_overnight.py — zincir mantığı (sahte adımlarla; gerçek deney koşmaz)."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_overnight as RO  # noqa: E402

PY = sys.executable
PAST = datetime.now() - timedelta(minutes=1)


@pytest.fixture(autouse=True)
def _fast_control(monkeypatch):
    monkeypatch.setattr(RO, "CONTROL_SETTLE_S", 0.0)


def _step(name, code, d, *, make_done=True, check_code=0):
    done = f"{name}.DONE.json"
    body = code + (f"\nopen(r'{d / done}','w').write('{{}}')" if make_done else "")
    return {"name": name, "cmd": [PY, "-c", body], "exp_log": f"{name}.explog",
            "check": [PY, "-c", f"import sys; print('check {name}'); sys.exit({check_code})"], "done": done}


def _fake(name, d, fail=False, n=3):
    """RO._FAKE_STEP: <d>/<name>.FIXED yoksa ve fail ise başarısız; deney loguna [k/N] yazar."""
    exp, done = f"{name}.explog", f"{name}.DONE.json"
    return {"name": name, "exp_log": exp, "done": done,
            "cmd": [PY, "-c", RO._FAKE_STEP, str(d), exp, done, "1" if fail else "0", name, str(n)],
            "check": [PY, "-c", "print('ok')"]}


def _chain(steps, d, **kw):
    kw.setdefault("wait_until", PAST)
    return RO.Chain(steps, d, poll=0.1, heartbeat_s=0.2, control_s=0.3, echo=False, **kw)


def _status(d):
    # Windows: orkestratör tmp → replace yaparken okuyucu kısa süre PermissionError alabilir
    for _ in range(40):
        try:
            return json.loads((d / "overnight_status.json").read_text(encoding="utf-8"))
        except PermissionError:
            time.sleep(0.05)
    return json.loads((d / "overnight_status.json").read_text(encoding="utf-8"))


def _by(d):
    return {s["name"]: s for s in _status(d)["steps"]}


def _write_control(d, obj_or_text):
    p = d / "overnight.CONTROL.json"
    p.write_text(obj_or_text if isinstance(obj_or_text, str) else json.dumps(obj_or_text), encoding="utf-8")
    return p


def _wait_for(pred, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.1)
    return False


# ---------------------------------------------------------------------------
def test_retry_failed_skip_and_wait_mode_deadline(tmp_path):
    counter = tmp_path / "cnt.txt"
    (tmp_path / "c_skip.DONE.json").write_text("{}")
    steps = [
        _step("a_ok", "print('ok')", tmp_path),
        _step("b_fail", "import sys; print('RuntimeError: boom'); sys.exit(3)", tmp_path, check_code=1),
        _step("c_skip", "raise SystemExit(7)", tmp_path),                       # DONE var → koşmaz
        _step("d_flaky", f"import sys, pathlib; p = pathlib.Path(r'{counter}')\n"
                         "n = int(p.read_text()) if p.exists() else 0; p.write_text(str(n + 1))\n"
                         "if n == 0: sys.exit(1)", tmp_path),                   # 2. denemede DONE
        _step("e_nodone", "print('exit 0 without DONE')", tmp_path, make_done=False),
    ]
    assert _chain(steps, tmp_path).run() == RO.EXIT_DEADLINE                    # FAILED var → bekle → süre doldu
    st, by = _status(tmp_path), _by(tmp_path)
    assert st["CHAIN_FINISHED"] and not st["ALL_DONE"] and st["n_failed"] == 2 and st["state"] == "deadline"
    assert by["a_ok"]["status"] == "DONE" and by["a_ok"]["attempts"] == 1 and by["a_ok"]["check"]["ok"]
    assert by["b_fail"]["status"] == "FAILED" and by["b_fail"]["exit_codes"] == [3, 3]
    assert "RuntimeError: boom" in by["b_fail"]["last_error_line"] and by["b_fail"]["check"]["exit_code"] == 1
    assert by["c_skip"]["status"] == "DONE" and by["c_skip"]["skipped"]
    assert by["d_flaky"]["status"] == "DONE" and by["d_flaky"]["exit_codes"] == [1, 0]
    assert by["e_nodone"]["status"] == "FAILED" and "DONE dosyası yok" in by["e_nodone"]["last_error_line"]
    assert not (tmp_path / "overnight.lock").exists()
    # İkinci başlatma: FAILED kalıcı (yeniden koşmaz), DONE'lar atlanır
    assert _chain(steps, tmp_path).run() == RO.EXIT_DEADLINE
    by2 = _by(tmp_path)
    assert by2["b_fail"]["exit_codes"] == [3, 3] and by2["e_nodone"]["status"] == "FAILED"
    assert _status(tmp_path)["orchestrator_starts"] == 2


def test_all_done_exits_zero(tmp_path):
    steps = [_fake("x", tmp_path), _fake("y", tmp_path)]
    assert _chain(steps, tmp_path, wait_until=datetime.now() + timedelta(hours=1)).run() == RO.EXIT_ALL_DONE
    st = _status(tmp_path)
    assert st["ALL_DONE"] is True and st["ALL_CHECKS_OK"] is True and st["state"] == "finished"


def test_control_retry_reruns_failed_step_in_new_process(tmp_path):
    """Sahte FAILED adım → bekleme modu → düzeltme + CONTROL retry → adım YENİDEN koşar → ALL_DONE."""
    steps = [_fake("a", tmp_path), _fake("b", tmp_path, fail=True), _fake("c", tmp_path)]
    ch = _chain(steps, tmp_path, wait_until=datetime.now() + timedelta(minutes=5))
    res = {}
    th = threading.Thread(target=lambda: res.setdefault("rc", ch.run()))
    th.start()
    assert _wait_for(lambda: (tmp_path / "overnight_status.json").exists()
                     and _status(tmp_path)["state"] == "waiting")
    assert _by(tmp_path)["b"]["status"] == "FAILED" and _by(tmp_path)["c"]["status"] == "DONE"
    (tmp_path / "b.FIXED").write_text("")                                     # "kod düzeltmesi"
    _write_control(tmp_path, {"retry": ["b"]})
    th.join(timeout=60)
    assert res.get("rc") == RO.EXIT_ALL_DONE
    by, st = _by(tmp_path), _status(tmp_path)
    assert by["b"]["status"] == "DONE" and by["b"]["attempts"] == 1 and by["b"]["exit_codes"] == [0]
    assert st["last_control"]["result"] == "applied"
    assert st["last_control"]["actions"][0]["result"].startswith("FAILED → PENDING")
    assert list(tmp_path.glob("overnight.CONTROL.applied-*.json")) and not (tmp_path / "overnight.CONTROL.json").exists()
    log = (tmp_path / "overnight_steps" / "b.log").read_text(encoding="utf-8")
    assert log.count("=====") == 3                                            # 2 başarısız + 1 yeni süreç


def test_control_invalid_json_rejected_chain_unaffected(tmp_path):
    steps = [_fake("a", tmp_path), _fake("b", tmp_path)]
    _write_control(tmp_path, '{"retry": ["a"')                                  # bozuk JSON
    assert _chain(steps, tmp_path).run() == RO.EXIT_ALL_DONE
    assert list(tmp_path.glob("overnight.CONTROL.rejected-*.json"))
    st = _status(tmp_path)
    assert st["ALL_DONE"] and st["last_control"]["result"] == "rejected"
    # bilinmeyen anahtar da reddedilir
    _write_control(tmp_path, {"retyr": ["a"]})
    assert _chain(steps, tmp_path).run() == RO.EXIT_ALL_DONE
    assert len(list(tmp_path.glob("overnight.CONTROL.rejected-*.json"))) >= 1
    assert "bilinmeyen anahtar" in _status(tmp_path)["last_control"]["error"]


def test_control_skip_and_retry_done_moves_done_file(tmp_path):
    steps = [_fake("a", tmp_path), _fake("b", tmp_path), _fake("c", tmp_path)]
    (tmp_path / "a.DONE.json").write_text("{}")
    _write_control(tmp_path, {"skip": ["b", "zzz"], "retry": ["a"]})
    assert _chain(steps, tmp_path).run() == RO.EXIT_DEADLINE                    # b SKIPPED → bekleme → süre doldu
    by, st = _by(tmp_path), _status(tmp_path)
    assert by["b"]["status"] == "SKIPPED" and by["c"]["status"] == "DONE"
    assert by["a"]["status"] == "DONE" and not by["a"]["skipped"]             # retry ile YENİDEN koştu
    assert (tmp_path / "a.DONE.json.bak").exists()
    acts = {(a["action"], a["step"]): a["result"] for a in st["last_control"]["actions"]}
    assert "bilinmeyen" in acts[("skip", "zzz")] and "DONE dosyası" in acts[("retry", "a")]
    assert not (tmp_path / "b.explog").exists()                              # SKIPPED adım hiç koşmadı


def test_stop_after_current_pauses_and_resume_continues(tmp_path):
    steps = [_fake("a", tmp_path, n=8), _fake("b", tmp_path)]
    ch = _chain(steps, tmp_path, wait_until=datetime.now() + timedelta(minutes=5))
    res = {}
    th = threading.Thread(target=lambda: res.setdefault("rc", ch.run()))
    th.start()
    assert _wait_for(lambda: (tmp_path / "a.explog").exists())
    _write_control(tmp_path, {"stop_after_current": True})                     # a koşarken
    assert _wait_for(lambda: _status(tmp_path)["state"] == "paused")
    assert _by(tmp_path)["a"]["status"] == "DONE" and _by(tmp_path)["b"]["status"] == "PENDING"
    hb = _status(tmp_path)["heartbeat"]
    assert hb["state"] == "paused"
    _write_control(tmp_path, {"resume": True})
    th.join(timeout=60)
    assert res.get("rc") == RO.EXIT_ALL_DONE and _by(tmp_path)["b"]["status"] == "DONE"


def test_restart_orchestrator_exit_code(tmp_path):
    steps = [_fake("a", tmp_path)]
    _write_control(tmp_path, {"restart_orchestrator": True})
    assert _chain(steps, tmp_path).run() == RO.EXIT_RESTART
    assert _by(tmp_path)["a"]["status"] == "PENDING"                         # adım sınırında çıktı


def test_heartbeat_reports_progress_line(tmp_path):
    steps = [_fake("a", tmp_path, n=6)]
    ch = _chain(steps, tmp_path)
    th = threading.Thread(target=ch.run)
    th.start()
    seen = []

    def got():
        try:
            hb = _status(tmp_path).get("heartbeat") or {}
        except Exception:
            return False
        if hb.get("current_step") == "a" and hb.get("last_progress_line"):
            seen.append(hb)
            return True
        return False
    assert _wait_for(got)
    th.join(timeout=60)
    assert "[" in seen[0]["last_progress_line"] and "/6]" in seen[0]["last_progress_line"]
    assert seen[0]["experiment_log"].endswith("a.explog")


def test_live_lock_blocks_second_copy(tmp_path):
    holder = subprocess.Popen([PY, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / "overnight.lock").write_text(json.dumps({"pid": holder.pid, "started": "x"}))
        assert _chain([_fake("x", tmp_path)], tmp_path).run() == RO.EXIT_LOCKED
        assert not (tmp_path / "x.DONE.json").exists() and (tmp_path / "overnight.lock").exists()
    finally:
        holder.kill()


def test_stale_lock_is_cleaned(tmp_path):
    dead = subprocess.Popen([PY, "-c", "pass"])
    dead.wait()
    time.sleep(0.2)
    (tmp_path / "overnight.lock").write_text(json.dumps({"pid": dead.pid, "started": "x"}))
    assert _chain([_fake("x", tmp_path)], tmp_path).run() == RO.EXIT_ALL_DONE
    assert "bayat kilit" in (tmp_path / "overnight.log").read_text(encoding="utf-8")


def test_stop_file_blocks_start_and_kills_running_step(tmp_path):
    (tmp_path / "overnight.STOP").write_text("")
    assert _chain([_fake("x", tmp_path)], tmp_path).run() == RO.EXIT_STOP
    (tmp_path / "overnight.STOP").unlink()
    stop = tmp_path / "overnight.STOP"
    steps = [_step("long", f"import time, pathlib; time.sleep(1); pathlib.Path(r'{stop}').write_text('');"
                           " time.sleep(60)", tmp_path)]
    t0 = time.time()
    assert _chain(steps, tmp_path).run() == RO.EXIT_STOP
    assert time.time() - t0 < 30 and _by(tmp_path)["long"]["status"] == "PENDING"


def test_parse_control_validation():
    assert RO.parse_control('{"retry": ["e4"]}') == {"retry": ["e4"]}
    for bad in ('[]', '{}', '{"retry": "e4"}', '{"stop_after_current": "yes"}', '{"x": 1}', 'nope'):
        with pytest.raises(Exception):
            RO.parse_control(bad)


def test_pid_alive_python():
    assert RO.pid_alive_python(__import__("os").getpid())
    assert not RO.pid_alive_python(0)


def test_real_step_list_matches_done_names():
    """Gerçek adımların DONE adları deney modüllerinin yazdığı adlarla aynı olmalı."""
    from src.m07_revision import common as C
    from src.m07_revision import e3_sensitivity as E3
    from src.m07_revision import e5_permutation as E5
    names = [s["done"] for s in RO.build_steps("python")]
    d3 = C.experiment_dirs("E3_sensitivity", "full")
    assert E3._done_path(d3, {"grouping"}).name == names[0] and E3._done_path(d3, set()).name == names[2]
    assert f"E5_permutation{E5._run_tag({'ryckewaert/lasso'})}.DONE.json" == names[4]
    assert f"E5_permutation{E5._run_tag(None, {'ryckewaert/lasso'}, 50)}.DONE.json" == names[6]
    assert names[1] == "E6_ga.G0-G1.DONE.json" and names[3] == "E6_ga.G2-G3.DONE.json"
    assert names[5] == "E4_simulation.DONE.json"
    assert all((C.ROOT / "outputs_rev1" / "logs").name == "logs" for _ in names)
