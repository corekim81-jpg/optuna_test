"""16. Optuna CLI & Optuna Dashboard

다루는 기능 (CLI: `optuna --help`)
- optuna create-study / study-names / studies / trials / best-trial / best-trials
- optuna ask / tell  (언어에 상관없이 쉘 스크립트로 최적화 루프 구성 가능)
- optuna study set-user-attr / delete-study / storage upgrade
- optuna-dashboard 로 웹 UI 확인 (pip install optuna-dashboard)

이 스크립트는 위 CLI 명령을 subprocess 로 실행해 결과를 보여주고,
대시보드에서 볼 수 있는 study 를 outputs/dashboard.log 에 만든다.
"""
import json
import shutil
import subprocess
import sys

from common import fresh_dashboard_study, header, out, remove, sqlite_url

import optuna

DB = out("16_cli.db")
URL = sqlite_url("16_cli.db")


def optuna_cli(*args: str) -> str:
    exe = shutil.which("optuna")
    cmd = [exe] if exe else [sys.executable, "-m", "optuna.cli"]
    cmd += ["--storage", URL, *args]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        raise RuntimeError(res.stderr)
    return res.stdout.strip()


def main() -> None:
    remove(DB)

    header("16-1. CLI: create-study / study set-user-attr / study-names")
    print(optuna_cli("create-study", "--study-name", "cli-study", "--direction", "minimize"))
    optuna_cli("study", "set-user-attr", "--study-name", "cli-study", "--key", "owner", "--value", "K")
    print("  study-names:", optuna_cli("study-names"))

    header("16-2. CLI: ask / tell 루프 (쉘에서도 동일하게 사용 가능)")
    space = json.dumps({
        "x": {"name": "FloatDistribution", "attributes": {"low": -10.0, "high": 10.0}},
        "y": {"name": "IntDistribution", "attributes": {"low": -10, "high": 10}},
    })
    for _ in range(10):
        trial = json.loads(optuna_cli("ask", "--study-name", "cli-study",
                                      # 주의: ask 마다 샘플러가 새로 생성되므로 seed 를 고정하면 같은 값만 나온다
                                      "--sampler", "TPESampler",
                                      "--search-space", space, "-f", "json"))
        x, y = trial["params"]["x"], trial["params"]["y"]
        optuna_cli("tell", "--study-name", "cli-study", "--trial-number", str(trial["number"]),
                   "--values", str((x - 2) ** 2 + (y + 1) ** 2))
    print("  10 trials ask/tell 완료")

    header("16-3. CLI: studies / trials / best-trial")
    print(optuna_cli("studies"))
    print(optuna_cli("trials", "--study-name", "cli-study", "-f", "table")[:1200])
    print(optuna_cli("best-trial", "--study-name", "cli-study", "-f", "yaml"))

    header("16-4. 대시보드용 DB 생성")
    for name, sampler in [("dash-tpe", optuna.samplers.TPESampler(seed=0)),
                          ("dash-random", optuna.samplers.RandomSampler(seed=0))]:
        st = fresh_dashboard_study(f"16_cli/{name}", sampler=sampler)

        def obj(trial: optuna.Trial) -> float:
            x = trial.suggest_float("x", -5, 5)
            y = trial.suggest_float("y", -5, 5)
            for s in range(5):
                trial.report((x**2 + y**2) * (1 + 1 / (s + 1)), s)
            return x**2 + y**2

        st.optimize(obj, n_trials=50)
    mo = fresh_dashboard_study("16_cli/dash-multiobjective", directions=["minimize", "minimize"])
    mo.set_metric_names(["cost", "latency"])
    mo.optimize(lambda t: (t.suggest_float("a", 0, 1), 1 - t.suggest_float("a", 0, 1) ** 0.5), n_trials=50)
    print("  저장: outputs/dashboard.log")
    print("  실행(레포 루트에서): python dashboard.py  → http://127.0.0.1:8080")
    print("  CLI 로 보려면 --storage outputs/dashboard.log --storage-class JournalFileBackend (16_cli.db 는 CLI 데모 전용)")


if __name__ == "__main__":
    main()
