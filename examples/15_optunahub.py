"""15. OptunaHub - 커뮤니티 샘플러/기능 패키지 불러오기

다루는 기능
- optunahub.load_module("samplers/auto_sampler") → AutoSampler
  (문제 유형에 따라 GP / TPE / NSGA-II 등을 자동 선택)
- deprecated 된 기능(CMA-ES 재시작 전략, Terminator 등)의 이전 위치 확인용
- 패키지 목록: https://hub.optuna.org

필요: pip install optunahub  (실행 시 GitHub 에서 패키지 소스를 내려받으므로 인터넷 필요)
"""
from common import header, rosenbrock

import optuna


def rosen_obj(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -3, 3)
    y = trial.suggest_float("y", -3, 3)
    return rosenbrock(x, y)


def mo_obj(trial: optuna.Trial):
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)
    return 4 * x**2 + 4 * y**2, (x - 5) ** 2 + (y - 5) ** 2


def main() -> None:
    try:
        import optunahub
    except ImportError:
        print("optunahub 미설치: pip install optunahub")
        return

    header("15-1. AutoSampler (단일 목적)")
    try:
        mod = optunahub.load_module("samplers/auto_sampler")
    except Exception as e:  # 네트워크 차단 등
        print("  OptunaHub 모듈 로드 실패 (인터넷/GitHub 접근 필요):", e)
        return
    study = optuna.create_study(sampler=mod.AutoSampler(seed=0))
    study.optimize(rosen_obj, n_trials=40)
    print(f"  best={study.best_value:.4f} params={ {k: round(v, 3) for k, v in study.best_params.items()} }")

    header("15-2. AutoSampler (다목적)")
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=mod.AutoSampler(seed=0))
    study.optimize(mo_obj, n_trials=60)
    print("  파레토 해 개수:", len(study.best_trials))

    print("\n  다른 패키지 예: optunahub.load_module('samplers/hebo'),"
          " 'samplers/implicit_natural_gradient', 'samplers/restart_cmaes' 등 (hub.optuna.org 참고)")


if __name__ == "__main__":
    main()
