"""07. 콜백, 실행 제어, 예외 처리, 사용자 속성, 로깅

다루는 기능
- callbacks=[...] (사용자 정의 콜백: 로깅, 조기 종료)
- optuna.study.MaxTrialsCallback (분산 환경에서 총 trial 수 제한)
- study.stop() (콜백/목적함수 안에서 중단)
- timeout (초 단위 제한)
- catch=(Exception,) (특정 예외 발생 시 FAIL 처리 후 계속)
- trial.set_user_attr / study.set_user_attr
- gc_after_trial, show_progress_bar
- optuna.logging (verbosity, 핸들러, 로그 파일)
- 반환값 None/NaN 및 TrialPruned 처리 결과
"""
import logging
import sys

from common import header, out, quadratic

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    value = quadratic(x, y)
    trial.set_user_attr("distance", (x**2 + y**2) ** 0.5)  # 부가 정보 저장
    return value


class EarlyStoppingCallback:
    """best 값이 patience 회 동안 개선되지 않으면 study.stop()."""

    def __init__(self, patience: int = 15) -> None:
        self.patience = patience
        self.best = float("inf")
        self.stale = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if study.best_value < self.best - 1e-6:
            self.best, self.stale = study.best_value, 0
        else:
            self.stale += 1
        if self.stale >= self.patience:
            print(f"    → {self.patience}회 개선 없음, trial #{trial.number} 에서 중단")
            study.stop()


def log_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    if trial.number % 10 == 0:
        print(f"    [cb] trial #{trial.number:3d} value={trial.value:.4f} best={study.best_value:.4f}")


def flaky_objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    if trial.number % 5 == 4:  # 5번에 1번 실패
        raise ValueError("외부 시스템 오류 시뮬레이션")
    return x**2


def main() -> None:
    header("07-1. 사용자 정의 콜백 (로깅 + 조기 종료)")
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=500, callbacks=[log_callback, EarlyStoppingCallback(15)])
    print("  실제 실행된 trial 수:", len(study.trials))

    header("07-2. MaxTrialsCallback (COMPLETE 상태 trial 수 기준 제한)")
    study = optuna.create_study()
    study.optimize(objective, n_trials=1000,
                   callbacks=[MaxTrialsCallback(25, states=(TrialState.COMPLETE,))])
    print("  trial 수:", len(study.trials))

    header("07-3. timeout (초 단위)")
    study = optuna.create_study()
    study.optimize(objective, timeout=1.0)
    print("  1초 동안 실행된 trial 수:", len(study.trials))

    header("07-4. catch: 예외 발생 trial 은 FAIL 로 기록하고 계속 진행")
    optuna.logging.set_verbosity(optuna.logging.ERROR)  # 실패 trial 의 [W] traceback 로그 숨김
    study = optuna.create_study()
    study.optimize(flaky_objective, n_trials=30, catch=(ValueError,))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    states = [t.state.name for t in study.trials]
    print("  COMPLETE:", states.count("COMPLETE"), "| FAIL:", states.count("FAIL"))

    header("07-5. 목적함수 안에서 study.stop()")

    def obj_stop(trial: optuna.Trial) -> float:
        v = objective(trial)
        if v < 1.0:
            trial.study.stop()
        return v

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
    study.optimize(obj_stop, n_trials=1000)
    print(f"  value<1 도달 시 중단: {len(study.trials)} trials, best={study.best_value:.4f}")

    header("07-6. user_attrs 활용")
    study = optuna.create_study()
    study.set_user_attr("experiment", "callbacks-demo")
    study.optimize(objective, n_trials=5)
    print("  study.user_attrs:", study.user_attrs)
    print("  best trial distance:", round(study.best_trial.user_attrs["distance"], 3))

    header("07-7. gc_after_trial / show_progress_bar")
    study = optuna.create_study()
    study.optimize(objective, n_trials=20, gc_after_trial=True, show_progress_bar=True)
    print("  완료:", len(study.trials))

    header("07-8. 로깅 제어")
    print("  현재 verbosity:", optuna.logging.get_verbosity())
    optuna.logging.set_verbosity(optuna.logging.INFO)
    log_path = out("07_optuna.log")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    optuna.logging.get_logger("optuna").addHandler(fh)
    optuna.logging.disable_default_handler()     # 콘솔 출력 끄기 (파일에만 기록)
    study = optuna.create_study(study_name="logging-demo")
    study.optimize(objective, n_trials=3)
    optuna.logging.enable_default_handler()
    optuna.logging.get_logger("optuna").removeHandler(fh)
    fh.close()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with open(log_path, encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    print(f"  로그 파일 {log_path} 에 {len(lines)} 줄 기록, 예: {lines[-1][:80]}...")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
