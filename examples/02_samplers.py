"""02. 샘플러(Sampler) 전체 비교

다루는 기능
- TPESampler (기본, multivariate, group, constant_liar, n_startup_trials, seed)
- RandomSampler
- CmaEsSampler (기본, popsize, use_separable_cma, with_margin)
- GPSampler (가우시안 프로세스 기반 베이지안 최적화, torch 필요)
- QMCSampler (Sobol / Halton 준난수)
- GridSampler / BruteForceSampler
- NSGAIISampler / NSGAIIISampler (단일목적에도 사용 가능, 다목적은 04 참고)
- PartialFixedSampler (일부 파라미터 고정)
- 샘플러 직렬화 (pickle) 로 재현성 유지
"""
import pickle
import time

from common import header, rastrigin, rosenbrock

import optuna
from optuna.samplers import (
    BruteForceSampler,
    CmaEsSampler,
    GPSampler,
    GridSampler,
    NSGAIISampler,
    NSGAIIISampler,
    PartialFixedSampler,
    QMCSampler,
    RandomSampler,
    TPESampler,
)

N_TRIALS = 80


def rosen_obj(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -3, 3)
    y = trial.suggest_float("y", -3, 3)
    return rosenbrock(x, y)


def rastrigin_obj(trial: optuna.Trial) -> float:
    xs = [trial.suggest_float(f"x{i}", -5.12, 5.12) for i in range(4)]
    return rastrigin(xs)


def run(name: str, sampler: optuna.samplers.BaseSampler, obj, n_trials: int = N_TRIALS) -> None:
    t0 = time.time()
    study = optuna.create_study(sampler=sampler)
    study.optimize(obj, n_trials=n_trials)
    print(f"  {name:38s} best={study.best_value:10.4f}  ({time.time() - t0:5.1f}s)")


def main() -> None:
    header("02-1. 연속형 문제(Rosenbrock)에서 샘플러 비교")
    run("RandomSampler", RandomSampler(seed=0), rosen_obj)
    run("TPESampler (기본)", TPESampler(seed=0), rosen_obj)
    run("TPESampler (multivariate=True)", TPESampler(seed=0, multivariate=True), rosen_obj)
    run("TPESampler (n_startup_trials=20)", TPESampler(seed=0, n_startup_trials=20), rosen_obj)
    run("CmaEsSampler", CmaEsSampler(seed=0), rosen_obj)
    run("CmaEsSampler (use_separable_cma)", CmaEsSampler(seed=0, use_separable_cma=True), rosen_obj)
    run("QMCSampler (sobol)", QMCSampler(qmc_type="sobol", seed=0), rosen_obj)
    run("QMCSampler (halton)", QMCSampler(qmc_type="halton", seed=0), rosen_obj)
    run("NSGAIISampler", NSGAIISampler(seed=0, population_size=20), rosen_obj)
    run("NSGAIIISampler", NSGAIIISampler(seed=0, population_size=20), rosen_obj)
    try:
        run("GPSampler", GPSampler(seed=0), rosen_obj, n_trials=40)
    except ImportError as e:  # torch 미설치 시
        print("  GPSampler 건너뜀 (torch 필요):", e)

    header("02-2. 다봉 함수(Rastrigin 4D): CMA-ES popsize 조정")
    run("CmaEsSampler (기본)", CmaEsSampler(seed=0), rastrigin_obj, 200)
    run("CmaEsSampler (popsize=16)", CmaEsSampler(seed=0, popsize=16), rastrigin_obj, 200)
    # 참고: restart_strategy("ipop"/"bipop")는 v4.4 부터 deprecated → OptunaHub 의 CMA-ES 패키지 사용
    run("TPESampler (multivariate)", TPESampler(seed=0, multivariate=True), rastrigin_obj, 200)

    header("02-3. 혼합(정수/범주) 공간 + CMA-ES with_margin")

    def mixed(trial: optuna.Trial) -> float:
        x = trial.suggest_int("x", -10, 10)
        y = trial.suggest_int("y", -10, 10)
        return (x - 3) ** 2 + (y + 2) ** 2

    run("CmaEsSampler (with_margin=True)", CmaEsSampler(seed=0, with_margin=True), mixed, 60)

    header("02-4. TPE group=True (조건부 공간에서 그룹 단위 multivariate)")

    def cond(trial: optuna.Trial) -> float:
        kind = trial.suggest_categorical("kind", ["a", "b"])
        if kind == "a":
            return trial.suggest_float("a1", -5, 5) ** 2 + trial.suggest_float("a2", -5, 5) ** 2
        return 1 + trial.suggest_float("b1", -5, 5) ** 2

    run("TPESampler (multivariate, group)", TPESampler(seed=0, multivariate=True, group=True), cond)

    header("02-5. GridSampler (모든 조합 탐색)")
    search_space = {"x": [-2, -1, 0, 1, 2], "y": [-1, 0, 1]}
    study = optuna.create_study(sampler=GridSampler(search_space, seed=0))
    study.optimize(lambda t: t.suggest_float("x", -2, 2) ** 2 + t.suggest_float("y", -1, 1) ** 2)
    print(f"  n_trials 미지정 → 그리드 소진 시 자동 종료: {len(study.trials)} trials, best={study.best_params}")

    header("02-6. BruteForceSampler (이산 공간 전수 탐색, 조건부도 지원)")

    def bf(trial: optuna.Trial) -> float:
        a = trial.suggest_int("a", 0, 3)
        if a % 2 == 0:
            b = trial.suggest_categorical("b", ["x", "y"])
            return a + (0 if b == "x" else 0.5)
        return a * 10.0

    study = optuna.create_study(sampler=BruteForceSampler(seed=0))
    study.optimize(bf)
    print(f"  전수 탐색 trial 수: {len(study.trials)}, best={study.best_params}")

    header("02-7. PartialFixedSampler (일부 파라미터 고정)")
    base = TPESampler(seed=0)
    sampler = PartialFixedSampler(fixed_params={"y": 1.0}, base_sampler=base)
    study = optuna.create_study(sampler=sampler)
    study.optimize(rosen_obj, n_trials=40)
    print("  y 고정(=1.0):", study.best_params)

    header("02-8. 샘플러 pickle 저장/복원 (재현성 & 재개)")
    sampler = TPESampler(seed=42)
    study = optuna.create_study(sampler=sampler)
    study.optimize(rosen_obj, n_trials=10)
    blob = pickle.dumps(study.sampler)
    restored = pickle.loads(blob)
    print("  복원된 샘플러 타입:", type(restored).__name__, f"({len(blob)} bytes)")


if __name__ == "__main__":
    main()
