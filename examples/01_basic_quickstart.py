"""01. 기본 사용법 & suggest API & 동적(조건부) 탐색 공간

다루는 기능
- optuna.create_study / study.optimize / best_value / best_params / best_trial
- trial.suggest_float (uniform, log, step)
- trial.suggest_int (step, log)
- trial.suggest_categorical
- Define-by-Run: 조건부 / 반복문 기반 동적 탐색 공간
- study.trials, study.trials_dataframe()
"""
from common import header, quadratic, out

import optuna


def simple_objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    return quadratic(x, y)


def all_suggest_objective(trial: optuna.Trial) -> float:
    # 실수형
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)          # 로그 스케일
    dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)  # 이산 간격
    momentum = trial.suggest_float("momentum", 0.0, 1.0)          # 균등
    # 정수형
    n_layers = trial.suggest_int("n_layers", 1, 4)
    batch = trial.suggest_int("batch_size", 16, 256, log=True)    # 로그 스케일 정수
    units = trial.suggest_int("units", 32, 256, step=32)          # 간격 32
    # 범주형
    optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam", "rmsprop"])
    use_bn = trial.suggest_categorical("use_bn", [True, False])

    # 가짜 '검증 손실' (실제 학습 대신 수식으로 대체)
    score = (abs(lr - 1e-3) * 100 + dropout + abs(momentum - 0.9)
             + abs(n_layers - 2) * 0.3 + abs(batch - 64) / 256 + abs(units - 128) / 256
             + {"sgd": 0.3, "adam": 0.0, "rmsprop": 0.1}[optimizer]
             + (0.0 if use_bn else 0.2))
    return score


def conditional_objective(trial: optuna.Trial) -> float:
    """Define-by-Run: 앞에서 고른 값에 따라 탐색 공간이 달라진다."""
    model = trial.suggest_categorical("model", ["svm", "rf", "mlp"])
    if model == "svm":
        c = trial.suggest_float("svm_c", 1e-3, 1e3, log=True)
        kernel = trial.suggest_categorical("svm_kernel", ["linear", "rbf"])
        return abs(c - 1.0) + (0.1 if kernel == "linear" else 0.0)
    if model == "rf":
        depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        return abs(depth - 8) / 10
    # mlp: 레이어 수에 따라 파라미터 수가 동적으로 변한다
    n_layers = trial.suggest_int("mlp_n_layers", 1, 3)
    loss = 0.0
    for i in range(n_layers):
        units = trial.suggest_int(f"mlp_units_l{i}", 4, 128, log=True)
        loss += abs(units - 64) / 64
    return loss + 0.05


def main() -> None:
    header("01-1. 가장 단순한 최적화")
    study = optuna.create_study(direction="minimize", study_name="quickstart")
    study.optimize(simple_objective, n_trials=50)
    print("best_value :", round(study.best_value, 5))
    print("best_params:", {k: round(v, 3) for k, v in study.best_params.items()})
    print("best_trial #", study.best_trial.number)
    print("전체 trial 수:", len(study.trials))

    header("01-2. 모든 suggest_* API")
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(all_suggest_objective, n_trials=60)
    print("best_value :", round(study.best_value, 4))
    print("best_params:", study.best_params)
    # 분포(distribution) 정보 확인
    for name, dist in study.best_trial.distributions.items():
        print(f"  {name:12s} -> {dist}")

    header("01-3. 조건부 / 동적 탐색 공간 (Define-by-Run)")
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(conditional_objective, n_trials=80)
    print("best_params:", study.best_params)
    print("trial 별로 다른 파라미터 집합 예시:")
    for t in study.trials[:5]:
        print(f"  #{t.number}: {sorted(t.params)}")

    header("01-4. trials_dataframe 로 결과 내보내기")
    df = study.trials_dataframe(attrs=("number", "value", "params", "state", "duration"))
    print(df.head())
    df.to_csv(out("01_trials.csv"), index=False)
    print("저장:", out("01_trials.csv"))

    header("01-5. maximize 방향")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: -simple_objective(t), n_trials=30)
    print("direction:", study.direction.name, "| best_value:", round(study.best_value, 4))


if __name__ == "__main__":
    main()
