"""06. Ask-and-Tell 인터페이스 & Trial 직접 다루기

다루는 기능
- study.ask() / study.tell() (외부 루프, 배치 평가)
- define-and-run: ask(fixed_distributions)
- tell 에 state 지정 (PRUNED / FAIL)
- study.enqueue_trial (사전 지식/기본값을 먼저 시도, skip_if_exists)
- study.add_trial / add_trials + optuna.trial.create_trial (과거 결과 주입, warm start)
- optuna.trial.FixedTrial (목적함수 단위 테스트, 고정 파라미터 재실행)
- trial.number, trial.params, trial.datetime_start/complete, duration
"""
import numpy as np
from common import header, quadratic

import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import TrialState, create_trial


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    return quadratic(x, y)


def main() -> None:
    header("06-1. 기본 Ask-and-Tell 루프")
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    for _ in range(30):
        trial = study.ask()
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        study.tell(trial, quadratic(x, y))
    print("  best:", round(study.best_value, 4), study.best_params)

    header("06-2. Define-and-Run: 분포를 미리 정의해 ask(fixed_distributions)")
    dists = {
        "lr": FloatDistribution(1e-5, 1e-1, log=True),
        "depth": IntDistribution(1, 10),
        "act": CategoricalDistribution(["relu", "tanh"]),
    }
    study = optuna.create_study()
    for _ in range(20):
        t = study.ask(dists)  # 파라미터가 이미 샘플링되어 있음
        p = t.params
        loss = abs(np.log10(p["lr"]) + 3) + abs(p["depth"] - 5) / 5 + (0 if p["act"] == "relu" else 0.3)
        study.tell(t, loss)
    print("  best:", study.best_params)

    header("06-3. 배치(batch) 평가: 여러 trial 을 한꺼번에 ask → 병렬 평가 → tell")
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    for _ in range(5):
        batch = [study.ask() for _ in range(8)]
        xs = np.array([[t.suggest_float("x", -10, 10), t.suggest_float("y", -10, 10)] for t in batch])
        values = (xs[:, 0] - 2) ** 2 + (xs[:, 1] + 1) ** 2  # 벡터화된 일괄 평가
        for t, v in zip(batch, values):
            study.tell(t, float(v))
    print("  5 배치 x 8 =", len(study.trials), "trials, best:", round(study.best_value, 4))

    header("06-4. tell 에 상태 지정 (PRUNED / FAIL)")
    study = optuna.create_study()
    t = study.ask()
    t.suggest_float("x", 0, 1)
    t.report(0.5, step=0)
    study.tell(t, state=TrialState.PRUNED)
    t = study.ask()
    t.suggest_float("x", 0, 1)
    study.tell(t, state=TrialState.FAIL)
    t = study.ask()
    study.tell(t, float("nan"))  # NaN 반환 → 자동으로 FAIL
    print("  상태:", [tr.state.name for tr in study.trials])

    header("06-5. enqueue_trial: 알려진 좋은 설정을 먼저 평가")
    study = optuna.create_study()
    study.enqueue_trial({"x": 2.0, "y": -1.0}, user_attrs={"source": "domain-knowledge"})
    study.enqueue_trial({"x": 0.0})  # 일부만 지정 → 나머지는 샘플러가 채움
    study.enqueue_trial({"x": 2.0, "y": -1.0}, skip_if_exists=True)  # 중복이면 건너뜀
    study.optimize(objective, n_trials=5)
    for tr in study.trials[:3]:
        print(f"  #{tr.number} params={ {k: round(v, 3) for k, v in tr.params.items()} } "
              f"value={tr.value:.4f} attrs={tr.user_attrs}")

    header("06-6. add_trial / create_trial: 과거 실험 결과 주입 (warm start)")
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    dist = {"x": FloatDistribution(-10, 10), "y": FloatDistribution(-10, 10)}
    past = [create_trial(params={"x": x, "y": y}, distributions=dist, value=quadratic(x, y))
            for x, y in [(1.5, -0.5), (2.5, -1.5), (3.0, 0.0)]]
    study.add_trials(past)
    study.add_trial(create_trial(params={"x": 1.9, "y": -1.1}, distributions=dist,
                                 value=quadratic(1.9, -1.1), user_attrs={"from": "paper"}))
    print("  주입 직후 best:", round(study.best_value, 4))
    study.optimize(objective, n_trials=20)
    print("  추가 최적화 후 trial 수:", len(study.trials), "best:", round(study.best_value, 4))

    header("06-7. FixedTrial: 목적함수 테스트 / 특정 파라미터로 재실행")
    val = objective(optuna.trial.FixedTrial({"x": 2.0, "y": -1.0}))
    print("  FixedTrial(x=2, y=-1) ->", val)
    val = objective(optuna.trial.FixedTrial(study.best_params))
    print("  FixedTrial(best_params) ->", round(val, 4))

    header("06-8. Trial 메타데이터")
    bt = study.best_trial
    print("  number:", bt.number, "| state:", bt.state.name)
    print("  start:", bt.datetime_start, "| duration:", bt.duration)
    print("  distributions:", bt.distributions)


if __name__ == "__main__":
    main()
