"""04. 다목적(Multi-objective) 최적화 & 제약조건(Constraints)

다루는 기능
- create_study(directions=[...]) / study.best_trials (파레토 최적해)
- study.set_metric_names (목적함수 이름 부여)
- NSGAIISampler / NSGAIIISampler / TPESampler / GPSampler 다목적 지원
- NSGAIISampler 의 crossover 종류 (SBX, BLXAlpha, Uniform, SPX, UNDX, VSBX)
- trial.set_constraint(key, value): 제약조건 (v5.0 신규 API, 값 <= 0 이면 만족)
  (v4 까지의 sampler(constraints_func=...) 방식은 v5.0 에서 deprecated)
- 하이퍼볼륨(Hypervolume) 계산 & 파레토 프론트 시각화
"""
import numpy as np
from common import header, out

import optuna
from optuna.samplers import NSGAIISampler, NSGAIIISampler, TPESampler
from optuna.samplers.nsgaii import (
    BLXAlphaCrossover,
    SBXCrossover,
    SPXCrossover,
    UNDXCrossover,
    UniformCrossover,
    VSBXCrossover,
)


def binh_korn(trial: optuna.Trial) -> tuple[float, float]:
    """Binh & Korn 문제: 두 목적 모두 최소화, 제약 2개."""
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)
    v0 = 4 * x**2 + 4 * y**2
    v1 = (x - 5) ** 2 + (y - 5) ** 2
    # 제약조건: c <= 0 이면 '만족'. v5 부터는 trial.set_constraint 로 기록하면
    # 제약을 지원하는 샘플러(NSGAII/III, TPE, GP)와 시각화가 자동으로 사용한다.
    trial.set_constraint("c0", (x - 5) ** 2 + y**2 - 25)
    trial.set_constraint("c1", -((x - 8) ** 2) - (y + 3) ** 2 + 7.7)
    return v0, v1


def is_feasible(trial: optuna.trial.FrozenTrial) -> bool:
    return all(v <= 0 for v in trial.constraints.values())


def hypervolume(study: optuna.Study, ref: tuple[float, float]) -> float:
    """2목적 최소화 문제의 하이퍼볼륨 (간단 구현)."""
    pts = sorted({tuple(t.values) for t in study.best_trials if is_feasible(t)})
    hv, prev_y = 0.0, ref[1]
    for x, y in pts:
        if x < ref[0] and y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return hv


def main() -> None:
    ref = (150.0, 60.0)

    header("04-1. NSGA-II 다목적 + 제약조건")
    sampler = NSGAIISampler(population_size=30, seed=0)
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
    study.set_metric_names(["f1_cost", "f2_latency"])
    study.optimize(binh_korn, n_trials=300)
    print("metric_names:", study.metric_names)
    print("파레토 최적해 개수:", len(study.best_trials))
    for t in study.best_trials[:5]:
        feasible = is_feasible(t)
        print(f"  #{t.number:3d} values={np.round(t.values, 2)} params="
              f"{ {k: round(v, 2) for k, v in t.params.items()} } feasible={feasible}")
    print("constraints 예:", {k: round(v, 2) for k, v in study.best_trials[0].constraints.items()})
    print("하이퍼볼륨:", round(hypervolume(study, ref), 2))

    header("04-2. 다목적 샘플러 비교 (하이퍼볼륨, 클수록 좋음)")
    samplers = {
        "NSGAII": NSGAIISampler(population_size=30, seed=0),
        "NSGAIII": NSGAIIISampler(population_size=30, seed=0),
        "TPE(MOTPE)": TPESampler(seed=0),
        "Random": optuna.samplers.RandomSampler(seed=0),
    }
    try:
        samplers["GP"] = optuna.samplers.GPSampler(seed=0)
    except ImportError:
        pass
    for name, s in samplers.items():
        n = 60 if name == "GP" else 200
        st = optuna.create_study(directions=["minimize", "minimize"], sampler=s)
        st.optimize(binh_korn, n_trials=n)
        print(f"  {name:12s} n_trials={n:3d}  HV={hypervolume(st, ref):9.2f}  pareto={len(st.best_trials)}")

    header("04-3. NSGA-II crossover 연산자 비교")
    for cx in [UniformCrossover(), BLXAlphaCrossover(), SBXCrossover(), VSBXCrossover(),
               UNDXCrossover(), SPXCrossover()]:
        st = optuna.create_study(directions=["minimize", "minimize"],
                                 sampler=NSGAIISampler(population_size=30, seed=0, crossover=cx))
        st.optimize(binh_korn, n_trials=200)
        print(f"  {type(cx).__name__:20s} HV={hypervolume(st, ref):9.2f}")

    header("04-4. 3목적 최적화 (minimize, minimize, maximize)")

    def three(trial: optuna.Trial):
        x = trial.suggest_float("x", 0, 1)
        y = trial.suggest_float("y", 0, 1)
        return x, (1 - x) * y, x * (1 - y)

    st = optuna.create_study(directions=["minimize", "minimize", "maximize"],
                             sampler=NSGAIIISampler(seed=0))
    st.optimize(three, n_trials=150)
    print("  3목적 파레토 해 개수:", len(st.best_trials))

    header("04-5. 파레토 프론트 / 하이퍼볼륨 히스토리 시각화 (HTML)")
    fig = optuna.visualization.plot_pareto_front(study)  # 저장된 constraints 로 feasible/infeasible 구분
    fig.write_html(out("04_pareto_front.html"))
    fig = optuna.visualization.plot_hypervolume_history(study, reference_point=list(ref))
    fig.write_html(out("04_hypervolume_history.html"))
    fig = optuna.visualization.plot_pareto_front(st)  # 3D 파레토
    fig.write_html(out("04_pareto_front_3d.html"))
    print("  저장:", out("04_pareto_front.html"), "외 2개")

    header("04-6. 다목적에서 특정 목적만 골라 시각화 (target 사용)")
    fig = optuna.visualization.plot_param_importances(
        study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0),
        target=lambda t: t.values[0], target_name="f1_cost")
    fig.write_html(out("04_importance_f1.html"))
    print("  저장:", out("04_importance_f1.html"))


if __name__ == "__main__":
    main()
