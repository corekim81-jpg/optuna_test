"""13. Distributions / Search Space 유틸리티 / 결과 분석

다루는 기능
- FloatDistribution / IntDistribution / CategoricalDistribution 직접 생성
- distribution_to_json / json_to_distribution (직렬화)
- check_distribution_compatibility
- distribution.to_internal_repr / to_external_repr / single()
- optuna.search_space.IntersectionSearchSpace (공통 탐색 공간 계산)
- optuna.search_space.intersection_search_space
- study.get_trials(states=..., deepcopy=False)
- study.best_trial vs 상위 N개 추출, trials_dataframe(multi_index=True)
"""
from common import header

import optuna
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
    check_distribution_compatibility,
    distribution_to_json,
    json_to_distribution,
)
from optuna.search_space import IntersectionSearchSpace, intersection_search_space
from optuna.trial import TrialState


def objective(trial: optuna.Trial) -> float:
    kind = trial.suggest_categorical("kind", ["A", "B"])
    x = trial.suggest_float("x", -5, 5)
    if kind == "A":
        a = trial.suggest_int("a", 1, 10)
        return x**2 + a
    b = trial.suggest_float("b", 0.1, 10, log=True)
    return x**2 + b


def main() -> None:
    header("13-1. Distribution 객체")
    dists = [
        FloatDistribution(0.0, 1.0),
        FloatDistribution(1e-5, 1e-1, log=True),
        FloatDistribution(0.0, 1.0, step=0.25),
        IntDistribution(1, 100, log=True),
        IntDistribution(0, 10, step=2),
        CategoricalDistribution(["a", "b", None, 1, 2.5]),
        FloatDistribution(3.0, 3.0),  # 값이 하나뿐인 분포
    ]
    for d in dists:
        j = distribution_to_json(d)
        back = json_to_distribution(j)
        print(f"  {str(d):70s} single={d.single()!s:5s} roundtrip={back == d}")

    header("13-2. internal / external 표현 (범주형은 인덱스로 저장됨)")
    c = CategoricalDistribution(["relu", "tanh", "gelu"])
    print("  external 'gelu' → internal", c.to_internal_repr("gelu"), "→ external", c.to_external_repr(2.0))

    header("13-3. 분포 호환성 검사 (같은 이름에 다른 분포를 쓰면 오류)")
    try:
        check_distribution_compatibility(FloatDistribution(0, 1), IntDistribution(0, 1))
    except ValueError as e:
        print("  ValueError:", e)
    check_distribution_compatibility(FloatDistribution(0, 1), FloatDistribution(0, 2))
    print("  Float(0,1) vs Float(0,2) → 호환 (범위만 다름, OK)")

    header("13-4. IntersectionSearchSpace (조건부 공간의 공통 파라미터)")
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=30)
    iss = IntersectionSearchSpace()
    print("  모든 trial 공통 파라미터:", sorted(iss.calculate(study)))
    print("  함수형 API          :", sorted(intersection_search_space(study.trials)))
    a_trials = [t for t in study.trials if t.params.get("kind") == "A"]
    print("  kind=A trial 들의 공통:", sorted(intersection_search_space(a_trials)))

    header("13-5. trial 필터링 & 상위 N개")
    complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    top3 = sorted(complete, key=lambda t: t.value)[:3]
    for t in top3:
        print(f"  #{t.number:2d} value={t.value:.4f} params={t.params}")

    header("13-6. trials_dataframe(multi_index=True)")
    df = study.trials_dataframe(multi_index=True)
    print("  컬럼 레벨:", list(df.columns.levels[0]))
    print(df[["value", "params"]].head(3))


if __name__ == "__main__":
    main()
