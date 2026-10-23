"""09. 하이퍼파라미터 중요도 (Parameter Importance)

다루는 기능
- optuna.importance.get_param_importances
- FanovaImportanceEvaluator (기본 v3 이전), PedAnovaImportanceEvaluator (v5 기본),
  MeanDecreaseImpurityImportanceEvaluator
- params= 로 일부 파라미터만 평가, target= 으로 평가 대상 변경, normalize=False
"""
from common import header

import optuna
from optuna.importance import (
    FanovaImportanceEvaluator,
    MeanDecreaseImpurityImportanceEvaluator,
    PedAnovaImportanceEvaluator,
    get_param_importances,
)


def objective(trial: optuna.Trial) -> float:
    # x 가 가장 중요, y 는 중간, z 는 거의 무관, noise 는 완전 무관하게 설계
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    z = trial.suggest_float("z", -10, 10)
    trial.suggest_float("noise", -10, 10)
    kind = trial.suggest_categorical("kind", ["good", "bad"])
    trial.set_user_attr("cost", abs(z) * 3 + abs(y))
    return 10 * x**2 + y**2 + 0.01 * z**2 + (0 if kind == "good" else 50)


def show(title: str, imp: dict[str, float]) -> None:
    print(f"  [{title}]")
    for k, v in imp.items():
        print(f"     {k:6s} {v:7.4f} " + "#" * int(v * 40))


def main() -> None:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    study.optimize(objective, n_trials=300)

    header("09-1. 평가기(evaluator)별 중요도")
    show("default (PED-ANOVA)", get_param_importances(study))
    show("fANOVA", get_param_importances(study, evaluator=FanovaImportanceEvaluator(seed=0)))
    show("MDI (RandomForest)",
         get_param_importances(study, evaluator=MeanDecreaseImpurityImportanceEvaluator(seed=0)))
    show("PED-ANOVA (top 10%)",
         get_param_importances(study, evaluator=PedAnovaImportanceEvaluator(target_quantile=0.1)))

    header("09-2. 일부 파라미터만 / 정규화 끄기")
    show("params=[x, y] normalize=False",
         get_param_importances(study, params=["x", "y"], normalize=False,
                               evaluator=FanovaImportanceEvaluator(seed=0)))

    header("09-3. target 변경: 목적값 대신 user_attr 'cost' 기준")
    show("target=cost",
         get_param_importances(study, target=lambda t: t.user_attrs["cost"],
                               evaluator=FanovaImportanceEvaluator(seed=0)))


if __name__ == "__main__":
    main()
