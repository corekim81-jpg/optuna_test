"""10. Terminator - 최적화 자동 종료 (더 돌려도 개선 여지가 작을 때 중단)

다루는 기능
- optuna.terminator.TerminatorCallback / Terminator
- report_cross_validation_scores (CV 점수 분산을 '오차'로 사용)
- Improvement evaluators: RegretBoundEvaluator(기본), EMMREvaluator, BestValueStagnationEvaluator
- Error evaluators: CrossValidationErrorEvaluator(기본), StaticErrorEvaluator, MedianErrorEvaluator
- plot_terminator_improvement

※ optuna 4.9 부터 optuna.terminator 모듈은 deprecated (v6.0 에서 제거 예정, OptunaHub 로 이전).
  v5.0 에서는 아직 정상 동작하므로 기능 확인용으로 포함.
"""
import warnings

from common import header, out
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

import optuna
from optuna.terminator import (
    BestValueStagnationEvaluator,
    EMMREvaluator,
    MedianErrorEvaluator,
    RegretBoundEvaluator,
    StaticErrorEvaluator,
    Terminator,
    TerminatorCallback,
    report_cross_validation_scores,
)

warnings.filterwarnings("ignore", category=FutureWarning, message=".*terminator.*")
X, y = load_wine(return_X_y=True)


def objective(trial: optuna.Trial) -> float:
    clf = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 10, 100),
        max_depth=trial.suggest_int("max_depth", 1, 10),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        random_state=0,
    )
    scores = cross_val_score(clf, X, y, cv=KFold(5, shuffle=True, random_state=0))
    report_cross_validation_scores(trial, scores.tolist())  # Terminator 가 사용할 CV 점수
    return float(scores.mean())


def main() -> None:
    header("10-1. 기본 TerminatorCallback (RegretBound + CV 오차)")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=60, callbacks=[TerminatorCallback()])
    print(f"  최대 60 중 {len(study.trials)} trials 에서 종료, best={study.best_value:.4f}")

    fig = optuna.visualization.plot_terminator_improvement(study, plot_error=True)
    fig.write_html(out("10_terminator_improvement.html"))
    print("  저장:", out("10_terminator_improvement.html"))

    header("10-2. 여러 Improvement/Error evaluator 조합")
    combos = {
        "BestValueStagnation(10) + Static(0)": Terminator(
            improvement_evaluator=BestValueStagnationEvaluator(max_stagnation_trials=10),
            error_evaluator=StaticErrorEvaluator(constant=0)),
        "EMMR + Median": Terminator(
            improvement_evaluator=EMMREvaluator(seed=0),
            error_evaluator=MedianErrorEvaluator(paired_improvement_evaluator=EMMREvaluator(seed=0)),
            min_n_trials=10),
        "RegretBound + Static(0.01)": Terminator(
            improvement_evaluator=RegretBoundEvaluator(seed=0),
            error_evaluator=StaticErrorEvaluator(constant=0.01), min_n_trials=10),
    }
    for name, term in combos.items():
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        st.optimize(objective, n_trials=60, callbacks=[TerminatorCallback(term)])
        print(f"  {name:40s} → {len(st.trials):2d} trials, best={st.best_value:.4f}")

    header("10-3. 콜백 없이 수동으로 종료 여부 확인 (ask-tell 루프 등)")
    term = Terminator(improvement_evaluator=BestValueStagnationEvaluator(5),
                      error_evaluator=StaticErrorEvaluator(0))
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    for i in range(100):
        st.optimize(objective, n_trials=1)
        if term.should_terminate(st):
            print(f"  should_terminate=True @ {i + 1} trials")
            break


if __name__ == "__main__":
    main()
