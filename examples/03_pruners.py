"""03. 프루너(Pruner) 전체 비교 - 성능이 나쁜 trial 조기 종료

다루는 기능
- trial.report(value, step) / trial.should_prune() / raise optuna.TrialPruned
- MedianPruner, PercentilePruner, SuccessiveHalvingPruner, HyperbandPruner
- ThresholdPruner, PatientPruner(래퍼), WilcoxonPruner, NopPruner
- 실제 학습(SGDClassifier.partial_fit)에서 epoch 단위 프루닝
"""
import numpy as np
from common import header
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

import optuna
from optuna.pruners import (
    HyperbandPruner,
    MedianPruner,
    NopPruner,
    PatientPruner,
    PercentilePruner,
    SuccessiveHalvingPruner,
    ThresholdPruner,
    WilcoxonPruner,
)
from optuna.trial import TrialState

X, y = load_digits(return_X_y=True)
X = X / 16.0
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, random_state=0)
CLASSES = np.unique(y)
N_EPOCHS = 20


def sgd_objective(trial: optuna.Trial) -> float:
    alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
    loss = trial.suggest_categorical("loss", ["hinge", "log_loss", "modified_huber"])
    clf = SGDClassifier(alpha=alpha, loss=loss, random_state=0)
    acc = 0.0
    for epoch in range(N_EPOCHS):
        clf.partial_fit(X_tr, y_tr, classes=CLASSES)
        acc = clf.score(X_va, y_va)
        trial.report(acc, epoch)        # 중간값 보고
        if trial.should_prune():        # 프루너에게 중단 여부 질의
            raise optuna.TrialPruned()
    return acc


def summarize(name: str, study: optuna.Study) -> None:
    pruned = len(study.get_trials(states=(TrialState.PRUNED,)))
    complete = len(study.get_trials(states=(TrialState.COMPLETE,)))
    print(f"  {name:40s} best={study.best_value:.4f} complete={complete:3d} pruned={pruned:3d}")


def run(name: str, pruner: optuna.pruners.BasePruner, n_trials: int = 40) -> None:
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0), pruner=pruner)
    study.optimize(sgd_objective, n_trials=n_trials)
    summarize(name, study)


def wilcoxon_demo() -> None:
    """WilcoxonPruner: '여러 인스턴스(fold/문제)' 평가 시, 현재 best 대비 통계적으로
    유의하게 나쁘면 중단. step 은 인스턴스 ID 로 사용한다."""
    rng = np.random.default_rng(0)
    instances = rng.normal(size=(30, 2))  # 30개의 서로 다른 '문제 인스턴스'

    def obj(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -3, 3)
        y = trial.suggest_float("y", -3, 3)
        order = rng.permutation(len(instances))
        losses = []
        for i in order:
            loss = (x - instances[i, 0]) ** 2 + (y - instances[i, 1]) ** 2
            losses.append(loss)
            trial.report(loss, int(i))
            if trial.should_prune():
                return float(np.mean(losses))  # Wilcoxon 은 return 으로 종료하는 것도 권장 방식
        return float(np.mean(losses))

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0),
                                pruner=WilcoxonPruner(p_threshold=0.1, n_startup_steps=5))
    study.optimize(obj, n_trials=40)
    n_steps = [len(t.intermediate_values) for t in study.trials]
    print(f"  WilcoxonPruner best={study.best_value:.4f}, 평균 평가 인스턴스 수={np.mean(n_steps):.1f}/30")


def main() -> None:
    header("03-1. 프루너별 비교 (SGDClassifier on digits, 20 epochs)")
    run("NopPruner (프루닝 없음)", NopPruner())
    run("MedianPruner", MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    run("PercentilePruner(25%)", PercentilePruner(25.0, n_startup_trials=5, n_warmup_steps=3))
    run("SuccessiveHalvingPruner", SuccessiveHalvingPruner(min_resource=1, reduction_factor=3))
    run("HyperbandPruner", HyperbandPruner(min_resource=1, max_resource=N_EPOCHS, reduction_factor=3))
    run("ThresholdPruner(lower=0.93)", ThresholdPruner(lower=0.93, n_warmup_steps=3))
    run("PatientPruner(Median, patience=3)",
        PatientPruner(MedianPruner(n_startup_trials=5), patience=3))

    header("03-2. WilcoxonPruner (인스턴스/폴드 단위 평가)")
    wilcoxon_demo()

    header("03-3. 중간값(intermediate_values) 확인")
    study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_startup_trials=3))
    study.optimize(sgd_objective, n_trials=10)
    for t in study.trials[:5]:
        print(f"  #{t.number} {t.state.name:8s} 마지막 step={t.last_step} "
              f"values={[round(v, 3) for v in list(t.intermediate_values.values())[:5]]}...")


if __name__ == "__main__":
    main()
