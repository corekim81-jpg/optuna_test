"""12. 실전 ML 하이퍼파라미터 튜닝 (scikit-learn / LightGBM + optuna-integration)

다루는 기능
- 교차검증 기반 목적함수 + 모델 선택 (조건부 탐색 공간)
- LightGBM 수동 튜닝 + LightGBMPruningCallback (반복 단위 프루닝)
- optuna_integration.OptunaSearchCV (sklearn GridSearchCV 대체)
- optuna_integration.lightgbm.LightGBMTunerCV (단계적 자동 튜닝)
- best 파라미터로 최종 모델 재학습 & 테스트 평가

필요 패키지: pip install optuna-integration lightgbm scikit-learn
"""
import warnings

import numpy as np
from common import header
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import optuna

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*terminator.*")  # OptunaSearchCV 내부 경고

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
CV = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)


def build_model(params: dict):
    kind = params["classifier"]
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=params["lr_C"], max_iter=2000))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=params["rf_n_estimators"],
                                      max_depth=params["rf_max_depth"], random_state=0, n_jobs=-1)
    return GradientBoostingClassifier(learning_rate=params["gb_lr"],
                                      n_estimators=params["gb_n_estimators"], random_state=0)


def model_selection_objective(trial: optuna.Trial) -> float:
    kind = trial.suggest_categorical("classifier", ["logreg", "rf", "gb"])
    if kind == "logreg":
        trial.suggest_float("lr_C", 1e-3, 1e2, log=True)
    elif kind == "rf":
        trial.suggest_int("rf_n_estimators", 50, 300, step=50)
        trial.suggest_int("rf_max_depth", 2, 16, log=True)
    else:
        trial.suggest_float("gb_lr", 1e-3, 0.3, log=True)
        trial.suggest_int("gb_n_estimators", 50, 200, step=50)
    model = build_model(trial.params)
    return float(cross_val_score(model, X_tr, y_tr, cv=CV, scoring="roc_auc").mean())


def main() -> None:
    header("12-1. 모델 선택 + 하이퍼파라미터 동시 탐색 (CV ROC-AUC)")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(model_selection_objective, n_trials=30)
    print("  best CV AUC:", round(study.best_value, 4), "|", study.best_params)
    final = build_model(study.best_params).fit(X_tr, y_tr)
    print("  test AUC   :", round(roc_auc_score(y_te, final.predict_proba(X_te)[:, 1]), 4))

    try:
        import lightgbm as lgb
        from optuna_integration import LightGBMPruningCallback, OptunaSearchCV
        from optuna_integration.lightgbm import LightGBMTunerCV
    except ImportError as e:
        print("\n  optuna-integration / lightgbm 미설치 → 12-2~12-4 건너뜀:", e)
        return

    header("12-2. LightGBM + LightGBMPruningCallback")
    # min_child_samples 를 탐색하려면 feature_pre_filter=False 필요
    dtrain = lgb.Dataset(X_tr, label=y_tr, params={"feature_pre_filter": False})
    dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)

    def lgb_objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary", "metric": "auc", "verbosity": -1, "seed": 0,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 128, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        booster = lgb.train(params, dtrain, num_boost_round=300, valid_sets=[dvalid],
                            callbacks=[LightGBMPruningCallback(trial, "auc"),
                                       lgb.early_stopping(30, verbose=False)])
        return roc_auc_score(y_te, booster.predict(X_te))

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20))
    st.optimize(lgb_objective, n_trials=30)
    pruned = len(st.get_trials(states=(optuna.trial.TrialState.PRUNED,)))
    print(f"  best AUC={st.best_value:.4f}, pruned={pruned}/{len(st.trials)}")

    header("12-3. OptunaSearchCV (sklearn 호환 API)")
    search = OptunaSearchCV(
        RandomForestClassifier(random_state=0),
        {
            "n_estimators": optuna.distributions.IntDistribution(50, 300, step=50),
            "max_depth": optuna.distributions.IntDistribution(2, 16, log=True),
            "max_features": optuna.distributions.CategoricalDistribution(["sqrt", "log2", None]),
        },
        cv=CV, n_trials=15, scoring="accuracy", random_state=0, verbose=0,
    )
    search.fit(X_tr, y_tr)
    print("  best_params_:", search.best_params_, "| best_score_:", round(search.best_score_, 4))
    print("  test acc   :", round(accuracy_score(y_te, search.predict(X_te)), 4))

    header("12-4. LightGBMTunerCV (feature_fraction → num_leaves → bagging → ... 단계적 튜닝)")
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    tuner = LightGBMTunerCV(
        {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 0},
        lgb.Dataset(X_tr, label=y_tr), folds=CV, num_boost_round=100,
        callbacks=[lgb.early_stopping(20, verbose=False)],
        optuna_seed=0, show_progress_bar=False, time_budget=60,
    )
    tuner.run()
    print("  best logloss:", round(tuner.best_score, 4))
    print("  best params :", {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in tuner.best_params.items()})
    print("  탐색 trial 수:", len(tuner.study.trials), "| 평균 trial 시간:",
          np.round(np.mean([t.duration.total_seconds() for t in tuner.study.trials]), 3), "s")


if __name__ == "__main__":
    main()
