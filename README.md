# Optuna 기능 테스트 예제 모음

Optuna(**v5.0.0** 기준)의 기능 전반을 기능별 스크립트로 나눠 테스트하는 예제입니다.
각 스크립트는 독립 실행되며, 결과 파일(CSV, HTML/PNG 그래프, SQLite DB, artifact)은 `outputs/` 에 생성됩니다.

## 설치 & 실행

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   /   Linux·macOS: source .venv/bin/activate
pip install -r requirements.txt

python run_all.py            # 전체 실행 + 요약 (약 2~4분)
python run_all.py 02 08      # 번호로 일부만 실행
cd examples && python 01_basic_quickstart.py   # 개별 실행
```

> `torch`(GPSampler), `optuna-integration`/`lightgbm`(12), `optunahub`(15) 가 없으면 해당 부분만 건너뜁니다.
> 15번(OptunaHub)은 실행 시 GitHub 에서 패키지를 내려받으므로 인터넷 연결이 필요합니다.

## 예제 구성

| # | 파일 | 다루는 기능 |
|---|------|-------------|
| 01 | `01_basic_quickstart.py` | `create_study`, `optimize`, `best_*`, `suggest_float/int/categorical` (log, step), 조건부·동적 탐색 공간(Define-by-Run), `trials_dataframe`, maximize |
| 02 | `02_samplers.py` | TPE(multivariate, group, n_startup_trials), Random, CMA-ES(popsize, separable, with_margin), **GPSampler**, QMC(Sobol/Halton), Grid, BruteForce, NSGA-II/III, PartialFixed, 샘플러 pickle |
| 03 | `03_pruners.py` | `report`/`should_prune`/`TrialPruned`, Median, Percentile, SuccessiveHalving, Hyperband, Threshold, Patient, **Wilcoxon**, Nop |
| 04 | `04_multi_objective_constraints.py` | 다목적(`directions`, `best_trials`, `set_metric_names`), **`trial.set_constraint`(v5 제약조건 API)**, NSGA-II crossover 6종, 3목적, 하이퍼볼륨, 파레토 시각화 |
| 05 | `05_storage_and_parallel.py` | RDB(SQLite), JournalStorage(File), resume/`load_study`/`copy_study`/`delete_study`/summaries, `n_jobs`, 멀티프로세스 분산, heartbeat + `RetryHeartbeatStaleTrialCallback`, gRPC 프록시 안내 |
| 06 | `06_ask_tell_and_trials.py` | Ask-and-Tell, define-and-run(`ask(fixed_distributions)`), 배치 평가, tell 상태(PRUNED/FAIL), `enqueue_trial`, `add_trial(s)`+`create_trial`, `FixedTrial` |
| 07 | `07_callbacks_and_control.py` | 사용자 콜백(조기 종료), `MaxTrialsCallback`, `study.stop()`, `timeout`, `catch`, user_attrs, `gc_after_trial`, `show_progress_bar`, 로깅 제어 |
| 08 | `08_visualization.py` | plotly/matplotlib 전체 plot: history(비교, error_bar), intermediate, parallel_coordinate, contour, slice, importances, edf, rank, timeline |
| 09 | `09_param_importance.py` | PED-ANOVA(기본), fANOVA, MDI 중요도, `params`/`target`/`normalize` 옵션 |
| 10 | `10_terminator.py` | Terminator(RegretBound, EMMR, BestValueStagnation / CV, Static, Median error), `report_cross_validation_scores`, `plot_terminator_improvement` *(v4.9 부터 deprecated)* |
| 11 | `11_artifacts.py` | `FileSystemArtifactStore`, `Backoff`, `upload_artifact`(trial/study), `get_all_artifact_meta`, `download_artifact` |
| 12 | `12_ml_sklearn_lightgbm.py` | 실전 튜닝: 모델 선택+CV, LightGBM + `LightGBMPruningCallback`, `OptunaSearchCV`, `LightGBMTunerCV` |
| 13 | `13_distributions_search_space.py` | Distribution 객체/JSON 직렬화/호환성, internal·external 표현, `IntersectionSearchSpace`, trial 필터링, multi-index DataFrame |
| 14 | `14_custom_sampler_pruner.py` | `BaseSampler` 상속(Simulated Annealing), `BasePruner` 상속 |
| 15 | `15_optunahub.py` | `optunahub.load_module("samplers/auto_sampler")` — AutoSampler (단일/다목적) |
| 16 | `16_cli_and_dashboard.py` | `optuna` CLI(create-study, ask/tell, studies, trials, best-trial, set-user-attr), 대시보드용 DB 생성 |

## Optuna Dashboard

```bash
python examples/16_cli_and_dashboard.py
optuna-dashboard sqlite:///outputs/16_dashboard.db
# → http://127.0.0.1:8080
```

## v5.0 에서 확인된 변경/주의 사항

예제 작성 중 실제로 경고가 발생한 항목이며, 예제는 모두 새 방식으로 작성했습니다.

| 기존 방식 | v5 권장 방식 |
|-----------|--------------|
| `Sampler(constraints_func=...)` (v5.0 deprecated) | 목적함수 안에서 `trial.set_constraint("name", value)` |
| `RDBStorage(failed_trial_callback=RetryFailedTrialCallback())` (v4.9 deprecated) | `heartbeat_stale_trial_callback=RetryHeartbeatStaleTrialCallback()` |
| `CmaEsSampler(restart_strategy="ipop")` (v4.4 deprecated) | OptunaHub 의 CMA-ES 패키지 |
| `optuna.terminator` (v4.9 deprecated, v6.0 제거 예정) | 아직 동작함 — 향후 OptunaHub 로 이전 |
| 중요도 기본 평가기 fANOVA | PED-ANOVA (`PedAnovaImportanceEvaluator`) |
| `TPESampler(constant_liar=False)` 기본값 | `constant_liar=True` 기본값 (병렬 최적화에 유리) |

## 폴더 구조

```
Optuna_test/
├── README.md
├── requirements.txt
├── run_all.py            # 전체 실행 + 요약
├── examples/
│   ├── common.py         # 공통 유틸(출력 경로, 벤치마크 함수)
│   └── 01_ ~ 16_*.py
└── outputs/              # 실행 결과 (git 제외)
```
