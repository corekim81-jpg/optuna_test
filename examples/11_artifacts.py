"""11. Artifact 저장소 - trial 별 모델/그림/로그 파일 관리

다루는 기능
- FileSystemArtifactStore (로컬), (참고) Boto3ArtifactStore(S3), GCSArtifactStore
- upload_artifact(study_or_trial=trial|study)
- get_all_artifact_meta / download_artifact
- Backoff (재시도 래퍼)
- best trial 의 모델 파일만 다시 내려받아 재사용하는 패턴
"""
import json
import os
import pickle
import shutil

from common import ARTIFACT_DIR, DASHBOARD_MODE, OUT_DIR, fresh_dashboard_study, header, out, example_storage
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

import optuna
from optuna.artifacts import (
    Backoff,
    FileSystemArtifactStore,
    download_artifact,
    get_all_artifact_meta,
    upload_artifact,
)

ART_DIR = ARTIFACT_DIR  # dashboard.py 와 공유 → 대시보드에서 artifact 미리보기 가능
TMP_DIR = OUT_DIR / "11_tmp"
X, y = load_iris(return_X_y=True)


def main() -> None:
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True)
    # Backoff 로 감싸면 일시적 I/O 오류 시 지수 백오프 재시도
    store = Backoff(FileSystemArtifactStore(base_path=str(ART_DIR)), max_retries=3)

    def objective(trial: optuna.Trial) -> float:
        c = trial.suggest_float("C", 1e-3, 1e2, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 1e0, log=True)
        clf = SVC(C=c, gamma=gamma)
        score = float(cross_val_score(clf, X, y, cv=3).mean())
        clf.fit(X, y)
        # 1) 모델 파일 업로드
        model_path = TMP_DIR / f"model_{trial.number}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        aid = upload_artifact(artifact_store=store, file_path=str(model_path), study_or_trial=trial)
        trial.set_user_attr("model_artifact_id", aid)
        # 2) 메타데이터 JSON 업로드 (mimetype 지정)
        meta_path = TMP_DIR / f"meta_{trial.number}.json"
        meta_path.write_text(json.dumps({"C": c, "gamma": gamma, "cv": score}))
        upload_artifact(artifact_store=store, file_path=str(meta_path), study_or_trial=trial,
                        mimetype="application/json")
        os.remove(model_path)
        os.remove(meta_path)
        return score

    header("11-1. trial 마다 모델/메타데이터 artifact 업로드")
    sampler = optuna.samplers.TPESampler(seed=0)
    if DASHBOARD_MODE:  # 대시보드 모드: 공용 대시보드 스토리지에 저장
        study = fresh_dashboard_study("11_artifacts/artifact-demo", direction="maximize", sampler=sampler)
    else:
        db = out("11_artifacts.db")
        if os.path.exists(db):
            os.remove(db)
        study = optuna.create_study(direction="maximize", storage=example_storage("11_artifacts.db"),
                                    study_name="artifact-demo", sampler=sampler)
    study.optimize(objective, n_trials=15)
    print("  best:", round(study.best_value, 4), study.best_params)
    print("  artifact 저장 위치:", ART_DIR)

    header("11-2. Study 레벨 artifact (요약 리포트)")
    report = TMP_DIR / "study_report.txt"
    report.write_text(f"best={study.best_value}\nparams={study.best_params}\n", encoding="utf-8")
    upload_artifact(artifact_store=store, file_path=str(report), study_or_trial=study)
    for m in get_all_artifact_meta(study):
        print("  study artifact:", m.filename, m.mimetype, m.artifact_id[:8])

    header("11-3. best trial 의 artifact 목록 조회 & 모델 다운로드 후 재사용")
    best = study.best_trial
    # FrozenTrial 은 storage 정보를 갖지 않으므로 storage 를 명시
    storage = optuna.storages.get_storage(example_storage("11_artifacts.db"))
    for m in get_all_artifact_meta(best, storage=storage):
        print(f"  {m.filename:16s} {m.mimetype:28s} id={m.artifact_id[:8]}...")
    dl_path = TMP_DIR / "best_model.pkl"
    download_artifact(artifact_store=store, file_path=str(dl_path),
                      artifact_id=best.user_attrs["model_artifact_id"])
    with open(dl_path, "rb") as f:
        model = pickle.load(f)
    print("  다운로드한 모델 정확도(train):", round(model.score(X, y), 4))

    print("\n  [참고] 클라우드 저장소:")
    print("   Boto3ArtifactStore(bucket_name='my-bucket')   # S3 / MinIO (boto3 필요)")
    print("   GCSArtifactStore(bucket_name='my-bucket')     # Google Cloud Storage")
    print("   `python run_all.py --dashboard` 후 `python dashboard.py` 로 artifact 미리보기/다운로드 가능")


if __name__ == "__main__":
    main()
