"""Optuna Dashboard 실행기.

outputs/dashboard.log (모든 예제 study, JournalStorage) 와 outputs/artifacts (이미지/CSV/모델 artifact) 를
함께 연결해서 대시보드를 띄운다.

사용법
    python run_all.py --dashboard   # 1) 모든 예제를 outputs/dashboard.log 에 저장하며 실행
    python dashboard.py             # 2) http://127.0.0.1:8080 접속

    python dashboard.py --port 8081 --host 0.0.0.0   # 포트/외부 접속 허용

CLI 로 직접 띄우려면:
    optuna-dashboard outputs/dashboard.log --artifact-dir outputs/artifacts
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "examples"))
from common import ARTIFACT_DIR, DASHBOARD_LOG, dashboard_storage  # noqa: E402

import optuna  # noqa: E402
from optuna.artifacts import FileSystemArtifactStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    try:
        from optuna_dashboard import run_server
    except ImportError:
        sys.exit("optuna-dashboard 미설치: pip install optuna-dashboard")

    if not DASHBOARD_LOG.exists():
        sys.exit("outputs/dashboard.log 가 없습니다. 먼저 `python run_all.py --dashboard` 를 실행하세요.")

    names = optuna.get_all_study_names(storage=dashboard_storage())
    print(f"study {len(names)}개 로드:")
    for n in names:
        print("  -", n)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n→ http://{args.host}:{args.port}  (종료: Ctrl+C)")
    run_server(dashboard_storage(), host=args.host, port=args.port,
               artifact_store=FileSystemArtifactStore(str(ARTIFACT_DIR)))


if __name__ == "__main__":
    main()
