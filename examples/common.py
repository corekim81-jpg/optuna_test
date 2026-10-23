"""예제 공통 유틸리티: 출력 폴더, 벤치마크 함수, 헤더 출력."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import warnings

import optuna

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# 예제 출력이 너무 길어지지 않도록 Optuna 로그는 WARNING 이상만 표시
optuna.logging.set_verbosity(optuna.logging.WARNING)
# 실험적(Experimental) 기능 경고는 숨김 (기능 자체는 정상 동작)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


# Windows 콘솔(cp949)에서 특수문자 출력 시 UnicodeEncodeError 방지
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except AttributeError:
        pass


def out(name: str) -> str:
    """outputs/ 아래 파일 경로를 문자열로 반환."""
    return str(OUT_DIR / name)


def sqlite_url(name: str) -> str:
    """outputs/<name> SQLite URL (Windows 경로도 안전하도록 / 구분자 사용)."""
    return f"sqlite:///{(OUT_DIR / name).as_posix()}"


_SECTION = ""  # 마지막 header() 제목 → 대시보드 study 이름에 사용


def header(title: str) -> None:
    global _SECTION
    _SECTION = title
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def quadratic(x: float, y: float) -> float:
    """최소값 0 @ (2, -1)."""
    return (x - 2) ** 2 + (y + 1) ** 2


def rosenbrock(x: float, y: float) -> float:
    """최소값 0 @ (1, 1)."""
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def rastrigin(xs: list[float]) -> float:
    """다봉(multi-modal) 함수, 최소값 0 @ 0-벡터."""
    return 10 * len(xs) + sum(x * x - 10 * math.cos(2 * math.pi * x) for x in xs)


def remove(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


# ---------------------------------------------------------------------------
# Optuna Dashboard 연동
# ---------------------------------------------------------------------------
# 대시보드는 '스토리지'를 읽어서 보여주므로, 메모리(InMemoryStorage)에만 있는
# study 는 볼 수 없다. 환경변수 OPTUNA_EXAMPLES_DASHBOARD=1 이면(run_all.py --dashboard)
# storage 를 지정하지 않은 모든 create_study() 를 outputs/dashboard.log 에 저장한다.
# 스토리지는 JournalStorage(파일) 사용: 같은 예제 기준 SQLite 대비 약 7배 빠르고,
# 멀티프로세스에서도 안전하다. (Windows 호환을 위해 JournalFileOpenLock 사용)
DASHBOARD_LOG = OUT_DIR / "dashboard.log"
_dashboard_storage = None
ARTIFACT_DIR = OUT_DIR / "artifacts"  # 대시보드에서 미리보기할 artifact 저장 위치
DASHBOARD_MODE = os.environ.get("OPTUNA_EXAMPLES_DASHBOARD") == "1"
_SCRIPT = Path(sys.argv[0]).stem or "interactive"


def dashboard_storage() -> optuna.storages.BaseStorage:
    """대시보드용 공용 스토리지 (outputs/dashboard.log)."""
    global _dashboard_storage
    if _dashboard_storage is None:
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

        path = str(DASHBOARD_LOG)
        _dashboard_storage = JournalStorage(
            JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))
    return _dashboard_storage


def example_storage(default_name: str):
    """대시보드 모드면 공용 dashboard 스토리지, 아니면 outputs/<default_name> SQLite URL."""
    return dashboard_storage() if DASHBOARD_MODE else sqlite_url(default_name)


def _enable_dashboard_storage() -> None:
    original_create_study = optuna.create_study
    created_here: set[str] = set()

    def create_study(*args, **kwargs):
        if kwargs.get("storage") is not None:
            return original_create_study(*args, **kwargs)
        # 이름 예) "02_samplers/02-1 TPESampler", "04_multi_objective_constraints/04-1 NSGAIISampler"
        if kwargs.get("study_name"):
            label = kwargs["study_name"]
        else:
            sampler = kwargs.get("sampler")
            sec = _SECTION.split()[0].rstrip(".") if _SECTION else ""
            label = f"{sec} {type(sampler).__name__ if sampler else 'default'}".strip()
        base = f"{_SCRIPT}/{label}"
        name, i = base, 1
        while name in created_here:  # 같은 실행 안에서 이름이 겹치면 번호 부여
            i += 1
            name = f"{base}-{i}"
        created_here.add(name)
        try:  # 이전 실행에서 남은 같은 이름의 study 는 지우고 새로 만든다
            optuna.delete_study(study_name=name, storage=dashboard_storage())
        except KeyError:
            pass
        kwargs.update(storage=dashboard_storage(), study_name=name, load_if_exists=False)
        study = original_create_study(*args, **kwargs)
        if _SECTION:
            study.set_user_attr("example_section", _SECTION)  # 대시보드에서 어떤 예제인지 확인용
        return study

    optuna.create_study = create_study
    optuna.study.create_study = create_study


def fresh_dashboard_study(study_name: str, **kwargs) -> optuna.Study:
    """대시보드 스토리지에 study 를 새로 만든다 (같은 이름이 있으면 지우고 생성 → 재실행 가능)."""
    try:
        optuna.delete_study(study_name=study_name, storage=dashboard_storage())
    except KeyError:
        pass
    return optuna.create_study(storage=dashboard_storage(), study_name=study_name, **kwargs)


if DASHBOARD_MODE:
    _enable_dashboard_storage()
