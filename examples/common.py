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


def header(title: str) -> None:
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
