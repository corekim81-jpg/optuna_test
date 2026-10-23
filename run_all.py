"""examples/ 의 모든 예제를 순서대로 실행하고 결과(성공/실패, 소요시간)를 요약한다.

사용법
    python run_all.py              # 전체 실행
    python run_all.py 02 05 08     # 번호로 일부만 실행
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXAMPLES = sorted(p for p in (ROOT / "examples").glob("[0-9][0-9]_*.py"))


def main() -> int:
    selected = sys.argv[1:]
    targets = [p for p in EXAMPLES if not selected or p.name[:2] in selected]
    results = []
    for path in targets:
        print(f"\n\n########## {path.name} ##########", flush=True)
        t0 = time.time()
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        rc = subprocess.call([sys.executable, path.name], cwd=path.parent, env=env)
        results.append((path.name, rc, time.time() - t0))

    print("\n\n" + "=" * 60 + "\n  요약\n" + "=" * 60)
    for name, rc, sec in results:
        print(f"  {'OK  ' if rc == 0 else 'FAIL'}  {name:40s} {sec:6.1f}s")
    failed = [r for r in results if r[1] != 0]
    print(f"\n  {len(results) - len(failed)}/{len(results)} 성공")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
