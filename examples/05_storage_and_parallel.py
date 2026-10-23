"""05. 스토리지(Storage), 스터디 관리, 병렬/분산 최적화

다루는 기능
- RDBStorage (SQLite URL 문자열 / RDBStorage 객체, heartbeat)
- JournalStorage + JournalFileBackend (NFS 등 파일 기반 분산용)
- InMemoryStorage (기본)
- load_study / load_if_exists / copy_study / delete_study
- get_all_study_names / get_all_study_summaries
- study.set_user_attr / study.user_attrs
- n_jobs (스레드 병렬) / multiprocessing (프로세스 분산: 같은 스토리지 공유)
- heartbeat + RetryHeartbeatStaleTrialCallback (죽은 trial 자동 재시도, v4.9+ 권장 API)
- fail_stale_trials
- (참고) GrpcStorageProxy / run_grpc_proxy_server, JournalRedisBackend
"""
import multiprocessing as mp
import time
import warnings

from common import header, out, quadratic, remove, sqlite_url

import optuna
from optuna.storages import JournalStorage, RDBStorage, RetryHeartbeatStaleTrialCallback
from optuna.storages.journal import JournalFileBackend

DB = out("05_example.db")
SQLITE_URL = sqlite_url("05_example.db")
JOURNAL = out("05_journal.log")


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    return quadratic(x, y)


def slow_objective(trial: optuna.Trial) -> float:
    time.sleep(0.02)
    return objective(trial)


def worker(study_name: str, storage_kind: str, n_trials: int) -> None:
    """별도 프로세스에서 같은 스터디를 load 해서 최적화 (분산 최적화 패턴)."""
    import optuna as _o
    _o.logging.set_verbosity(_o.logging.WARNING)
    storage = SQLITE_URL if storage_kind == "rdb" else JournalStorage(JournalFileBackend(JOURNAL))
    study = _o.load_study(study_name=study_name, storage=storage)
    study.optimize(objective, n_trials=n_trials)


def main() -> None:
    remove(DB)
    remove(JOURNAL)

    header("05-1. SQLite(RDB) 스토리지에 저장 & 재개(resume)")
    study = optuna.create_study(study_name="rdb-study", storage=SQLITE_URL)
    study.set_user_attr("owner", "K")
    study.set_user_attr("dataset", "synthetic-quadratic")
    study.optimize(objective, n_trials=20)
    print("  1차 trial 수:", len(study.trials))
    # 같은 이름으로 다시 열기 → 이어서 최적화
    study = optuna.create_study(study_name="rdb-study", storage=SQLITE_URL, load_if_exists=True)
    study.optimize(objective, n_trials=10)
    print("  재개 후 trial 수:", len(study.trials), "| user_attrs:", study.user_attrs)

    header("05-2. 스터디 목록/요약/복사/삭제")
    optuna.copy_study(from_study_name="rdb-study", from_storage=SQLITE_URL,
                      to_storage=SQLITE_URL, to_study_name="rdb-study-copy")
    print("  study names:", optuna.get_all_study_names(storage=SQLITE_URL))
    for s in optuna.get_all_study_summaries(storage=SQLITE_URL):
        print(f"   - {s.study_name}: n_trials={s.n_trials}, best={s.best_trial.value:.4f}")
    optuna.delete_study(study_name="rdb-study-copy", storage=SQLITE_URL)
    print("  삭제 후:", optuna.get_all_study_names(storage=SQLITE_URL))

    header("05-3. JournalStorage (파일 로그 기반, 분산 환경 NFS 친화적)")
    jstorage = JournalStorage(JournalFileBackend(JOURNAL))
    jstudy = optuna.create_study(study_name="journal-study", storage=jstorage)
    jstudy.optimize(objective, n_trials=20)
    print("  journal trial 수:", len(jstudy.trials), "| 파일:", JOURNAL)

    header("05-4. n_jobs 스레드 병렬 (I/O·외부 프로세스 호출형 목적함수에 유리)")
    for n_jobs in (1, 4):
        st = optuna.create_study()
        t0 = time.time()
        st.optimize(slow_objective, n_trials=40, n_jobs=n_jobs)
        print(f"  n_jobs={n_jobs}: {time.time() - t0:.2f}s, best={st.best_value:.4f}")

    header("05-5. 멀티프로세스 분산 최적화 (RDB / Journal 공유)")
    for kind, name, storage in [("rdb", "mp-rdb", SQLITE_URL),
                                ("journal", "mp-journal", JournalStorage(JournalFileBackend(JOURNAL)))]:
        optuna.create_study(study_name=name, storage=storage)
        procs = [mp.Process(target=worker, args=(name, kind, 15)) for _ in range(3)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        st = optuna.load_study(study_name=name, storage=storage)
        print(f"  [{kind}] 3 프로세스 x 15 = {len(st.trials)} trials, best={st.best_value:.4f}")

    header("05-6. Heartbeat + RetryHeartbeatStaleTrialCallback (프로세스가 죽은 trial 자동 재시도)")
    storage = RDBStorage(
        url=SQLITE_URL,
        heartbeat_interval=1,   # 1초마다 생존 신호
        grace_period=2,         # 2초 동안 신호가 없으면 FAIL 처리
        heartbeat_stale_trial_callback=RetryHeartbeatStaleTrialCallback(max_retry=2),
    )
    st = optuna.create_study(study_name="heartbeat", storage=storage)
    # 'RUNNING 상태로 남은 죽은 trial' 을 인위적으로 만든다
    warnings.filterwarnings("ignore", message="Heartbeat of storage")
    dead = st.ask({"x": optuna.distributions.FloatDistribution(-10, 10),
                   "y": optuna.distributions.FloatDistribution(-10, 10)})
    storage.record_heartbeat(dead._trial_id)
    time.sleep(3)
    optuna.storages.fail_stale_trials(st)   # optimize() 시작 시에도 자동 호출됨
    st.optimize(objective, n_trials=3)
    for t in st.trials:
        print(f"  #{t.number} {t.state.name:8s} retried_from="
              f"{RetryHeartbeatStaleTrialCallback.retried_trial_number(t)}")

    print("\n  [참고] 다른 스토리지/프록시:")
    print("   - JournalRedisBackend('redis://localhost:6379') : Redis 기반 저널")
    print("   - run_grpc_proxy_server(storage, port=13000) + GrpcStorageProxy(host, port)")
    print("     : 다수 워커가 DB 에 직접 붙지 않고 gRPC 프록시를 통해 접근 (대규모 분산)")


if __name__ == "__main__":
    main()
