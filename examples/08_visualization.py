"""08. 시각화 (plotly / matplotlib 백엔드 전체)

다루는 기능 (optuna.visualization.* → HTML, optuna.visualization.matplotlib.* → PNG)
- plot_optimization_history   (여러 study 비교, error_bar)
- plot_intermediate_values     (프루닝/학습곡선)
- plot_parallel_coordinate
- plot_contour
- plot_slice
- plot_param_importances
- plot_edf                     (Empirical Distribution Function, 여러 study 비교)
- plot_rank
- plot_timeline
- plot_pareto_front / plot_hypervolume_history  (→ 04 예제)
- plot_terminator_improvement                    (→ 10 예제)
"""
import warnings

import matplotlib

matplotlib.use("Agg")  # GUI 없는 환경에서 PNG 저장
import matplotlib.pyplot as plt
from common import header, out

import optuna
import optuna.visualization as vis
import optuna.visualization.matplotlib as vis_mpl


warnings.filterwarnings("ignore", message="Tight layout not applied")


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    z = trial.suggest_int("z", 0, 10)
    c = trial.suggest_categorical("c", ["a", "b", "c"])
    loss = (x - 1) ** 2 + 0.5 * (y + 2) ** 2 + 0.01 * (z - 5) ** 2 + {"a": 0, "b": 1, "c": 2}[c]
    for step in range(10):  # 학습 곡선 흉내 (intermediate values)
        trial.report(loss * (1 + 2.0 / (step + 1)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return loss


def make_study(sampler: optuna.samplers.BaseSampler, name: str) -> optuna.Study:
    st = optuna.create_study(study_name=name, sampler=sampler,
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    st.optimize(objective, n_trials=80)
    return st


def main() -> None:
    header("08-1. 스터디 생성")
    tpe = make_study(optuna.samplers.TPESampler(seed=0), "TPE")
    rnd = make_study(optuna.samplers.RandomSampler(seed=0), "Random")
    print("  TPE best:", round(tpe.best_value, 4), "| Random best:", round(rnd.best_value, 4))
    print("  plotly 사용 가능:", vis.is_available(), "| matplotlib 사용 가능:", vis_mpl.is_available())

    header("08-2. Plotly (interactive HTML)")
    figs = {
        "optimization_history": vis.plot_optimization_history(tpe),
        "optimization_history_compare": vis.plot_optimization_history([tpe, rnd]),
        "intermediate_values": vis.plot_intermediate_values(tpe),
        "parallel_coordinate": vis.plot_parallel_coordinate(tpe, params=["x", "y", "c"]),
        "contour": vis.plot_contour(tpe, params=["x", "y"]),
        "contour_all": vis.plot_contour(tpe),
        "slice": vis.plot_slice(tpe),
        "param_importances": vis.plot_param_importances(tpe),
        "edf_compare": vis.plot_edf([tpe, rnd]),
        "rank": vis.plot_rank(tpe, params=["x", "y", "z"]),
        "timeline": vis.plot_timeline(tpe),
        # 사용자 정의 target: trial 소요 시간 기준 중요도
        "importance_duration": vis.plot_param_importances(
            tpe, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0),
            target=lambda t: t.duration.total_seconds(), target_name="duration"),
    }
    for name, fig in figs.items():
        path = out(f"08_{name}.html")
        fig.write_html(path)
        print("  저장:", path)

    header("08-3. 여러 반복 실험의 평균±표준편차 (error_bar=True)")
    studies = [make_study(optuna.samplers.TPESampler(seed=s), f"tpe-{s}") for s in range(3)]
    vis.plot_optimization_history(studies, error_bar=True).write_html(out("08_history_error_bar.html"))
    print("  저장:", out("08_history_error_bar.html"))

    header("08-4. Matplotlib 백엔드 (PNG)")
    mpl_funcs = {
        "optimization_history": lambda: vis_mpl.plot_optimization_history(tpe),
        "intermediate_values": lambda: vis_mpl.plot_intermediate_values(tpe),
        "parallel_coordinate": lambda: vis_mpl.plot_parallel_coordinate(tpe, params=["x", "y"]),
        "contour": lambda: vis_mpl.plot_contour(tpe, params=["x", "y"]),
        "slice": lambda: vis_mpl.plot_slice(tpe),
        "param_importances": lambda: vis_mpl.plot_param_importances(tpe),
        "edf": lambda: vis_mpl.plot_edf([tpe, rnd]),
        "rank": lambda: vis_mpl.plot_rank(tpe, params=["x", "y"]),
        "timeline": lambda: vis_mpl.plot_timeline(tpe),
    }
    for name, fn in mpl_funcs.items():
        ax = fn()
        fig = ax.flat[0].figure if hasattr(ax, "flat") else ax.figure
        path = out(f"08_mpl_{name}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close("all")
        print("  저장:", path)


if __name__ == "__main__":
    main()
