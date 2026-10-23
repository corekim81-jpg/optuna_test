"""17. Optuna Dashboard 전용 기능

실행 후 `python dashboard.py` 로 대시보드를 띄우면 아래 study 들을 브라우저에서 확인할 수 있다.
(모든 study 는 outputs/dashboard.log, artifact 는 outputs/artifacts/ 에 저장)

다루는 기능 (pip install optuna-dashboard)
- save_note             : study / trial 에 Markdown 노트 기록 (대시보드에서 보기·편집)
- save_plotly_graph_object : 사용자 정의 plotly 그래프를 study 에 첨부
- study.set_metric_names : 목적함수 이름 표시 (od.set_objective_names 는 deprecated)
- artifact 연동          : trial 별 이미지/CSV 를 대시보드에서 미리보기·다운로드
- Human-in-the-loop     : register_objective_form_widgets (ChoiceWidget / SliderWidget /
                          TextInputWidget / ObjectiveUserAttrRef) → 사람이 대시보드에서 평가값 입력
- register_user_attr_form_widgets : 완료된 trial 에 사람이 추가 정보(user_attr) 입력
- Preferential optimization : optuna_dashboard.preferential (여러 후보 중 사람이 '가장 나쁜 것'을 골라 선호도 학습)
"""
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from common import ARTIFACT_DIR, OUT_DIR, dashboard_storage, fresh_dashboard_study, header, quadratic

import optuna
from optuna.artifacts import FileSystemArtifactStore, upload_artifact

try:
    import optuna_dashboard as od
    from optuna_dashboard import preferential
except ImportError:  # pragma: no cover
    od = None

PREFIX = "17_dashboard"
TMP = OUT_DIR / "17_tmp"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
STORE = FileSystemArtifactStore(str(ARTIFACT_DIR))  # dashboard.py 와 같은 위치를 사용해야 미리보기 가능


def color_image(trial: optuna.Trial, r: float, g: float, b: float, name: str) -> str:
    """파라미터(RGB)로 그림을 그려 artifact 로 업로드하고 artifact_id 를 반환."""
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.add_patch(plt.Circle((0.5, 0.5), 0.4, color=(r, g, b)))
    ax.set_axis_off()
    path = TMP / f"{name}_{trial.number}.png"
    fig.savefig(path, dpi=60)
    plt.close(fig)
    return upload_artifact(artifact_store=STORE, file_path=str(path), study_or_trial=trial)


def main() -> None:
    if od is None:
        print("optuna-dashboard 미설치: pip install optuna-dashboard")
        return
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)

    # ------------------------------------------------------------------
    header("17-1. 노트 + 사용자 정의 그래프 + 목적함수 이름 + artifact")
    study = fresh_dashboard_study(f"{PREFIX}/notes-graphs-artifacts",
                                  sampler=optuna.samplers.TPESampler(seed=0))
    study.set_metric_names(["loss"])
    od.save_note(study, "# 실험 노트\n\n- 목적: quadratic 함수 최소화\n- 담당: K\n\n"
                        "대시보드의 **Note** 영역에서 직접 수정할 수도 있습니다.")

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -5, 5)
        y = trial.suggest_float("y", -5, 5)
        loss = quadratic(x, y)
        od.save_note(trial, f"## Trial {trial.number}\n\n| x | y | loss |\n|---|---|---|\n"
                            f"| {x:.3f} | {y:.3f} | {loss:.4f} |")
        # trial 별 CSV artifact (대시보드에서 표로 미리보기)
        csv = TMP / f"curve_{trial.number}.csv"
        steps = np.arange(10)
        csv.write_text("step,loss\n" + "\n".join(f"{s},{loss * (1 + 3 / (s + 1)):.4f}" for s in steps))
        upload_artifact(artifact_store=STORE, file_path=str(csv), study_or_trial=trial)
        return loss

    study.optimize(objective, n_trials=30)
    xs = [t.params["x"] for t in study.trials]
    ys = [t.params["y"] for t in study.trials]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers",
                               marker=dict(color=[t.value for t in study.trials], colorscale="Viridis",
                                           showscale=True)))
    fig.update_layout(title="사용자 정의 그래프: 탐색한 (x, y) 위치", xaxis_title="x", yaxis_title="y")
    od.save_plotly_graph_object(study, fig, graph_object_id="search-points")
    print("  study:", study.study_name, "| best:", round(study.best_value, 4))

    # ------------------------------------------------------------------
    header("17-2. Human-in-the-loop: 사람이 대시보드에서 평가")
    hitl = fresh_dashboard_study(f"{PREFIX}/human-in-the-loop", direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=0))
    hitl.set_metric_names(["사람 평가 점수"])
    od.register_objective_form_widgets(hitl, widgets=[
        od.SliderWidget(min=1, max=10, step=1, description="이 색이 얼마나 마음에 드나요? (1~10)",
                        labels=[(1, "별로"), (5, "보통"), (10, "최고")]),
    ])
    for _ in range(4):  # trial 을 RUNNING 상태로 남겨두면 대시보드에 입력 폼이 나타난다
        t = hitl.ask()
        r, g, b = (t.suggest_float(c, 0, 1) for c in "rgb")
        aid = color_image(t, r, g, b, "hitl")
        od.save_note(t, f"색상 RGB = ({r:.2f}, {g:.2f}, {b:.2f})\n\n아래 artifact 이미지를 보고 점수를 입력하세요.")
        t.set_user_attr("image_artifact_id", aid)
    print("  RUNNING trial 4개 생성 → 대시보드의 해당 study 에서 슬라이더로 점수 입력")

    # 자동 평가(코드)와 사람 평가를 섞는 예: ObjectiveUserAttrRef
    mixed = fresh_dashboard_study(f"{PREFIX}/objective-from-user-attr",
                                  directions=["minimize", "minimize"])  # 위젯 수 = 목적 수
    mixed.set_metric_names(["auto_score", "human_judgement"])
    od.register_objective_form_widgets(mixed, widgets=[
        od.ObjectiveUserAttrRef(key="auto_score"),             # 코드가 미리 계산해 둔 값 사용
        od.ChoiceWidget(choices=["Good", "Bad"], values=[0, 1], description="사람 판단"),
    ])
    for _ in range(2):
        t = mixed.ask()
        x = t.suggest_float("x", -5, 5)
        t.set_user_attr("auto_score", x**2)
    print("  ObjectiveUserAttrRef 예제 study 생성 (목적 2개 중 1개는 자동, 1개는 사람)")

    # 완료된 trial 에 사람이 메타 정보(user_attr)를 추가로 입력하는 폼
    # ※ objective 폼과 같은 저장 키를 쓰므로 한 study 에는 둘 중 하나만 등록 가능
    uattr = fresh_dashboard_study(f"{PREFIX}/user-attr-form", sampler=optuna.samplers.TPESampler(seed=0))
    od.register_user_attr_form_widgets(uattr, widgets=[
        od.TextInputWidget(description="코멘트 (선택)", user_attr_key="comment", optional=True),
        od.ChoiceWidget(choices=["채택", "보류", "기각"], values=[1, 0, -1],
                        description="리뷰 결과", user_attr_key="review"),
    ])
    uattr.optimize(lambda t: quadratic(t.suggest_float("x", -5, 5), t.suggest_float("y", -5, 5)),
                   n_trials=5)
    print("  user_attr 입력 폼 study 생성 (완료된 trial 5개에 리뷰 결과 입력)")

    # ------------------------------------------------------------------
    header("17-3. Preferential optimization (후보 비교로 사람 선호도 학습)")
    try:
        optuna.delete_study(study_name=f"{PREFIX}/preferential", storage=dashboard_storage())
    except KeyError:
        pass
    pstudy = preferential.create_study(n_generate=4, storage=dashboard_storage(),
                                       study_name=f"{PREFIX}/preferential",
                                       sampler=optuna.samplers.RandomSampler(seed=0))
    od.register_preference_feedback_component(pstudy, "artifact", "rgb_image")
    n = 0
    while pstudy.should_generate() and n < 4:  # 사람 평가 대기 중인 후보를 4개까지 생성
        t = pstudy.ask()
        r, g, b = (t.suggest_float(c, 0, 1) for c in "rgb")
        t.set_user_attr("rgb_image", color_image(t, r, g, b, "pref"))
        n += 1
    print(f"  후보 {n}개 생성 → 대시보드에서 'WORST' 를 누르면 선호도가 기록됨")
    print("  (실서비스에선 while True 루프로 계속 돌리며 사람이 고른 결과를 반영해 새 후보 생성)")

    print("\n  대시보드 실행:  python dashboard.py   → http://127.0.0.1:8080")


if __name__ == "__main__":
    main()
