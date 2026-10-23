"""14. 사용자 정의 Sampler / Pruner 만들기

다루는 기능
- BaseSampler 상속: infer_relative_search_space / sample_relative / sample_independent
- before_trial / after_trial 훅
- BasePruner 상속: prune(study, trial)
- study.get_trials / trial.intermediate_values 활용
"""
import numpy as np
from common import header, rosenbrock

import optuna
from optuna.distributions import BaseDistribution, FloatDistribution
from optuna.samplers import BaseSampler, RandomSampler
from optuna.search_space import IntersectionSearchSpace
from optuna.trial import FrozenTrial, TrialState


class SimulatedAnnealingSampler(BaseSampler):
    """간단한 담금질(Simulated Annealing) 샘플러 (Optuna 공식 튜토리얼 방식)."""

    def __init__(self, temperature: float = 100.0, cooldown: float = 0.9, seed: int | None = None):
        self._rng = np.random.RandomState(seed)
        self._independent = RandomSampler(seed=seed)
        self._search_space = IntersectionSearchSpace()
        self._current: FrozenTrial | None = None
        self._temperature = temperature
        self._cooldown = cooldown

    def infer_relative_search_space(self, study, trial) -> dict[str, BaseDistribution]:
        return self._search_space.calculate(study)

    def sample_relative(self, study, trial, search_space):
        if not search_space:
            return {}
        prev = study.trials[-2] if len(study.trials) >= 2 else None
        if self._current is None or (prev and prev.state == TrialState.COMPLETE
                                     and self._accept(prev.value, self._current.value)):
            if prev and prev.state == TrialState.COMPLETE:
                self._current = prev
        self._temperature *= self._cooldown
        if self._current is None:
            return {}
        params = {}
        for name, dist in search_space.items():
            if not isinstance(dist, FloatDistribution):
                continue
            width = (dist.high - dist.low) * 0.1
            v = self._current.params[name] + self._rng.uniform(-width, width)
            params[name] = float(np.clip(v, dist.low, dist.high))
        return params

    def _accept(self, new: float, cur: float) -> bool:
        if new <= cur:
            return True
        return self._rng.uniform() < np.exp((cur - new) / max(self._temperature, 1e-12))

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self._independent.sample_independent(study, trial, param_name, param_distribution)

    # (선택) 훅
    def before_trial(self, study, trial) -> None:
        pass

    def after_trial(self, study, trial, state, values) -> None:
        pass


class LastStepWorseThanBestPruner(optuna.pruners.BasePruner):
    """현재 step 의 값이 '지금까지 완료된 trial 들의 같은 step 최솟값' 보다
    margin 배 이상 나쁘면 프루닝."""

    def __init__(self, margin: float = 1.5, n_warmup_steps: int = 2) -> None:
        self.margin = margin
        self.n_warmup_steps = n_warmup_steps

    def prune(self, study: optuna.Study, trial: FrozenTrial) -> bool:
        step = trial.last_step
        if step is None or step < self.n_warmup_steps:
            return False
        others = [t.intermediate_values[step]
                  for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
                  if step in t.intermediate_values]
        if not others:
            return False
        return trial.intermediate_values[step] > min(others) * self.margin


def rosen_obj(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -3, 3)
    y = trial.suggest_float("y", -3, 3)
    return rosenbrock(x, y)


def curve_obj(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    final = (x - 1) ** 2 + 0.1
    for step in range(10):
        trial.report(final * (1 + 5 / (step + 1)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return final


def main() -> None:
    header("14-1. 사용자 정의 SimulatedAnnealingSampler")
    for name, sampler in [("Random", RandomSampler(seed=0)),
                          ("SimulatedAnnealing", SimulatedAnnealingSampler(seed=0))]:
        st = optuna.create_study(sampler=sampler)
        st.optimize(rosen_obj, n_trials=200)
        print(f"  {name:20s} best={st.best_value:.4f} params="
              f"{ {k: round(v, 3) for k, v in st.best_params.items()} }")

    header("14-2. 사용자 정의 Pruner")
    st = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0),
                             pruner=LastStepWorseThanBestPruner(margin=1.5))
    st.optimize(curve_obj, n_trials=40)
    n_pruned = len(st.get_trials(states=(TrialState.PRUNED,)))
    print(f"  best={st.best_value:.4f}, pruned={n_pruned}/{len(st.trials)}")


if __name__ == "__main__":
    main()
