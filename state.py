# smeft_rl/state.py

from dataclasses import dataclass, field
from typing import List, Dict, Callable, Tuple
import torch


@dataclass
class SMEFTState:
    obs_names: List[str]
    pulls: List[float]
    chi2: float                          # now required — comes from the minimizer, not derived
    fired_ops: List[str] = field(default_factory=list)

    def __post_init__(self):
        assert len(self.obs_names) == len(self.pulls), \
            f"obs_names ({len(self.obs_names)}) and pulls ({len(self.pulls)}) length mismatch"

    def pulls_dict(self) -> Dict[str, float]:
        return dict(zip(self.obs_names, self.pulls))

    def top_pulls(self, k: int = 5) -> List[tuple]:
        return sorted(self.pulls_dict().items(), key=lambda kv: -abs(kv[1]))[:k]

    def __repr__(self) -> str:
        obs_str = "\n".join(f"  {n}: {p:+.3f}" for n, p in zip(self.obs_names, self.pulls))
        ops_str = ", ".join(self.fired_ops) if self.fired_ops else "(none)"
        return f"SMEFTState(\n{obs_str}\n  Ops fired: [{ops_str}]\n  Chi2: {self.chi2:.4f}\n)"


# pull_fn now returns chi2 explicitly too — it owns the minimization, not the state
# (fired_ops) -> (obs_names, pulls, chi2)
PullFn = Callable[[List[str]], Tuple[List[str], List[float], float]]


def step(state: SMEFTState, action_op: str, pull_fn: PullFn) -> SMEFTState:
    """
    Fire `action_op`, call pull_fn to get fresh pulls AND chi2 for the new
    active operator set (pull_fn is responsible for the minimization —
    whether that's a real Minuit fit, the closed-form GLS, or a toy stand-in).
    """
    if action_op in state.fired_ops:
        raise ValueError(f"{action_op} already fired in this state")

    fired = state.fired_ops + [action_op]
    obs_names, pulls, chi2 = pull_fn(fired)

    return SMEFTState(
        obs_names=obs_names,
        pulls=pulls,
        chi2=chi2,
        fired_ops=fired,
    )
