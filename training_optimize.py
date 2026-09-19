from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import os
import json

import warnings
import flavio
from wilson import wcxf
from iminuit import Minuit

import time

from dataclasses import dataclass, field
from typing import List

debut = time.time()

OBS_NAMES = [
    'GammaZ', 'sigma_had', 'R_e', 'R_mu', 'R_tau',
    'AFB(Z->ee)', 'AFB(Z->mumu)', 'AFB(Z->tautau)',
    'A(Z->ee)', 'A(Z->mumu)', 'A(Z->tautau)', 
    'R_b', 'R_c', 'AFB(Z->bb)','AFB(Z->cc)', 'A(Z->bb)', 'A(Z->cc)',
    'm_W', 'GammaW',
    'BR(W->enu)', 'BR(W->munu)', 'BR(W->taunu)',
    'R(W->cX)', 'Rmue(W->lnu)', 'Rtaue(W->lnu)', 'Rtaumu(W->lnu)', 'A(Z->ss)']


COEFFICIENTS_PATH = "ewp_linear_coefficients_2.json"

# --- Override m_W Measurement --- #
mW_new_central = 80.411  # GeV
mW_new_error = 0.008    # GeV

for name in list(flavio.Measurement.instances.keys()):
    if name == 'CDF_measurement': continue
    m = flavio.Measurement.instances[name]
    if 'm_W' in m.all_parameters:
        print(f"Removing old m_W measurement: {name}")
        flavio.Measurement.del_instance(name)

my_mW = flavio.Measurement('CDF_measurement')
my_mW.add_constraint(['m_W'],flavio.statistics.probability.NormalDistribution(mW_new_central, mW_new_error))

with open(COEFFICIENTS_PATH, "r", encoding="utf-8") as f: ewp_data = json.load(f)

ewp_coefficients = ewp_data["observables"]


def predict_observables(wc_dict):
    def predict_ewp(obs: str, wc_dict: dict) -> float:
        """Computes linearized SMEFT prediction for a single observable."""
        if obs not in ewp_coefficients: raise KeyError(f"Observable '{obs}' not found in loaded JSON data.")
    
        data = ewp_coefficients[obs]
    
        # Fall back to 0.0 if sm_prediction isn't present
        sm = data.get("sm_prediction", 0.0)
        
        coeffs = data["coefficients"]
    
        delta = sum(coeffs[wc] * val for wc, val in wc_dict.items() if wc in coeffs)
        return sm *(1+ delta)
    """Computes predictions for all observables in OBS_NAMES."""
    return np.array([predict_ewp(obs, wc_dict) for obs in OBS_NAMES])


def get_experimental_constraint(obs_name: str) -> tuple:
    """Extracts central value and standard deviation for a given observable from flavio."""
    for name, measurement in flavio.Measurement.instances.items():
        if obs_name not in measurement.all_parameters: continue

        for distribution, parameters in measurement._constraints:
            if obs_name in parameters:
                if hasattr(distribution, "covariance"):
                    idx = parameters.index(obs_name)
                    central = distribution.central_value[idx]
                    covariance = np.array(distribution.covariance)
                    sigma = np.sqrt(covariance[idx, idx])
                    return float(central), float(sigma)
                else:
                    central = distribution.central_value
                    sigma = distribution.standard_deviation
                    if isinstance(central, (list, tuple)): central = central[0]
                    if isinstance(sigma, (list, tuple)): sigma = sigma[0]
                    return float(central), float(sigma)

    raise ValueError(f"No experimental constraint found for {obs_name}")
    

EXP_CENTRAL = []
EXP_SIGMA = []
EXP_SIGMA2 = []

for obs in OBS_NAMES:
    exp, sigma = get_experimental_constraint(obs)
    sigma2 = sigma**2 + flavio.sm_uncertainty(obs, N=100)**2
    
    EXP_CENTRAL.append(exp)
    EXP_SIGMA.append(np.sqrt(sigma2))
    EXP_SIGMA2.append(sigma2)


def compute_chi2(predictions: list) -> float:
    return np.sum((predictions - EXP_CENTRAL)**2 / EXP_SIGMA2)
        
# Run after helper functions are defined
CHI2_SM = compute_chi2(predict_observables({}))
print(f"[chi2_reward] chi2_SM = {CHI2_SM:.6f}")


def _minimise_chi2(wc_names: list) -> tuple:
    chi2_fn = _build_chi2_fn(wc_names)
    # Initial values
    start = {name: 0.0 for name in wc_names}

    m = Minuit(chi2_fn, **start)

    for name in wc_names:
        m.errors[name] = 1e-3
        m.limits[name] = (-1.0, 1.0)

    m.migrad()

    chi2_min = float(m.fval)
    best_wc = {k: float(v) for k, v in zip(wc_names, m.values)}
    
    return chi2_min, best_wc

def _build_chi2_fn(wc_names: list):
    def chi2(*values):
        try:
            wc_dict = {wc_names[i]: float(values[i]) for i in range(len(wc_names))}

            predictions = predict_observables(wc_dict)
            result = compute_chi2(predictions)

            return float(result) if np.isfinite(result) else 1e20

        except Exception as e: print(f"[chi2] Exception: {e}")

    fn_src = (f"def chi2_named({', '.join(wc_names)}):\n" f"    return chi2({', '.join(wc_names)})\n")

    globs = {"chi2": chi2}
    exec(fn_src, globs)
    return globs["chi2_named"]


def flavio_pull_fn(fired_ops: list) -> tuple:
    """
    Given a list of active operators (fired_ops), minimizes chi2
    and returns predictions, pulls, and the best-fit Wilson coefficients.
    """
    if len(fired_ops) == 0: chi2_min, best_wc = CHI2_SM, {}
        
    else: chi2_min, best_wc = _minimise_chi2(fired_ops)

    predictions = predict_observables(best_wc)

    pulls = (predictions - EXP_CENTRAL) / EXP_SIGMA

    if not np.all(np.isfinite(pulls)): raise ValueError(f"Non-finite pulls encountered: {pulls}")
    if np.max(np.abs(pulls)) > 10000:  raise ValueError(f"Unphysical pull detected: {pulls}")

    return pulls.tolist(), float(chi2_min), predictions.tolist()#, best_wc

BASELINE_PULLS, _, BASELINE_PREDICTIONS = flavio_pull_fn([])

for obs_name, prediction, central, sigma, pull in zip(
    OBS_NAMES, BASELINE_PREDICTIONS, EXP_CENTRAL, EXP_SIGMA, BASELINE_PULLS):
    print(f"{obs_name:20s} SM={prediction:.6g} exp={central:.6g} sigma={sigma:.6g} pull={pull:.3f}")


# _, test_chi2, _ = flavio_pull_fn([ 'phid_33', 'phiq1_23'])
# _, test_chi2, _ = flavio_pull_fn([ 'll_1122'])

# print("\n\n", 53.8 - test_chi2, "\n\n")

@dataclass
class SMEFTState:
    obs_names: List[str]
    pulls: List[float]
    chi2: float                          # now required — comes from the minimizer, not derived
    fired_ops: List[str] = field(default_factory=list)

    def __post_init__(self):
        assert len(self.obs_names) == len(self.pulls), \
            f"obs_names ({len(self.obs_names)}) and pulls ({len(self.pulls)}) length mismatch"

    def copy(self):
         new_state = SMEFTState(pulls=np.copy(self.pulls), chi2=self.chi2, obs_names=self.obs_names)
         new_state.active_ops = list(self.active_ops) if hasattr(self, 'active_ops') else []
         new_state.step = self.step if hasattr(self, 'step') else 0
         return new_state
     
    def to_dict(self):
         return {"pulls": self.pulls.tolist(),
                 "chi2": self.chi2,
                 "obs_names": self.obs_names,
                 "active_ops": getattr(self, 'active_ops', []),
                 "step": getattr(self, 'step', 0) }
       

# LOAD OPERATOR VOCABULARY

warnings.filterwarnings("ignore")

def _build_catalogue_from_wcxf() -> dict[str, list[str]]:
    """
    Query the wcxf package for every WC in the SMEFT Warsaw basis and organise them by their wcxf 'sector' label.
    """

    basis_obj = wcxf.Basis["SMEFT", "Warsaw"]
    catalogue: dict[str, list[str]] = {}

    for sector_name, sector_data in basis_obj.sectors.items():
        if sector_name == "dB=de=dmu=dtau=0":
            wcs = sorted(sector_data.keys())
            if wcs: catalogue[sector_name] = wcs

    return catalogue


OPERATOR_CATALOGUE = _build_catalogue_from_wcxf()

operator_vocab = [wc for ops in OPERATOR_CATALOGUE.values() for wc in ops]

# VOCAB_FILE = Path("vocab_0f2f.txt")

# with open(VOCAB_FILE, "r", encoding="utf-8") as f:
#     operator_vocab = [ line.strip() for line in f if line.strip() and not line.startswith("#")]


print(f"Loaded {len(operator_vocab)} operators")

SEED = 42

print(f"Using seed = {SEED}")

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

n_ops = len(operator_vocab)
n_obs = len(OBS_NAMES)

M_ij = np.zeros((n_ops,n_obs), dtype=np.float32)
print("M_ij shape =", M_ij.shape)

print("operator_vocab =", len(operator_vocab))
print("constraints =", len(OBS_NAMES))


# BOTH actor and critic receive this SAME state.

# Actor: pi_theta(a | s_t)
# Critic: V_phi(s_t)

# The sampled return-to-go  G_t = r_t + gamma r_{t+1} + ...
# is used as the Monte-Carlo estimate of Q^pi(s_t, a_t) and A_t = G_t - V_phi(s_t)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SMEFTActorCritic(nn.Module):
    def __init__(

        self,  op_vocab, obs_names, sensitivity_matrix, chi2_reference,
        max_steps=10, embed_dim=32, n_heads=4, alpha_init=1.0, pull_gain=2.5):

        super().__init__()
        self.op_vocab = op_vocab
        self.obs_names = obs_names
        self.num_ops = len(op_vocab)
        self.num_obs = len(obs_names)
        self.embed_dim = embed_dim
        self.max_steps = max_steps

        # op to integer index 
        self.op_to_idx = {op: i for i, op in enumerate(op_vocab) }

        # Sensitivity matrix
        sm_shape = tuple(sensitivity_matrix.shape)

        if sm_shape == (self.num_obs, self.num_ops): M = sensitivity_matrix.T

        elif sm_shape == (self.num_ops, self.num_obs): M = sensitivity_matrix

        else: raise ValueError("sensitivity matrix shape does not match")

        self.register_buffer("M", torch.as_tensor(M, dtype=torch.float32))

        # We normalize chi2 so the network does not receive a potentially large raw number.
        self.chi2_reference = float( max(abs(chi2_reference), 1.0)) 

        # STATE ENCODING

        # Operator identity
        self.op_embed = nn.Embedding(self.num_ops, embed_dim)

        # Observable pulls
        self.pull_proj = nn.Linear(1, embed_dim)

        # Active operator status: 0 = not active   1 = active
        self.active_embed = nn.Embedding(2, embed_dim)

        # Fired/evaluated status:  0 = not fired   1 = fired
        self.fired_embed = nn.Embedding(2, embed_dim)

        # Cross-attention between operators and observable pulls
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)

        # Global state featuresvchi2 + step
        self.global_state_encoder = nn.Sequential(nn.Linear(2, embed_dim), nn.Tanh(), nn.Linear(embed_dim, embed_dim))

        # ====================================================
        # ACTOR
        # ====================================================

        # Each operator receives: operator identity, pull/attention information, active status, fired status, global state from s_t
        self.actor_head = nn.Sequential(nn.Linear(embed_dim * 5, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1))

        # ====================================================
        # CRITIC
        # ====================================================

        # Critic receives a pooled representation of the SAME operator-level state representation used by actor together with global state
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim * 6, embed_dim), nn.ReLU(), 
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), 
            nn.Linear(embed_dim // 2, 1))

        # Trainable scaling parameters
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(pull_gain, dtype=torch.float32))
    
    # ========================================================
    # STATE ENCODER
    # ========================================================

    def encode_state(self, state, fired_ops=None):
        """
        Construct ONE explicit representation of s_t.
        This function is the central guarantee that actor and critic receive the SAME state.
        """

        device = self.M.device

        pulls_data = (state.pulls if hasattr(state, "pulls") else state["pulls"])

        if not isinstance(pulls_data, torch.Tensor): pulls_tensor = torch.tensor(pulls_data, dtype=torch.float32, device=device)
        else: pulls_tensor = pulls_data.to(device=device, dtype=torch.float32)
        if pulls_tensor.ndim != 1: pulls_tensor = pulls_tensor.flatten()
        if pulls_tensor.numel() != self.num_obs: raise ValueError(f"Expected {self.num_obs} pulls, " f"got {pulls_tensor.numel()}")

        # ----------------------------------------------------
        # Current chi2
        # ----------------------------------------------------

        chi2 = float(getattr(state, "chi2", state.get("chi2", 0.0) if isinstance(state, dict) else 0.0))
        chi2_normalized = (chi2 / self.chi2_reference)

        step_value = float(getattr( state, "step", state.get("step", 0) if isinstance(state, dict) else 0))
        step_fraction = (step_value / float(max(self.max_steps, 1)))

        op_ids = torch.arange( self.num_ops, device=device)
        op_repr = self.op_embed( op_ids )

        pull_tokens = self.pull_proj(pulls_tensor.unsqueeze(-1))

        # MultiheadAttention expects: query = operator tokens, key/value = observable pull tokens
        # Shape: (1, num_ops, embed_dim),  (1, num_obs, embed_dim)
        op_query = op_repr.unsqueeze(0)
        pull_tokens = pull_tokens.unsqueeze(0)

        attn_out, attn_weights = self.cross_attn(query=op_query, key=pull_tokens, value=pull_tokens)
        attn_out = attn_out.squeeze(0)

        active_mask = torch.zeros(self.num_ops, dtype=torch.long, device=device)

        active_ops = getattr(state, "active_ops", [])

        for op in active_ops:
            if op in self.op_to_idx: active_mask[self.op_to_idx[op]] = 1

        fired_mask = torch.zeros(self.num_ops, dtype=torch.long, device=device)

        if fired_ops is not None:
            for idx in fired_ops:
                idx = int(idx)
                if 0 <= idx < self.num_ops: fired_mask[idx] = 1


        active_repr = self.active_embed(active_mask)
        fired_repr = self.fired_embed(fired_mask)
        global_scalar_state = torch.tensor([chi2_normalized, step_fraction], dtype=torch.float32, device=device).unsqueeze(0)
        global_repr = self.global_state_encoder(global_scalar_state)

        # Broadcast global state to every operator
        global_per_operator = (global_repr.expand(self.num_ops, -1))

        # ONE SHARED STATE REPRESENTATION
        operator_state = torch.cat([op_repr, attn_out, active_repr, fired_repr, global_per_operator], dim=-1)

        return (operator_state, global_repr, attn_weights, pulls_tensor, active_mask, fired_mask)

    def forward(self, state, fired_ops=None):

        """
        Returns: logits, V(s), attention, diagnostics
        Both actor and critic are computed from the SAME state encoding produced by encode_state().
        """

        (operator_state, global_repr, attn_weights, pulls_tensor, active_mask, fired_mask) = self.encode_state(state, fired_ops=fired_ops)

        # ====================================================
        # ACTOR
        # ====================================================

        neural_logits = (self.actor_head(operator_state).squeeze(-1) * self.alpha)

        # Current policy
        final_logits = neural_logits

        # Mask operators that have already been fired
        final_logits = final_logits.masked_fill(fired_mask.bool(),-1e9)

        # ====================================================
        # CRITIC
        # ====================================================

        # actor = f(s), critic = g(s) with the same s.

        pooled_operator_state = (operator_state.mean(dim=0, keepdim=True))
        
        critic_input = torch.cat([pooled_operator_state, global_repr], dim=-1)

        # Defensive check: operator_state contains 5 * embed_dim features,
        # and global_repr contributes another embed_dim.
        expected_critic_dim = self.embed_dim * 6
        if critic_input.shape[-1] != expected_critic_dim:
            raise RuntimeError(
                f"Critic input dimension mismatch: got "
                f"{critic_input.shape[-1]}, expected {expected_critic_dim}. "
                f"pooled_operator_state={tuple(pooled_operator_state.shape)}, "
                f"global_repr={tuple(global_repr.shape)}"
            )

        state_value = self.critic_head(critic_input).squeeze()

        return (final_logits, state_value, attn_weights.squeeze(0))#, diagnostics)


# ============================================================
# STEP-LEVEL REWARD
# ============================================================

def compute_step_rewards(history, alpha_step=0.1, dead_step_penalty=0.5, min_delta_chi2=1e-3, solved_bonus=10.0, chi2_threshold=1e-6):
    """
    One reward for every action. delta_chi2 = trial_chi2 - old_chi2
    Hence chi2_drop = -delta_chi2 is positive when the action improves the fit.
    """
    rewards = []
    
    for step in history:
    
        delta = float(step["delta_chi2"])
        reward = - delta - alpha_step

        if -delta < min_delta_chi2: reward -= dead_step_penalty

        if (step["trial_chi2"] < chi2_threshold): reward += solved_bonus

        rewards.append( reward )

    return rewards


# ============================================================
# RETURN-TO-GO
# ============================================================

def compute_returns_to_go(step_rewards, gamma=0.99):
    """
    G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...
    This is the Monte-Carlo estimate used as: G_t ~ Q^pi(s_t, a_t)
    """
    len_step_rewards = len(step_rewards)
    returns = [0.0]*len_step_rewards

    running = 0.0

    for t in range(len_step_rewards - 1, -1, -1):
        running = ( step_rewards[t] + gamma * running )
        returns[t] = running

    return returns


# ============================================================
# TRAJECTORY REWARD
# ============================================================

def compute_rollout_reward( final_state, baseline_chi2, trajectory_length=None, history=None, 
        alpha_step=0.1, dead_step_penalty=0.5, min_delta_chi2=1e-3, solved_bonus=10.0, chi2_threshold=1e-6):

    """
    Logging / best-trajectory reward. NOT used for policy gradient.
    """

    if isinstance(final_state, dict):
        if "chi2_total" in final_state: final_chi2 = final_state["chi2_total"]
        else: final_chi2 = final_state["chi2"]
    elif hasattr(final_state, "chi2"):  final_chi2 = final_state.chi2
    else: raise ValueError("final_state does not contain chi2")

    delta_chi2 = ( baseline_chi2 - final_chi2 )

    reward = delta_chi2

    if (abs(delta_chi2)<= min_delta_chi2): reward -= 30.0

    if final_chi2 < chi2_threshold: reward += solved_bonus

    if trajectory_length is not None: reward -= (alpha_step * trajectory_length)

    if history is not None:
        for item in history:
            delta = float( item.get("delta_chi2", 0.0))
            chi2_drop = -delta

            if (chi2_drop < min_delta_chi2): reward -= dead_step_penalty

    return (reward, float(delta_chi2))

# ============================================================
# RUNNING RETURN NORMALIZER
# ============================================================

class RunningReturnNormalizer:
    """
    Running normalization of Monte-Carlo returns.
    The critic therefore learns a normalized value:
        V_phi(s) ~ normalized G_t
    rather than a value on an ever-growing raw reward scale.
    """

    def __init__(self, momentum=0.01, eps=1e-6):

        self.momentum = momentum
        self.eps = eps
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False

    def update(self, returns_t):
        batch_mean = returns_t.mean().item()
        batch_var  = returns_t.var(unbiased=False).item()

        if not self.initialized:
            self.mean = batch_mean
            self.var = max(batch_var, self.eps)
            self.initialized = True

        else:
            self.mean = (1.0 - self.momentum) * self.mean + self.momentum * batch_mean
            self.var  = max((1.0 - self.momentum) * self.var + self.momentum * batch_var, self.eps)

    def normalize(self, returns_t):
        return (returns_t - self.mean) / ((self.var ** 0.5) + self.eps)

    def update_and_normalize(self, returns_t):
        self.update(returns_t)
        return self.normalize(returns_t)


# ============================================================
# ROLLOUT
# ============================================================

def rollout(policy, pull_fn, obs_names, baseline_pulls, baseline_chi2,
    max_steps=10, verbose=True, trajectory_idx=1, total_trajectories=30):

    # initial step
    current_state = SMEFTState(pulls=np.array(baseline_pulls, dtype=np.float32), chi2=float(baseline_chi2), obs_names=obs_names)
    current_state.active_ops = []
    current_state.step = 0

    # Operators that have already been evaluated

    all_evaluated_op_indices = set()
    accepted_op_indices = []

    # Training quantities
    log_probs, entropies, attentions = [], [], []

    # V(s_t)
    state_values = []
    state_trajectory = [current_state.copy()]

    history = []

    if verbose: print(f"\n--- Trajectory " f"[{trajectory_idx:02d} / " f"{total_trajectories:02d}] ---")

    for step_idx in range(max_steps):
        (logits, state_val, attn_weights) = policy(current_state, fired_ops=all_evaluated_op_indices)

        # Policy distribution
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        action_idx = int(action.item())

        # Store action information

        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        attentions.append(attn_weights.detach())

        # IMPORTANT: This is V(s_t), i.e. the value BEFORE taking action a_t.
        state_values.append(state_val)

        # Mark action as evaluated
        all_evaluated_op_indices.add(action_idx)

        proposed_op = (policy.op_vocab[action_idx])

        # Environment transition. The active operators define the current SMEFT theory.
        trial_ops = (list(current_state.active_ops) + [proposed_op])

        ( trial_pulls, trial_chi2, _) = pull_fn(trial_ops)

        trial_chi2 =  trial_chi2 
        old_chi2 = current_state.chi2
        delta_raw = trial_chi2 - old_chi2

        accepted = False

        if (-delta_raw > deltachi2_min):
            accepted = True

            accepted_op_indices.append(action_idx)
            current_state = SMEFTState( pulls=np.array(trial_pulls, dtype=np.float32 ), chi2=trial_chi2, obs_names=obs_names)
            current_state.active_ops = (list(trial_ops))

        # state step increases whether the operator is accepted or not
        current_state.step = step_idx + 1

        state_trajectory.append(current_state.copy())
        if verbose:
            status = "ACCEPT" if accepted else "REJECT"
            if status == "ACCEPT": print(f"STEP {step_idx:02d}: " f"ADD {proposed_op:20s} " f"Deltachi^2={delta_raw:+8.3f} " f"{status}")

        history.append({
            "step": step_idx,
            "proposed_operator": proposed_op,
            "accepted": accepted,
            "state_before": state_trajectory[-2].to_dict(),
            "state_after": current_state.to_dict(),
            "trial_chi2": float(trial_chi2),
            "delta_chi2": float(delta_raw),
        })

    if verbose:
        if current_state.active_ops: final_ops_str = (" -> ".join(current_state.active_ops))
        else: final_ops_str = ("None (Pure SM)")

        total_delta_chi2 = (float(baseline_chi2) - current_state.chi2)

        print(f"FINAL SEQUENCE " f"[{len(current_state.active_ops)} ops]: " f"{final_ops_str}")
        print(f"FINAL RESULTS: " f"Total Deltachi^2 = " f"{total_delta_chi2:+.3f} | " f"Final chi^2 = " f"{current_state.chi2:.3f}")

    return {
        "final_state": current_state,
        "state_trajectory": state_trajectory,
        "ops": accepted_op_indices,
        "op_names": current_state.active_ops,
        "log_probs": torch.stack(log_probs),
        "entropies": torch.stack(entropies),
        "state_values": torch.stack( state_values ),
        "attention": attentions,
        "history": history,
    }


# ============================================================
# HYPERPARAMETERS
# ============================================================

MAX_STEPS = 10
ROLLOUTS_PER_BATCH = 25
N_BATCHES = 300

deltachi2_min = 1e-2

GAMMA = 0.99

CRITIC_LOSS_COEF = 0.5

ADV_NORMALIZE = True

RETURN_NORM_MOMENTUM = 0.01

# ============================================================
# MODEL INITIALIZATION
# ============================================================

M_tensor = torch.tensor(M_ij, dtype=torch.float32)
M_mean = M_tensor.mean()
M_std = (M_tensor.std() + 1e-6)
M_normalized = (M_tensor - M_mean) / M_std

model = SMEFTActorCritic(
    op_vocab=operator_vocab, obs_names=OBS_NAMES, sensitivity_matrix= M_normalized, chi2_reference= CHI2_SM, max_steps= MAX_STEPS,
    embed_dim=32, n_heads=4, alpha_init=1.0, pull_gain=2.5).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

return_normalizer = (RunningReturnNormalizer(momentum=RETURN_NORM_MOMENTUM))

# ============================================================
# OUTPUT
# ============================================================
OUTPUT_DIR = Path(__file__).parent
THEORY_ARCHIVE_PATH = os.path.join(OUTPUT_DIR, "theory_archive_vArxiv.json")

if not os.path.exists(THEORY_ARCHIVE_PATH):
    with open(THEORY_ARCHIVE_PATH, "w") as f: json.dump( [], f, indent=2)

theory_archive = []

entropy_coeff_arr = [0.01 * (1.0 - batch_idx / float(N_BATCHES)) + 0.0001 * batch_idx / float(N_BATCHES) 
                     for batch_idx in range(N_BATCHES)] 

# TRAINING LOOP
for batch_idx in range(N_BATCHES):

    model.train()

    batch_traj_rewards = []
    batch_delta_chi2 = []

    # Step-level quantities
    all_log_probs = []
    all_entropies = []
    all_state_values = []
    all_returns = []

    print("\n==================== " f"BATCH {batch_idx:03d} " "====================")

    for rollout_idx in range(ROLLOUTS_PER_BATCH):
        result = rollout(
            policy=model,
            pull_fn=flavio_pull_fn,
            obs_names=OBS_NAMES,
            baseline_pulls= BASELINE_PULLS,
            baseline_chi2= CHI2_SM,
            max_steps= MAX_STEPS,
            verbose=False,
            trajectory_idx= rollout_idx + 1,
            total_trajectories= ROLLOUTS_PER_BATCH)
        
        final_state = result["final_state"]
        hist = result["history"]
        traj_ops = result["ops"]
        traj_len = len(traj_ops)

        (traj_reward, delta_chi2) = compute_rollout_reward(
            final_state= final_state,
            baseline_chi2= CHI2_SM,
            trajectory_length= traj_len,
            history= hist,
            alpha_step=0.1,
            dead_step_penalty=0.5,
        )

        step_rewards = compute_step_rewards(hist, alpha_step=0.1, dead_step_penalty=0.5)
        returns_to_go = compute_returns_to_go(step_rewards, gamma=GAMMA)

        returns_t = torch.tensor(returns_to_go, dtype=torch.float32, device=DEVICE)

        # STORE ACTOR / CRITIC DATA
        all_log_probs.append(result["log_probs"])
        all_entropies.append(result["entropies"])
        all_state_values.append(result["state_values"])
        all_returns.append(returns_t)

        # ARCHIVE
        theory_archive.append({
            "batch": batch_idx,
            "rollout": rollout_idx,
            "traj_reward": float(traj_reward),
            "step_rewards": [float(r) for r in step_rewards],
            "returns_to_go": [float(r) for r in returns_to_go],
            "delta_chi2": float(delta_chi2),
            "chi2_min": float( final_state.chi2 ),
            "operators": result["op_names"],
            "pulls":  np.asarray(final_state.pulls).tolist(),
            "history": hist
        })

        batch_traj_rewards.append(float(traj_reward))
        batch_delta_chi2.append(float(delta_chi2))


    # write in the file not every step
    if batch_idx%20==0 or batch_idx == N_BATCHES - 1:
        with open(THEORY_ARCHIVE_PATH, "w") as f: json.dump(theory_archive, f, indent=2)

    # NORMALIZED RETURNS
    flat_returns_norm = (return_normalizer.update_and_normalize(torch.cat(all_returns)))
    # FLATTEN ALL STEPS
    flat_values = torch.cat(all_state_values)

    # RAW ADVANTAGE G_t is the Monte-Carlo Q estimate. V(s_t) is the critic. Therefore: A_t = G_t - V(s_t)
    # ADVANTAGE NORMALIZATION
    advantages = ( flat_returns_norm - flat_values.detach())

    if (ADV_NORMALIZE and advantages.numel() > 1 and advantages.std().item() > 1e-6):
        advantages = ( ( advantages - advantages.mean() ) / ( advantages.std() + 1e-8 ) )

    actor_loss = -(advantages.detach() * torch.cat(all_log_probs)).mean()
    critic_loss = nn.functional.mse_loss(flat_values, flat_returns_norm.detach())
    entropy_loss = (-entropy_coeff_arr[batch_idx] * torch.cat(all_entropies).mean())

    total_loss = actor_loss + CRITIC_LOSS_COEF * critic_loss + entropy_loss

    # BACKPROP
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()

    print(f"\n>>> SUMMARY " f"BATCH {batch_idx:03d} <<<")

    print(
        f"Actor: {actor_loss.item():.4f} | "
        f"Critic: {critic_loss.item():.4f} | "
        f"Entropy: {entropy_loss.item():.4f} | "
        f"Mean Deltachi^2: {np.mean(batch_delta_chi2):.2f} | "
        f"Max Traj Reward: {np.max(batch_traj_rewards):.1f}"
    )

    # print(f"Gradient norm: " f"{raw_grad_norm:.4f}")


with open(THEORY_ARCHIVE_PATH, "r") as f: results = json.load(f)
# Keep only the best occurrence of each unique operator set
unique_theories = {}

for theory in results:
    # Operator ordering should not matter
    key = tuple(sorted(theory["operators"]))
    # Keep the one with the highest reward
    if (key not in unique_theories
        or theory["traj_reward"] > unique_theories[key]["traj_reward"]):
        unique_theories[key] = theory

# Sort unique theories by reward
top_theories = sorted(unique_theories.values(), key=lambda theory: theory["traj_reward"], reverse=True)[:50]

print("\n========== TOP 50 UNIQUE THEORIES ==========\n")
for rank, theory in enumerate(top_theories, start=1):
    print( f"{rank:2d}. "
           f"Reward = {theory['traj_reward']:.6f}, "
           f"Operators = {sorted(theory['operators'])}, "
           f"Δχ² = {theory['delta_chi2']:.4f}")
    
print("Total computation time :", time.time()-debut, " sec")
input("")