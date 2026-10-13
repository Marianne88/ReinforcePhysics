from pathlib import Path
import time
import json
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
from itertools import accumulate

import flavio
from flavio.statistics.probability import NormalDistribution
from wilson import wcxf
from iminuit import Minuit

debut = time.time()

fictitious_anomalies = True # if False, the analysis will be on the m_W mass anomaly

# --- random seed --- #
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
print(f"Using seed = {SEED}")

# --- load data from json --- #
with open("ewp_linear_coefficients_2.json", "r") as f: observables_data = json.load(f)["observables"]

# --- load observable names --- #
if not fictitious_anomalies:
    # Override m_W Measurement
    for name in list(flavio.Measurement.instances.keys()):
        if 'm_W' in flavio.Measurement.instances[name].all_parameters:
            print(f"Removing old m_W measurement: {name}"); flavio.Measurement.del_instance(name)
    
    flavio.Measurement('CDF_measurement').add_constraint(['m_W'], NormalDistribution(80.411, 0.008))
    
    OBS_NAMES = ['GammaZ', 'sigma_had', 'm_W', 'GammaW', 'BR(W->enu)', 'BR(W->munu)', 'BR(W->taunu)',
                 'R_e', 'R_mu', 'R_tau', 'R_b', 'R_c', 
                 'R(W->cX)', 'Rmue(W->lnu)', 'Rtaue(W->lnu)', 'Rtaumu(W->lnu)',
                 'AFB(Z->ee)', 'AFB(Z->mumu)', 'AFB(Z->tautau)', 'AFB(Z->bb)','AFB(Z->cc)',
                 'A(Z->ee)', 'A(Z->mumu)', 'A(Z->tautau)', 'A(Z->ss)', 'A(Z->bb)', 'A(Z->cc)' ]
    
    anomaly_sigmas = {}

else:
    # OBS_NAMES = ['GammaZ', 'sigma_had', 'm_W', 'GammaW', 'BR(W->enu)', 'BR(W->munu)', 'BR(W->taunu)',
    #              'R_e', 'R_mu', 'R_tau', # missing 'R_b', 'R_c'
    #              'R(W->cX)', 'Rmue(W->lnu)', 'Rtaue(W->lnu)', 'Rtaumu(W->lnu)', 
    #              'AFB(Z->ee)', 'AFB(Z->mumu)', 'AFB(Z->tautau)', 'AFB(Z->bb)',
    #              'A(Z->ee)', 'A(Z->mumu)', 'A(Z->tautau)', 'A(Z->ss)', # missing 'A(Z->bb)', 'A(Z->cc)'
    #              # new
    #              'BR(KL->pinunu)',  'BR(B+->Knunu)', 'eps_K', 'BR(Bs->mumu)', ("<dBR/dq2>(Bs->phimumu)", 1, 6),
    #              'Rtaul(B->Dlnu)', 'epsp/eps', 'DeltaM_s', 'BR(K+->pinunu)', ("<P5p>(B0->K*mumu)", 4, 6) ]
    
    OBS_NAMES = [
    'GammaZ',
    'sigma_had',
    'R_e',
    'R_mu',
    'R_tau',
    'AFB(Z->ee)',
    'AFB(Z->mumu)',
    'AFB(Z->tautau)',
    'A(Z->ee)',
    'A(Z->mumu)',
    'A(Z->tautau)',
    'GammaW',
    'BR(W->enu)',
    'BR(W->munu)',
    'BR(W->taunu)',
    'R(W->cX)',
    'Rmue(W->lnu)',
    'Rtaue(W->lnu)',
    'Rtaumu(W->lnu)',
    'A(Z->ss)',

    'm_W',          # Delta F=0 
    'AFB(Z->bb)',

    'BR(B+->Knunu)',  # CC
    'Rtaul(B->Dlnu)',

    'epsp/eps', # Delta F=1

    # Binned observables
    ("<P5p>(B0->K*mumu)", 4, 6), # Delta F=1
    ("<dBR/dq2>(Bs->phimumu)", 1, 6),

    'BR(K+->pinunu)', # Delta F=1
    'BR(KL->pinunu)',

    'DeltaM_s', # Delta F=1

    'eps_K', # constraint

    'BR(Bs->mumu)', # constraint
]
    
    anomaly_sigmas = {'m_W': +3.7, 'AFB(Z->bb)': -3.2, 'BR(B+->Knunu)': +4.1, 'Rtaul(B->Dlnu)': +3.5,
                      'epsp/eps': -4.5, "<P5p>(B0->K*mumu)": -3.9,  "<dBR/dq2>(Bs->phimumu)":-3.4,
                      'BR(K+->pinunu)': +4.3, 'BR(KL->pinunu)': +3.6, 'DeltaM_s': -4.8}

# Validate once, not on every call
labels = [obs if isinstance(obs, str) else obs[0] for obs in OBS_NAMES]
missing = [l for l in labels if l not in observables_data]
if missing: raise KeyError(f"Observables not found in loaded JSON data: {missing}")

# --- inject anomalies --- #
# for obs in anomaly_sigmas:
#     if obs not in labels: print(f"invalid anomaly name {obs}")
#     if obs not in observables_data: raise ValueError(f"No ewp_coefficients entry for label: {obs}")
#     data = observables_data[obs]

#     for name in list(flavio.Measurement.instances.keys()):
#         if obs in flavio.Measurement.instances[name].all_parameters :
#             print(f"Removing old measurement for {obs}: {name}"); flavio.Measurement.del_instance(name)

#     sm_val, sm_err = data["sm_prediction"], data["sm_uncertainty"]
#     new = sm_val + anomaly_sigmas[obs] * np.sqrt(2) * sm_err #not ok

#     flavio.Measurement(f'{obs}_override').add_constraint([obs], NormalDistribution(new, sm_err))
#     print(f"({obs}) SM = {sm_val:.4g} +/- {sm_err:.2g}, new = {new:.4g} +/- {sm_err:.2g} ({anomaly_sigmas[obs]} σ)")

# --- load operator vocabulary --- #
for sector_name, sector_data in wcxf.Basis["SMEFT", "Warsaw"].sectors.items():
    if sector_name == "dB=de=dmu=dtau=0": operator_vocab = sorted(sector_data.keys())

n_ops, n_obs = len(operator_vocab), len(OBS_NAMES)
print(f"Loaded {n_ops} operators\n constraints = {n_obs}")
    
def get_experimental_constraint(obs_name):
    """Extracts central value and standard deviation for a given observable from flavio."""
    for name, measurement in flavio.Measurement.instances.items():
        if obs_name not in measurement.all_parameters: continue

        for distribution, parameters in measurement._constraints:
            if obs_name not in parameters: continue
            idx = list(parameters).index(obs_name)
            if hasattr(distribution, "covariance"):
                return float(distribution.central_value[idx]), float(np.sqrt(np.asarray(distribution.covariance)[idx, idx]))
            elif hasattr(distribution, "standard_deviation"):
                central, sigma = distribution.central_value, distribution.standard_deviation
                if np.ndim(central) > 0: central, sigma = np.ravel(central)[idx], np.ravel(sigma)[idx]
                return float(central), float(sigma)
    raise ValueError(f"No experimental constraint found for {obs_name}")

# def _unpack(obs): return (obs,) if isinstance(obs, str) else tuple(obs)
EXP_CENTRAL, EXP_SIGMA, EXP_SIGMA2, SM = [], [], [], []
for i, obs in enumerate(OBS_NAMES):
    label = labels[i]
    sm = observables_data[label]["sm_prediction"]
    sm_err = observables_data[label]["sm_uncertainty"]
    if label in anomaly_sigmas:
        # _, sigma = get_experimental_constraint(obs)
        σ2 = 2 * sm_err**2        # sigma_exp = sm_err not ok
        exp = sm + anomaly_sigmas[label] * np.sqrt( σ2 )
    else:
        exp, sigma = get_experimental_constraint(obs)
        σ2 = sigma**2 + sm_err**2
    
    # exp, sigma = get_experimental_constraint(obs)
    # sigma2 = sigma**2 + sm_err**2
    
    EXP_CENTRAL.append(exp); EXP_SIGMA.append(np.sqrt(σ2)); EXP_SIGMA2.append(σ2), SM.append(sm)

# Fixed ordering of Wilson coefficients (union across all observables)
WC_NAMES = sorted({wc for obs in labels for wc in observables_data[obs]["coefficients"]})
WC_INDEX = {wc: i for i, wc in enumerate(WC_NAMES)}

# Coefficient matrix: shape (n_obs, n_wc), zeros where an observable lacks a coefficient
C = np.zeros((n_obs, len(WC_NAMES)))
for i, obs in enumerate(labels):
    for wc, val in observables_data[obs]["coefficients"].items(): C[i, WC_INDEX[wc]] = val

# SM = np.array([observables_data[obs].get("sm_prediction", 0.0) for obs in labels]) # SM predictions
BASELINE_PULLS = (np.array(SM) - np.array(EXP_CENTRAL)) / np.array(EXP_SIGMA)
CHI2_SM = np.sum(np.square(BASELINE_PULLS))

print(f"chi2_SM = {CHI2_SM:.6f}")
for obs_name, prediction, central, sigma, pull in zip(labels, SM, EXP_CENTRAL, EXP_SIGMA, BASELINE_PULLS):
    print(f"{obs_name:20s} SM={prediction:.6g} exp={central:.6g} sigma={sigma:.6g} pull={pull:.3f}")

# ============================================================
# CHI2 MINIMIZATION
# ============================================================
def flavio_pull_fn(fired_ops):
    """ Given a list of active operators (fired_ops), minimizes chi2 and returns pulls, chi2 """
    fired_ops = [w for w in fired_ops if w in WC_INDEX]
    if len(fired_ops) == 0: return BASELINE_PULLS, CHI2_SM
    
    C_sub = C[:, [WC_INDEX[wc] for wc in fired_ops]] # Columns of C just for fitted WCs: shape (n_obs, n_fired)
    def predict_sub(theta): return SM * (1 + C_sub @ theta)
    def chi2(theta): return np.sum((predict_sub(theta) - EXP_CENTRAL)**2 / EXP_SIGMA2)

    m = Minuit(chi2, np.zeros(len(fired_ops)))
    m.errors = 1e-3; m.limits = (-1.0, 1.0)
    m.migrad()

    pulls = (predict_sub(np.array(m.values)) - EXP_CENTRAL) / EXP_SIGMA
    if not np.all(np.isfinite(pulls)) or np.max(np.abs(pulls)) > 1000: raise ValueError(f"Invalid pulls: {pulls}")

    return pulls, float(m.fval)

     
class SMEFTActorCritic(nn.Module):
    def __init__(self, embed_dim=32, n_heads=4, alpha_init=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.op_to_idx = {op: i for i, op in enumerate(operator_vocab)} # op to integer index
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32)) # Trainable scaling parameters

        # STATE ENCODING
        self.op_embed = nn.Embedding(n_ops, embed_dim) # Operator identity
        self.pull_proj = nn.Linear(1, embed_dim) # Observable pulls
        self.active_emb = nn.Embedding(2, embed_dim) # Active operator status: 0 = not active   1 = active
        self.fired_emb = nn.Embedding(2, embed_dim) # Fired/evaluated status:  0 = not fired   1 = fired

        # Cross-attention between operators and observable pulls
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)

        # Global state featuresvchi2 + step
        self.global_state_encoder = nn.Sequential(nn.Linear(2, embed_dim), nn.Tanh(), nn.Linear(embed_dim, embed_dim))

        # ACTOR : pi_theta(a | s_t) Each op receives: op id, pull/attention info, active stat, fired stat, global state s_t
        self.actor_head = nn.Sequential(nn.Linear(embed_dim * 5, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1))

        # CRITIC : V_phi(s_t) Critic receives pooled representation of SAME state rep used by actor + global state
        self.critic_head = nn.Sequential(nn.Linear(embed_dim * 6, embed_dim), nn.ReLU(), 
                                         nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), 
                                         nn.Linear(embed_dim // 2, 1))

    def forward(self, state, fired_ops=[]):
        """ Returns: logits, V(s). Both actor and critic are computed from the SAME state encoding """

        # STATE ENCODING: guarantees actor and critic receive same state        
        pulls_tensor = torch.tensor(state.pulls, dtype=torch.float32).flatten()

        # MultiheadAttention : query = op tokens, key/value = obs pull tokens
        op_rep = self.op_embed(torch.arange(n_ops))
        pull_tokens = self.pull_proj(pulls_tensor.unsqueeze(-1)).unsqueeze(0)
        attn = self.cross_attn(query=op_rep.unsqueeze(0), key=pull_tokens, value=pull_tokens)[0].squeeze(0)

        active = torch.zeros(n_ops, dtype=torch.long)
        for op in state.active_ops: active[self.op_to_idx[op]] = 1      

        fired = torch.zeros(n_ops, dtype=torch.long)
        fired[fired_ops] = 1
        
        scalar_state = [state.chi2/CHI2_SM, state.step/MAX_STEPS]
        global_rep = self.global_state_encoder(torch.tensor(scalar_state, dtype=torch.float32).unsqueeze(0))

        # ONE SHARED STATE REPRESENTATION
        op_state = torch.cat([op_rep,attn,self.active_emb(active),self.fired_emb(fired),global_rep.expand(n_ops,-1)],dim=-1)
        
        # ACTOR mask operators already tried to avoid picking twice the same
        final_logits = (self.actor_head(op_state).squeeze(-1) * self.alpha).masked_fill(fired.bool(),-1e9) 

        # CRITIC
        state_value = self.critic_head(torch.cat([op_state.mean(dim=0, keepdim=True), global_rep], dim=-1)).squeeze()

        return final_logits, state_value

# ============================================================
# HYPERPARAMETERS AND MODEL INITIALIZATION
# ============================================================
MAX_STEPS, ROLLOUTS_PER_BATCH, N_BATCHES = 10, 25, 300
N = ROLLOUTS_PER_BATCH * MAX_STEPS

DEAD_STEP_PENALTY, ALPHA_STEP, deltachi2_min = 0.5, 0.1, 1e-3 # hyperparams for step rewards used in policy optimization

GAMMA = 0.99
CRITIC_LOSS_COEF = 0.5

model = SMEFTActorCritic(embed_dim=32, n_heads=4, alpha_init=1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

epsilon = 1e-6; momentum = 0.01 # return normalizer hyperparams
entropy_coeff_arr = [0.01 * (1.0 - batch_idx / N_BATCHES) + 0.0001 * batch_idx / N_BATCHES for batch_idx in range(N_BATCHES)]
theory_archive = []

# ============================================================
# TRAINING LOOP
# ============================================================
@dataclass
class SMEFTState:
    pulls: List[float];
    chi2: float
    active_ops: List[str] = field(default_factory=list)
    step: int = 0
  
for batch_idx in range(N_BATCHES):
    model.train()
    returns, batch_delta_chi2, max_traj_rewards = [], [], -np.inf
    log_probs, entropies, values, k = torch.empty(N), torch.empty(N), torch.empty(N), 0
    
    for rollout_idx in range(ROLLOUTS_PER_BATCH):        
        state, evaluated, step_rewards = SMEFTState(BASELINE_PULLS, CHI2_SM, [], 0), [], []
        
        # ROLLOUTS
        for step_idx in range(MAX_STEPS):
            logits, value = model(state, fired_ops=evaluated)
    
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            action_idx = action.item()
    
            log_probs[k], entropies[k], values[k] = dist.log_prob(action), dist.entropy(), value; k+=1
            evaluated.append(action_idx)
    
            # Environment transition: try adding the proposed operator
            proposed_op = operator_vocab[action_idx]
            trial_ops = state.active_ops + [proposed_op]
            trial_pulls, trial_chi2 = flavio_pull_fn(trial_ops)
            delta_chi2 = trial_chi2 - state.chi2
    
            accepted = -delta_chi2 > deltachi2_min
            if accepted: state = SMEFTState(trial_pulls, trial_chi2, trial_ops, step_idx + 1)
            else: state.step = step_idx + 1  # step advances even on rejection
            
            # Compute step rewards
            step_rewards.append(-delta_chi2 - ALPHA_STEP - DEAD_STEP_PENALTY * (accepted==False))        
        
        traj_reward = np.sum(step_rewards)
        max_traj_rewards = max(traj_reward, max_traj_rewards)
        
        delta_chi2 = CHI2_SM - state.chi2
        batch_delta_chi2.append(delta_chi2)
        
        # COMPUTE RETURNS-TO-GO : Monte-Carlo estimate G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...        
        returns.append(torch.tensor(list(accumulate(reversed(step_rewards), lambda running, r: r + GAMMA * running))[::-1], dtype=torch.float32))
        
        # ARCHIVE
        theory_archive.append({"batch":batch_idx, "roll":rollout_idx, "ops":state.active_ops, "pulls":state.pulls.tolist(),
            "traj_reward":traj_reward, "step_rewards":step_rewards, "delta_chi2":delta_chi2})

    returns = torch.cat(returns)
    batch_mean = returns.mean().item(); batch_var = returns.var(unbiased=False).item()
    if batch_idx == 0: mean = batch_mean; var = max(batch_var, epsilon)
    else: mean = (1.0 - momentum)*mean + momentum*batch_mean; var = max((1.0 - momentum)*var + momentum*batch_var, epsilon)
    returns = (returns - mean) / ((var ** 0.5) + epsilon)   # NORMALIZE RETURNS

    # RAW ADVANTAGE G_t is the Monte-Carlo Q estimate. V(s_t) = critic. Therefore: A_t = G_t - V(s_t)
    adv = returns - values.detach() 
    if adv.std()>epsilon: adv = (adv - adv.mean()) / (adv.std() + 1e-8) #NORMALIZE ADVANTAGES

    actor_loss = -(adv * log_probs).mean()
    critic_loss = nn.functional.mse_loss(values, returns)
    entropy_loss = -entropy_coeff_arr[batch_idx] * entropies.mean()

    total_loss = actor_loss + CRITIC_LOSS_COEF * critic_loss + entropy_loss

    # BACKPROP
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f"\n>>> SUMMARY " f"BATCH {batch_idx:03d} <<<\n"
          f"Actor: {actor_loss.item():.4f} | Critic: {critic_loss.item():.4f} | Entropy: {entropy_loss.item():.4f} | "
          f"Mean Deltachi^2: {np.mean(batch_delta_chi2):.2f} | Max Traj Reward: {max_traj_rewards:.1f} ")

with open(Path(__file__).parent / "theory_archive.json", "w") as f: json.dump(theory_archive, f)

# Keep only the best occurrence of each unique accepted operator set and sort by reward
unique_theories = {}
for th in theory_archive:
    key = tuple(sorted(th["ops"])) # Operator ordering should not matter
    if key not in unique_theories or th["traj_reward"] > unique_theories[key]["traj_reward"]: unique_theories[key] = th

print("\n========== TOP 20 UNIQUE THEORIES ==========\n")
for rank, th in enumerate(sorted(unique_theories.values(), key=lambda th: th["traj_reward"], reverse=True)[:20], start=1):
    print( f"{rank}. Reward = {th['traj_reward']:.6f}, Op = {sorted(th['ops'])}, Δχ² = {th['delta_chi2']:.4f}")
    
input(f"Total computation time :{time.time()-debut} sec")
