import time
import json
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import lsq_linear
import flavio

# from wilson import wcxf
# --- load operator vocabulary --- #
# for sector_name, sector_data in wcxf.Basis["SMEFT", "Warsaw"].sectors.items():
#     if sector_name == "dB=de=dmu=dtau=0": operator_vocab = sorted(sector_data.keys())

debut = time.time()

fictitious_anomalies = True # if False, the analysis will be on the m_W mass anomaly

SEED = 45
np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
print(f"Using seed = {SEED}")

# --- load observable names --- #
if not fictitious_anomalies:
    # Override m_W Measurement
    for name in list(flavio.Measurement.instances.keys()):
        if 'm_W' in flavio.Measurement.instances[name].all_parameters:
            print(f"Removing old m_W measurement: {name}"); flavio.Measurement.del_instance(name)
    
    flavio.Measurement('override').add_constraint(['m_W'], flavio.statistics.probability.NormalDistribution(80.411, 0.008))
    
    OBS_NAMES = ['GammaZ', 'sigma_had', 'm_W', 'GammaW', 'BR(W->enu)', 'BR(W->munu)', 'BR(W->taunu)',
                 'R_e', 'R_mu', 'R_tau', 'R_b', 'R_c', 
                 'R(W->cX)', 'Rmue(W->lnu)', 'Rtaue(W->lnu)', 'Rtaumu(W->lnu)',
                 'AFB(Z->ee)', 'AFB(Z->mumu)', 'AFB(Z->tautau)', 'AFB(Z->bb)','AFB(Z->cc)',
                 'A(Z->ee)', 'A(Z->mumu)', 'A(Z->tautau)', 'A(Z->ss)', 'A(Z->bb)', 'A(Z->cc)' ]
    binned = {}
    
    anomaly_sigmas = {}

else:
    OBS_NAMES = ['GammaZ', 'sigma_had', 'm_W', 'GammaW', 'BR(W->enu)', 'BR(W->munu)', 'BR(W->taunu)',
                 'R_e', 'R_mu', 'R_tau', # missing 'R_b', 'R_c'
                 'R(W->cX)', 'Rmue(W->lnu)', 'Rtaue(W->lnu)', 'Rtaumu(W->lnu)', 
                 'AFB(Z->ee)', 'AFB(Z->mumu)', 'AFB(Z->tautau)', 'AFB(Z->bb)',
                 'A(Z->ee)', 'A(Z->mumu)', 'A(Z->tautau)', 'A(Z->ss)', # missing 'A(Z->bb)', 'A(Z->cc)'
                 # new
                 'BR(KL->pinunu)',  'BR(B+->Knunu)', 'eps_K', 'BR(Bs->mumu)', "<dBR/dq2>(Bs->phimumu)",
                 'Rtaul(B->Dlnu)', 'epsp/eps', 'DeltaM_s', 'BR(K+->pinunu)', "<P5p>(B0->K*mumu)" ]
    
    binned = {"<dBR/dq2>(Bs->phimumu)":("<dBR/dq2>(Bs->phimumu)", 1, 6), "<P5p>(B0->K*mumu)":("<P5p>(B0->K*mumu)", 4, 6)}
    
    anomaly_sigmas = {'m_W': +3.7, 'AFB(Z->bb)': -3.2, 'BR(B+->Knunu)': +4.1, 'Rtaul(B->Dlnu)': +3.5,
                      'epsp/eps': -4.5, "<P5p>(B0->K*mumu)": -3.9,  "<dBR/dq2>(Bs->phimumu)":-3.4,
                      'BR(K+->pinunu)': +4.3, 'BR(KL->pinunu)': +3.6, 'DeltaM_s': -4.8}

# --- load data from json --- #
with open("ewp_linear_coefficients_2.json", "r") as f: observables_data = json.load(f)["observables"]

missing = [l for l in OBS_NAMES if l not in observables_data]
if missing: raise KeyError(f"Observables not found in loaded JSON data: {missing}")
    
def get_experimental_constraint(obs_name):
    """Extracts central value and standard deviation for a given observable from flavio"""
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

operator_vocab = list(sorted({wc for obs in OBS_NAMES for wc in observables_data[obs]["coefficients"]}))
WC_INDEX = {wc: i for i, wc in enumerate(operator_vocab)}

n_obs, n_ops = len(OBS_NAMES), len(operator_vocab)
print(f"\nLoaded {n_ops} operators and {n_obs} constraints")

EXP_CENTRAL, EXP_ERR, SM = np.zeros(n_obs), np.zeros(n_obs), np.zeros(n_obs)
C = np.zeros((n_obs, n_ops))

# --- compute and store observables quantities calculated once all in the same loop --- #
for i, label in enumerate(OBS_NAMES):
    data = observables_data[label]
    
    for wc, val in data["coefficients"].items(): C[i, WC_INDEX[wc]] = val
    sm, sm_err = data["sm_prediction"], data["sm_uncertainty"]
    
    if label in anomaly_sigmas:
        # _, sigma = get_experimental_constraint(obs)
        σ2 = 2 * sm_err**2        # sigma_exp = sm_err not ok
        exp = sm + anomaly_sigmas[label] * np.sqrt( σ2 )
    else:
        exp, sigma = get_experimental_constraint(binned.get(label, label))
        σ2 = sigma**2 + sm_err**2
    
    EXP_CENTRAL[i], EXP_ERR[i], SM[i] = exp, np.sqrt(σ2), data["sm_prediction"]

for name, sm, exp, σ in zip(OBS_NAMES, SM, EXP_CENTRAL, EXP_ERR): print(f"{name:20s} SM={sm:.6g} exp={exp:.6g} err={σ:.6g}")

χ2_SM = np.sum(((SM - EXP_CENTRAL)/ EXP_ERR)**2)
print(f"\nχ2_SM = {χ2_SM:.6f}\n")

# --- CHI2 MINIMIZATION --- #
A = (SM[:, None] * C ) / EXP_ERR[:, None]   # (n_obs, n_ops)
y = (EXP_CENTRAL - SM) / EXP_ERR            # chi2(theta) = |A @ theta - y|^2

# the normalization is never used. why bounds -1,1?
@lru_cache(maxsize=None)
def _chi2(cols): return 2 * lsq_linear(A[:, cols], y, bounds=(-1, 1)).cost # cols: sorted tuple of column indices
def chi2_fn(active_ops): return _chi2(tuple(active_ops)) if active_ops else χ2_SM

# --- ACTOR MODEL --- #
class ActorModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.op_embed = nn.Embedding(n_ops, embed_dim) # Operator identity
        self.actor_head  = nn.Sequential(nn.Linear(embed_dim + 2, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1))
    
    def forward(self, χ2, active_ops):
        active = torch.zeros(n_ops, dtype=torch.bool)
        active[active_ops] = True
        op_state = torch.cat([self.op_embed.weight, active.unsqueeze(-1).float(), torch.full((n_ops,1), χ2/χ2_SM)], dim=-1)
   
        return self.actor_head(op_state).squeeze(-1).masked_fill(active, -1e9) # avoid picking same op twice

# --- HYPERPARAMETERS AND MODEL INITIALIZATION --- #
N_STEPS = 10; N_ROLLOUTS = 25; N_BATCHES = 300

model = ActorModel(embed_dim = 32)
optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-3, weight_decay = 1e-4)
unique_th = {} # store unique operator sets and corresponding χ2 drop

# --- GREEDY ALGORITHM BENCHMARK --- #
op_selected = []
for i in range(N_STEPS):
    min_χ2 = np.inf
    for op in range(n_ops):
        if chi2_fn(op_selected + [op]) < min_χ2: min_χ2 = chi2_fn(op_selected + [op]); best_op = op
    op_selected.append(best_op)

print(f"\ngreedy algorithm benchmark (found in {time.time()-debut} sec ):")
print(f"operators found in : {[operator_vocab[op] for op in op_selected]}\n Δχ2 = {χ2_SM - chi2_fn(op_selected)}")

# --- TRAINING LOOP --- #
for batch in range(N_BATCHES):
    batch_Δχ2, log_probs, rewards = [], torch.empty((N_ROLLOUTS, N_STEPS)), torch.empty((N_ROLLOUTS, N_STEPS))
    for roll in range(N_ROLLOUTS):
        χ2, active_ops = χ2_SM, []

        for step in range(N_STEPS):
            dist = torch.distributions.Categorical(logits=model(χ2, active_ops))
            action = dist.sample()
            log_probs[roll,step] = dist.log_prob(action)

            active_ops += [action.item()]
            new_χ2 = chi2_fn(sorted(active_ops))
            rewards[roll,step] = χ2 - new_χ2
            χ2 = new_χ2
            
        Δχ2 = χ2_SM - χ2
        batch_Δχ2.append(Δχ2)
        ops = tuple(operator_vocab[op] for op in active_ops)
        if Δχ2 > unique_th.get(ops, -np.inf): unique_th[ops] = Δχ2
            
    R = rewards.flip(1).cumsum(1).flip(1) # returns
    adv = (R - R.mean(0)) /(R.std(unbiased = False) + 1e-6) # normalize advantage

    loss = - ( adv * log_probs ).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"\n>>> BATCH {batch:03d} <<<   loss : {loss.item():.4f} || Max Δχ2 : {max(batch_Δχ2):.1f}")

n = 0
for d in unique_th.values():
    if d > 150: n+=1
print(f"\n {n} traj found with Δχ2 > 150")

print("\n========== TOP 20 UNIQUE THEORIES ==========\n")
for ops, d in sorted(unique_th.items(), key=lambda kv: kv[1], reverse=True)[:20]:print(f"- Ops = {ops}, Δχ² = {d:.1f}")
input(f"Total computation time :{time.time()-debut} sec")
