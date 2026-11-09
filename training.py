import time
from itertools import combinations  
import matplotlib.pyplot as plt
from scipy import stats

import numpy as np
import torch
import torch.nn as nn

from data import chi2_fn, operator_vocab, WC_INDEX, n_obs, n_ops, χ2_SM, fictitious_anomalies

debut = time.time()
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
print(f"Using seed = {SEED}")

# fictitious_anomalies = True # if False, the analysis will be on the m_W mass anomaly
plot_learning_curve = True
random_baseline = False

# case specific hyperparams 
if fictitious_anomalies:
    # define minimal solution
    cut_all, N = 140, 6
    N_BATCHES = 500
    
else:
    # define minimal solution
    cut_all, N = 20, 2
    N_BATCHES = 200
    
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
N_STEPS = 10; N_ROLLOUTS = 25
p_val = 0.1

model = ActorModel(embed_dim = 64)
# initialize weights and bias to 0 so initially every operator has the same chance of being selected
nn.init.zeros_(model.actor_head[-1].weight); nn.init.zeros_(model.actor_head[-1].bias)

optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 1e-4)
unique_th = {} # store unique operator sets and corresponding χ2 drop
theory_archive = []
unique_subsets, evaluated = {}, set() # store unique subsets of N operators with p-value >= p_val

# --- RANDOM SEARCH BENCHMARK --- #
if random_baseline:
    print("\nTesting with random search\n")
    unique_random, tested_random, lowest_chi2 = {}, set(), np.inf
    rng = np.random.default_rng(SEED)
    # evaluate as many trajectories as the RL algorithm
    for i in range(N_ROLLOUTS*N_BATCHES):
        test10 = tuple(sorted(rng.choice(n_ops, size=(N_STEPS), replace=False).tolist()))
        if χ2_SM - chi2_fn(test10)<(cut_all/1.5) : continue # avoid testing bad sets
        for subset in list(combinations(test10, N)):
            subset = tuple(sorted(subset))
            if subset in tested_random: continue
            tested_random.add(subset)
            lowest_chi2 = min(lowest_chi2, chi2_fn(subset))
            if subset not in unique_random and stats.chi2.sf(chi2_fn(subset), n_obs-N) > p_val:
                print("a minimal theory found")
                unique_random[subset] = χ2_SM - chi2_fn([WC_INDEX[o] for o in subset]) 

    print("The random search found ", len(unique_random), "trajectories")
    print("The lowest χ2 is ", lowest_chi2, "\nThis corresponds to a p-value of ", stats.chi2.sf(lowest_chi2, n_obs-N))

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
            rewards[roll,step] = (χ2 - new_χ2)#**2
            χ2 = new_χ2
            
        Δχ2 = χ2_SM - χ2
        batch_Δχ2.append(Δχ2)
        ops = tuple(sorted([operator_vocab[op] for op in active_ops]))
        
        # store each trajectory reward and ops for plotting learning curve
        theory_archive.append({"batch":batch, "ops":ops, "delta_chi2": Δχ2 })

        # new trajectory
        if Δχ2 > unique_th.get(ops, -np.inf): 
            unique_th[ops] = Δχ2
            if Δχ2 > cut_all: # check only high reward trajectories
                for subset in list(combinations(ops, N)):
                    subset = tuple(sorted(subset))
                    if subset not in evaluated:
                        evaluated.add(subset)
                        chi2_reduce = chi2_fn(sorted([WC_INDEX[o] for o in subset]))
                        if stats.chi2.sf(chi2_reduce, n_obs-N) >= p_val:
                            print("a minimal theory found")
                            unique_subsets[subset] = χ2_SM - chi2_reduce
        
        
    R = rewards.flip(1).cumsum(1).flip(1) # returns (eq. (2))
    adv = (R - R.mean(0)) /(R.std(unbiased = False) + 1e-6) # normalized returns

    loss = - ( adv * log_probs ).mean() # loss = negative log likelihood (eq. (1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f">>> BATCH {batch:03d} <<<   loss : {loss.item():.4f} || Max Δχ2 : {max(batch_Δχ2):.1f}")

for ops, drop in unique_subsets.items(): print(ops, drop)
print(f"\n{len(unique_subsets)} minimal theor{'y' if len(unique_subsets)==1 else 'ies'} found with p-val >= {p_val*100}%")

print("\n========== TOP 10 UNIQUE THEORIES ==========\n")
for ops, d in sorted(unique_th.items(), key=lambda kv: kv[1], reverse=True)[:10]:print(f"- Ops = {ops}, Δχ² = {d:.1f}")

if plot_learning_curve:
    y2 = [0]*N_BATCHES
    for th2 in theory_archive: y2[th2["batch"]] += th2["delta_chi2"]/N_ROLLOUTS
    
    plt.plot(range(N_BATCHES), y2, alpha = 0.5, color = "black")
    plt.plot(range(N_BATCHES), [np.mean(y2[max(0, i-5):min(N_BATCHES,i+5)]) for i in range (N_BATCHES)], color = "black")
    plt.grid(); plt.ylim(40,160); plt.xlabel("Batch"); plt.ylabel("Trajectory reward"); plt.title("Learning curve")
    plt.savefig("plot.pdf")
    
    plt.show()

input(f"Total computation time :{int(time.time()-debut)} sec")
