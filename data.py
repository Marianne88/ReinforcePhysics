import json
from functools import lru_cache

import numpy as np
from scipy.optimize import lsq_linear
import flavio
from wilson import wcxf


fictitious_anomalies = True # if False, the analysis will be on the m_W mass anomaly

# --- load operator vocabulary --- #
for sector_name, sector_data in wcxf.Basis["SMEFT", "Warsaw"].sectors.items():
    if sector_name == "dB=de=dmu=dtau=0": operator_vocab = sorted(sector_data.keys())

# SEED = 42
# np.random.seed(SEED)

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
                 'AFB(Z->ee)', 'AFB(Z->mumu)', 'AFB(Z->tautau)', 'AFB(Z->bb)', 'AFB(Z->cc)', #
                 'A(Z->ee)', 'A(Z->mumu)', 'A(Z->tautau)', 'A(Z->ss)', 'A(Z->bb)', 'A(Z->cc)' ]
    binned = {}
    
    anomaly_sigmas = {}

else:
    OBS_NAMES = ['GammaZ', 'sigma_had', 'm_W', 'GammaW', 'BR(W->enu)', 'BR(W->munu)', 'BR(W->taunu)',
                 'R_e', 'R_mu', 'R_tau', # missing 'R_b', 'R_c'
                 'R(W->cX)', 'Rmue(W->lnu)', 'Rtaue(W->lnu)', 'Rtaumu(W->lnu)', 
                 'AFB(Z->ee)', 'AFB(Z->mumu)', 'AFB(Z->tautau)', 'AFB(Z->bb)', # missing 'AFB(Z->cc)'
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

# limit the operator vocabulary to the operators that contribute to the chosen observables
# operator_vocab = list(sorted({wc for obs in OBS_NAMES for wc,v in observables_data[obs]["coefficients"].items()}))
WC_INDEX = {wc: i for i, wc in enumerate(operator_vocab)}

n_obs, n_ops = len(OBS_NAMES), len(operator_vocab)
print(f"\nLoaded {n_ops} operators and {n_obs} constraints")

EXP_CENTRAL, EXP_ERR, SM = np.zeros(n_obs), np.zeros(n_obs), np.zeros(n_obs)
C = np.zeros((n_obs, n_ops))

# --- compute and store observables quantities calculated once all in the same loop --- #
for i, label in enumerate(OBS_NAMES):
    data = observables_data[label]
    
    for wc, val in data["coefficients"].items():
        if wc in operator_vocab: C[i, WC_INDEX[wc]] = val 
    
    sm, sm_err = data["sm_prediction"], data["sm_uncertainty"]
        
    if label in anomaly_sigmas:
        σ2 = 2*sm_err**2        # sigma_exp = sm_err
        exp = sm + anomaly_sigmas[label] * np.sqrt( σ2 )
    
    else: 
        exp, sigma = get_experimental_constraint(binned.get(label, label))
        σ2 = sigma**2 + sm_err**2 # include SM prediction error in experimental value error to take it into account
    
    EXP_CENTRAL[i], EXP_ERR[i], SM[i] = exp, np.sqrt(σ2), sm
    print(f"{label:25s} SM={SM[i]:.6g} exp=({EXP_CENTRAL[i]:.6g} +/- {EXP_ERR[i]:.2g}), deviation from SM = {(exp-sm)/np.sqrt(σ2)}")

χ2_SM = np.sum(((SM - EXP_CENTRAL)/ EXP_ERR)**2)
print(f"\nχ2_SM = {χ2_SM:.2f}\n")

# --- CHI2 MINIMIZATION --- #

A = (SM[:, None] * C ) / EXP_ERR[:, None]   # (n_obs, n_ops)
y = (EXP_CENTRAL - SM) / EXP_ERR            # chi2(theta) = |A @ theta - y|^2

@lru_cache(maxsize=None)
def _chi2(cols): return 2 * lsq_linear(A[:, cols], y, bounds=(-1, 1)).cost # cols: sorted tuple of column indices
def chi2_fn(active_ops): return _chi2(tuple(active_ops)) if active_ops else χ2_SM
