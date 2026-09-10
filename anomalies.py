# flavio_overrides.py
import flavio
from flavio.statistics.probability import NormalDistribution
#from smeft_new.pulls_linear_10 import ewp_coefficients, obs_correspondence  # adjust import path as needed
import json


obs_correspondence = {
    'GammaZ': 'GammaZ',
    'sigma_had': 'sigma_had',
    'R_e': 'R_e',
    'R_mu': 'R_mu',
    'R_tau': 'R_tau',
    'AFB(Z->ee)': 'AFB(Z->ee)',
    'AFB(Z->mumu)': 'AFB(Z->mumu)',
    'AFB(Z->tautau)': 'AFB(Z->tautau)',
    'A(Z->ee)': 'A(Z->ee)',
    'A(Z->mumu)': 'A(Z->mumu)',
    'A(Z->tautau)': 'A(Z->tautau)',
    'GammaW': 'GammaW',
    'BR(W->enu)': 'BR(W->enu)',
    'BR(W->munu)': 'BR(W->munu)',
    'BR(W->taunu)': 'BR(W->taunu)',
    'R(W->cX)': 'R(W->cX)',
    'Rmue(W->lnu)': 'Rmue(W->lnu)',
    'Rtaue(W->lnu)': 'Rtaue(W->lnu)',
    'Rtaumu(W->lnu)': 'Rtaumu(W->lnu)',
    'A(Z->ss)': 'A(Z->ss)',

    'm_W': 'm_W',          # Delta F=0 
    'AFB(Z->bb)': 'AFB(Z->bb)',

    'BR(B+->Knunu)': 'BR(B+->Knunu)',  # CC
    'Rtaul(B->Dlnu)': 'Rtaul(B->Dlnu)',

    'epsp/eps': 'epsp/eps', # Delta F=1

    # Binned observables
    ("<P5p>(B0->K*mumu)", 4, 6): 'P5p46', # Delta F=1
    ("<dBR/dq2>(Bs->phimumu)", 1, 6): 'Bsphimumu16',

    'BR(K+->pinunu)': 'BR(K+->pinunu)', # Delta F=1
    'BR(KL->pinunu)': 'BR(KL->pinunu)',

    'DeltaM_s': 'DeltaM_s', # Delta F=1

    'eps_K': 'eps_K', # constraint

    'BR(Bs->mumu)': 'BR(Bs->mumu)', # constraint
}
                       

# Path to pre-computed SMEFT linear expansion coefficients
COEFFICIENTS_PATH = (
    "/home/kumarj/links/projects/def-london/kumarj/gitrepos/"
    "smeft-anomaly/smeft_new/ewp_linear_coefficients_2.json"
)

# ============================================================
# 3. LOAD LINEAR SMEFT COEFFICIENTS
# ============================================================

with open(COEFFICIENTS_PATH, "r", encoding="utf-8") as f:
    ewp_data = json.load(f)

ewp_coefficients = ewp_data["observables"]


def inject_anomaly(flavio_obs, n_sigma, exp_unc_frac=None):
    if flavio_obs not in obs_correspondence:
        raise ValueError(f"Unknown observable: {flavio_obs}")

    label = obs_correspondence[flavio_obs]
    obs_name = flavio_obs if isinstance(flavio_obs, str) else flavio_obs[0]

    if label not in ewp_coefficients:
        raise ValueError(f"No ewp_coefficients entry for label: {label}")

    data = ewp_coefficients[label]

    for name in list(flavio.Measurement.instances.keys()):
        m = flavio.Measurement.instances[name]
        if flavio_obs in m.all_parameters or obs_name in m.all_parameters:
            print(f"Removing old measurement for {obs_name}: {name}")
            flavio.Measurement.del_instance(name)

    sm_val = data["sm_prediction"]
    sm_unc = data["sm_uncertainty"]

    exp_unc = exp_unc_frac * abs(sm_val) if exp_unc_frac else sm_unc
    combined_unc = (sm_unc**2 + exp_unc**2) ** 0.5
    new_central = sm_val + n_sigma * combined_unc

    new_name = f'my_{label}_override'
    meas = flavio.Measurement(new_name)
    meas.add_constraint([flavio_obs], NormalDistribution(new_central, exp_unc))

    print(f"{label:15s} (flavio: {obs_name}) "
          f"SM = {sm_val:.4g} +/- {sm_unc:.2g}, "
          f"new central = {new_central:.4g} +/- {exp_unc:.2g}  "
          f"({n_sigma:+.2f} sigma, fixed)")

    return new_name


anomaly_sigmas = {
    'm_W':          +3.7,
    'AFB(Z->bb)':   -3.2,
    'BR(B+->Knunu)': +4.1,
    'Rtaul(B->Dlnu)': +3.5,
    'epsp/eps':     -4.5,
    'P5p46':        -3.9,
    'Bsphimumu16':  -3.4,
    'BR(K+->pinunu)': +4.3,
    'BR(KL->pinunu)': +3.6,
    'DeltaM_s':     -4.8,
}

def apply_overrides():
    overrides = {}
    for flavio_obs, label in obs_correspondence.items():
        if label in anomaly_sigmas:
            overrides[label] = inject_anomaly(flavio_obs, anomaly_sigmas[label])
    return overrides

_overrides = apply_overrides()
