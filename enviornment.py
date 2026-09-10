import json
import math
import numpy as np
import flavio
import smeft_new.flavio_overrides  # one line, applies all anomalies immediately
from wilson import Wilson
from iminuit import Minuit
#from flavio.statistics.likelihood import FastLikelihood

# ============================================================
# 1. CONFIGURATION & EXPERIMENTAL CONSTRAINTS
# ============================================================

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

constraints_ewp = keys = list(obs_correspondence.values())
constraints_ewp_exp = list(obs_correspondence.keys())


OBS_NAMES = constraints_ewp
N_OBS = len(constraints_ewp)


SIGMA_TARGET = 1.0
EXPONENT_CLAMP = 50.0

# Path to pre-computed SMEFT linear expansion coefficients
COEFFICIENTS_PATH = (
#    "/home/kumarj/links/scratch/tmp_real/ewp_linear_coefficients_quad_fixed.json"
    "/home/kumarj/links/scratch/tmp_real/ewp_linear_coefficients_2.json"
)

# ============================================================
# 3. LOAD LINEAR SMEFT COEFFICIENTS
# ============================================================

with open(COEFFICIENTS_PATH, "r", encoding="utf-8") as f:
    ewp_data = json.load(f)

ewp_coefficients = ewp_data["observables"]
ewp_coefficients_meta = ewp_data["metadata"]

# ============================================================
# 4. KINEMATIC & PREDICTION HELPERS
# ============================================================

def predict_ewp(obs: str, wc_dict: dict, ewp_data: dict) -> float:
    """
    Computes linearized SMEFT prediction for a single observable, per the
    linear-only JSON schema:

        O/O_SM = 1 + sum_i coefficient_i * (NORMALIZE * C_i)

    Parameters
    ----------
    obs : str
        Observable name, must be a key in ewp_data["observables"].
    wc_dict : dict
        Wilson coefficients {operator_name: value}, in the *raw*
        (un-normalized) convention -- normalization is applied here.
    ewp_data : dict
        The full parsed JSON (contains "metadata" and "observables").
    """
    observables = ewp_data.get("observables", {})
    if obs not in observables:
        raise KeyError(f"Observable '{obs}' not found in loaded JSON data.")

    data = observables[obs]
    metadata = ewp_data.get("metadata", {})
    normalize = metadata.get("normalize", 1.0)

    sm = data.get("sm_prediction", 0.0)
    coeffs = data.get("coefficients", {})

    # Scale each relevant Wilson coefficient by NORMALIZE once
    scaled_wc = {
        wc: val #* normalize
        for wc, val in wc_dict.items()
        if wc in coeffs
    }

    linear_term = sum(
        coeffs[wc] * scaled_wc[wc]
        for wc in scaled_wc
    )

    delta = linear_term
    return sm * (1 + delta)

def predict_observables(wc_dict: dict) -> dict:
    """Computes predictions for all observables in constraints_ewp."""
    return {obs: predict_ewp(obs, wc_dict, ewp_data) for obs in constraints_ewp}

# ============================================================
# EXPERIMENTAL CONSTRAINTS & CHI2 HELPERS
# ============================================================

obs_correspondence_reverse = {
    value: key
    for key, value in obs_correspondence.items()
}


def get_experimental_constraint(obs_name: str) -> tuple:
    """
    Extract an effective experimental central value and 1-sigma
    uncertainty from Flavio.

    Returns
    -------
    central, sigma : tuple(float, float)

    Handles:
        - covariance/Gaussian distributions
        - NormalDistribution
        - AsymmetricNormalDistribution
        - NumericalDistribution
        - MultivariateNumericalDistribution
        - upper-limit distributions such as GeneralGammaUpperLimit

    For asymmetric distributions, sigma is chosen according to the
    side of the prediction only in the chi2 calculation; this function
    returns the average of the +/- 1 sigma errors.

    For upper limits, an effective sigma is constructed from the
    95% upper limit assuming a Gaussian centered at zero.
    """

    # ------------------------------------------------------------
    # Convert RL/internal observable name -> Flavio observable
    # ------------------------------------------------------------
    if obs_name in obs_correspondence_reverse:
        obs_name = obs_correspondence_reverse[obs_name]
    # ------------------------------------------------------------
    # Allow binned observable input:
    #
    # ("<P5p>(B0->K*mumu)", 4, 6)
    # ------------------------------------------------------------

    if isinstance(obs_name, tuple):

        if len(obs_name) != 3:
            raise ValueError(
                f"Invalid observable tuple: {obs_name}"
            )

        obs, q2min, q2max = obs_name

        target = (obs, q2min, q2max)

    else:

        target = obs_name

    # ------------------------------------------------------------
    # Search Flavio measurements
    # ------------------------------------------------------------

    for name, measurement in flavio.Measurement.instances.items():

        for distribution, parameters in measurement._constraints:

            if target not in parameters:
                continue

            idx = parameters.index(target)

            # ====================================================
            # 1. Covariance / Gaussian distribution
            # ====================================================

            if hasattr(distribution, "covariance"):

                central = np.asarray(
                    distribution.central_value
                )

                covariance = np.asarray(
                    distribution.covariance
                )

                central_value = float(
                    central[idx]
                )

                sigma = float(
                    np.sqrt(covariance[idx, idx])
                )

                return central_value, sigma

            # ====================================================
            # 2. Ordinary distribution with standard_deviation
            # ====================================================

            if hasattr(
                distribution,
                "standard_deviation"
            ):

                central = distribution.central_value
                sigma = distribution.standard_deviation

                if isinstance(
                    central,
                    (list, tuple, np.ndarray)
                ):
                    central = central[idx]

                if isinstance(
                    sigma,
                    (list, tuple, np.ndarray)
                ):
                    sigma = sigma[idx]

                return (
                    float(central),
                    float(sigma),
                )

            # ====================================================
            # 3. Asymmetric / numerical distributions
            # ====================================================

            if hasattr(distribution, "ppf"):

                try:

                    q16 = distribution.ppf(
                        0.158655254
                    )

                    q50 = distribution.ppf(
                        0.5
                    )

                    q84 = distribution.ppf(
                        0.841344746
                    )

                    # Handle vector distributions
                    if np.ndim(q16) > 0:
                        q16 = q16[idx]

                    if np.ndim(q50) > 0:
                        q50 = q50[idx]

                    if np.ndim(q84) > 0:
                        q84 = q84[idx]

                    sigma_minus = (
                        float(q50) - float(q16)
                    )

                    sigma_plus = (
                        float(q84) - float(q50)
                    )

                    sigma = 0.5 * (
                        sigma_minus + sigma_plus
                    )

                    return (
                        float(q50),
                        float(sigma),
                    )

                except Exception:
                    pass

            # ====================================================
            # 4. Upper-limit distributions
            # ====================================================

            distribution_name = (
                type(distribution).__name__
            )

            if "UpperLimit" in distribution_name:

                # ------------------------------------------------
                # Try to obtain the upper limit directly
                # ------------------------------------------------

                upper_limit = None

                # Some Flavio distributions expose an upper_limit
                # attribute.
                if hasattr(
                    distribution,
                    "upper_limit"
                ):

                    try:
                        upper_limit = float(
                            distribution.upper_limit
                        )
                    except Exception:
                        pass

                # ------------------------------------------------
                # Otherwise infer it from the 95% quantile
                # ------------------------------------------------

                if upper_limit is None and hasattr(
                    distribution,
                    "ppf"
                ):

                    try:

                        upper_limit = float(
                            distribution.ppf(0.95)
                        )

                    except Exception:
                        pass

                # ------------------------------------------------
                # Convert 95% upper limit to effective 1 sigma
                #
                # For a Gaussian centered at zero:
                #
                #       95% CL = 1.645 sigma
                #
                # ------------------------------------------------

                if (
                    upper_limit is not None
                    and np.isfinite(upper_limit)
                    and upper_limit > 0
                ):

                    sigma = (
                        upper_limit / 1.645
                    )

                    return (
                        0.0,
                        float(sigma),
                    )

                # ------------------------------------------------
                # If we cannot determine the limit
                # ------------------------------------------------

                raise ValueError(
                    f"Could not determine upper limit "
                    f"for {obs_name} "
                    f"using distribution "
                    f"{distribution_name}"
                )

    # ------------------------------------------------------------
    # Nothing found
    # ------------------------------------------------------------

    raise ValueError(
        f"No experimental constraint found "
        f"for {obs_name}"
    )

def compute_chi2(predictions: dict) -> float:
    """Computes total chi2 for a set of predictions against experimental data."""
    chi2 = 0.0
    for obs, prediction in predictions.items():
        data = ewp_coefficients.get(obs, {})

        # Experimental central value + uncertainty (from flavio measurements,
        # including your overridden ones)
        exp, exp_sigma = get_experimental_constraint(obs)

        # Theory uncertainty from the linearized SMEFT coefficients JSON
        theory_sigma = data.get("sm_uncertainty", 0.0)

        # Combine in quadrature — same convention as compute_baseline_pulls
        sigma = (theory_sigma**2 + exp_sigma**2) ** 0.5

        chi2 += ((prediction - exp) / sigma) ** 2

    return chi2

def _compute_chi2_SM() -> float:
    """Compute baseline SM chi2 using zero Wilson coefficients."""
    predictions = predict_observables({})
    return float(compute_chi2(predictions))


# Run after helper functions are defined
CHI2_SM: float = _compute_chi2_SM()
print(f"[chi2_reward] chi2_SM = {CHI2_SM:.6f}")

def compute_baseline_pulls(obs_names):
    """
    Compute SM baseline pulls using the linearized prediction data and
    flavio experimental constraints. Combines theory and experimental
    uncertainties in quadrature.
    """
    pulls = []

    for obs_name in obs_names:
        if obs_name not in ewp_coefficients:
            raise ValueError(f"Unknown observable: {obs_name}")

        data = ewp_coefficients[obs_name]

        # SM prediction from JSON or linear prediction at zero WC
        prediction = data.get("sm_prediction", predict_ewp(obs_name, {}, ewp_data))

        # SM theory uncertainty from JSON, falling back to flavio's
        # own SM uncertainty if not precomputed/stored
        theory_sigma = data["sm_uncertainty"]

        # Retrieve experimental central value and uncertainty from flavio
        # (This correctly accounts for your custom mW and AFB measurements)
        central, exp_sigma = get_experimental_constraint(obs_name)

        # Combine theory + experimental uncertainty in quadrature
        sigma = (theory_sigma**2 + exp_sigma**2) ** 0.5

        # Compute SM pull
        pull = (prediction - central) / sigma
        pulls.append(float(pull))

        print(
            f"{obs_name:20s} "
            f"SM={prediction:.6g} "
            f"exp={central:.6g} "
            f"theory_sig={theory_sigma:.4g} "
            f"exp_sig={exp_sigma:.4g} "
            f"sigma={sigma:.6g} "
            f"pull={pull:.3f}"
        )

    return pulls
BASELINE_PULLS = compute_baseline_pulls(constraints_ewp)

# ============================================================
# 5. MINIMIZATION & PULL EVALUATION
# ============================================================

def n_dof(n_ops: int) -> int:
    return max(N_OBS - n_ops, 1)


def _build_chi2_fn(wc_names: list):
    n = len(wc_names)

    def chi2(*values):
        try:
            wc_dict = {wc_names[i]: float(values[i]) for i in range(n)}

            if any(abs(v) > 1e7 for v in wc_dict.values()):
                return 1e20

            predictions = predict_observables(wc_dict)
            result = compute_chi2(predictions)

            if not np.isfinite(result):
                return 1e20

            return float(result)

        except Exception as e:
            print(f"[chi2] Exception: {e}")
            return 1e20

    fn_src = (
        f"def chi2_named({', '.join(wc_names)}):\n"
        f"    return chi2({', '.join(wc_names)})\n"
    )

    globs = {"chi2": chi2}
    exec(fn_src, globs)
    return globs["chi2_named"]

"""
def _minimise_chi2(wc_names: list) -> tuple:
    chi2_fn = _build_chi2_fn(wc_names)
    start = {name: 0.0 for name in wc_names}

    m = Minuit(chi2_fn, **start)
    for name in wc_names:
        m.errors[name] = 1e-3

    m.migrad()
    m.hesse()

    chi2_min = float(m.fval)
    best_wc = {k: float(v) for k, v in zip(wc_names, m.values)}
    return chi2_min, best_wc
"""
def _minimise_chi2(wc_names: list) -> tuple:

    chi2_fn = _build_chi2_fn(wc_names)

    # Initial values
    start = {
        name: 0.0
        for name in wc_names
    }

    m = Minuit(chi2_fn, **start)

    # --------------------------------------------------------
    # Set parameter errors and bounds
    # --------------------------------------------------------

    for name in wc_names:

        # Initial step size
        m.errors[name] = 1e-3

        # Restrict WC to [-1, 1]
        m.limits[name] = (-1.0, 1.0)

    # --------------------------------------------------------
    # Minimize
    # --------------------------------------------------------

    m.migrad()

    # Estimate uncertainties
    m.hesse()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    chi2_min = float(m.fval)

    best_wc = {
        k: float(v)
        for k, v in zip(wc_names, m.values)
    }

    return chi2_min, best_wc

def flavio_pull_fn(fired_ops: list) -> tuple:
    """
    Given a list of active operators (fired_ops), minimizes chi2
    and returns predictions, pulls, and the best-fit Wilson coefficients.
    """
    if len(fired_ops) == 0:
        best_wc = {}
        chi2_min = CHI2_SM
    else:
        chi2_min, best_wc = _minimise_chi2(fired_ops)

    prediction_dict = predict_observables(best_wc)

    predictions = [
        float(prediction_dict[obs_name])
        for obs_name in constraints_ewp
    ]

    pulls = []
    for obs_name, prediction in zip(constraints_ewp, predictions):
        central, exp_sigma = get_experimental_constraint(obs_name)

        # SM theory uncertainty from the linearized coefficients data
        theory_sigma = ewp_coefficients[obs_name]["sm_uncertainty"]

        # Combine theory + experimental uncertainty in quadrature
        sigma = (theory_sigma**2 + exp_sigma**2) ** 0.5

        pull = (prediction - central) / sigma
        pulls.append(float(pull))

    if not np.all(np.isfinite(pulls)):
        raise ValueError(f"Non-finite pulls encountered: {pulls}")

    if np.max(np.abs(pulls)) > 10000:
        raise ValueError(f"Unphysical pull detected: {pulls}")

    return (
        OBS_NAMES,
        pulls,
        float(chi2_min),
        predictions,
        best_wc,
    )


import numpy as np
import flavio
from wilson import Wilson

# ==============================================================================
# 1. SAMPLING & FLAVIO EVALUATION FUNCTIONS
# ==============================================================================

def get_available_wcs():
    """Extract all Wilson Coefficient names present in the linear dictionary."""
    wc_set = set()
    for obs, data in ewp_coefficients.items():
        # Check all standard coefficient keys in JSON structure
        coeffs = (
            data.get("absolute_coefficients") or 
            data.get("coefficients") or 
            data.get("relative_coefficients") or 
            {}
        )
        wc_set.update(coeffs.keys())

    wcs = sorted(list(wc_set))
    if not wcs:
        print("\n[WARNING] No WC names found! Inspecting first JSON entry:")
        if ewp_coefficients:
            sample_obs = list(ewp_coefficients.keys())[0]
            print(f"Keys inside ewp_coefficients['{sample_obs}']:", list(ewp_coefficients[sample_obs].keys()))
    return wcs


def generate_random_wcs(wc_names, n_ops=5, scale_min=-1.0, scale_max=1.0):
    """
    Generates a dictionary with Order(1) Wilson coefficients for linear expansions.
    """
    if not wc_names:
        print("[WARNING] wc_names is empty. Returning empty WC dict.")
        return {}

    selected = np.random.choice(wc_names, size=min(n_ops, len(wc_names)), replace=False)
    values = np.random.uniform(scale_min, scale_max, size=len(selected))
    return {str(wc): float(val) for wc, val in zip(selected, values)}

