#### Code Availability

This repository contains all the codes used to reproduce the results presented in [arXiv:2609.10382](https://arxiv.org/abs/2609.10382), as well as a brief description of how they were developed as well as dependencies.

# ReinforcePhysics

### A new method for sequential decision-making for selecting physics interactions with Reinforcement Learning

**ReinforcePhysics** is a reinforcement-learning framework for efficiently searching for new physics interactions (of effective operators) using the **Standard Model Effective Field Theory (SMEFT)**.

The framework formulates the search for SMEFT operators that explain experimental anomalies as a **sequential decision-making problem**. An RL policy proposes SMEFT operators one at a time, while an observable calculator and global-fit procedure evaluate how each proposal improves the agreement between theory and experimental data.

The method is designed to explore the large and highly correlated SMEFT operator space without relying on phenomenological intuition to preselect the relevant operators.

---

## Overview

Finding new physics through experimental anomalies requires determining which combinations of SMEFT operators can explain deviations from Standard Model predictions.

This is a challenging search problem because:

* the SMEFT operator space is very large;
* operators are correlated through quantum corrections;
* a single SMEFT operator can affect many observables;
* viable explanations often involve combinations of operators;
* exhaustive enumeration of operator combinations quickly becomes computationally prohibitive.

ReinforcePhysics addresses this problem using **reinforcement learning**.

Rather than specifying beforehand which operators are relevant, the RL policy learns to navigate the SMEFT operator space using the improvement in the global fit as its reward.

---

## Reinforcement-learning formulation

ReinforcePhysics maps the components of a reinforcement learning problem onto the SMEFT operator search as follows:

| RL component    | SMEFT search                                                        |
| --------------- | ------------------------------------------------------------------- |
| **Policy**      | Decision-making rule used to propose SMEFT operators                |
| **Action**      | Proposing an SMEFT operator                                         |
| **State**       | Proposed operators, observable pulls, and current \(\chi^2_{\min}\) |
| **Environment** | Observable calculator, pulls, \(\chi^2\) function, and minimizer    |
| **Reward**      | Improvement in the global fit                                       |

At each step, the policy proposes an operator. The Wilson coefficient of the proposed operator is fitted to the data, and the resulting change in the minimum \(\chi^2\) determines whether the operator is retained and the reward assigned to the action.

The reward used in the framework is

$$
r_t = \Delta\chi^2 - c_P ,
$$

where \(c_P\) is a penalty applied to ineffective proposals ("dead steps").

---

## Method

The policy is parameterized by a **transformer-based neural network** and trained using an **actor-critic reinforcement-learning algorithm**.

The actor proposes SMEFT operators, while the critic estimates the expected return of the resulting state and provides an advantage estimate to guide policy updates.

The policy sequentially proposes a trajectory of operators:

$$
s_0
\xrightarrow{O_1}
s_1
\xrightarrow{O_2}
s_2
\xrightarrow{\cdots}
s_T .
$$

At every step, the state contains information about the current fit and the operators selected so far. The policy learns to assign larger probabilities to operators that lead to larger improvements in the global fit.

For the setup studied in the paper, the policy searches over **912 baryon- and lepton-number-conserving SMEFT operators** while preserving lepton flavour.

---

## Physics environment

ReinforcePhysics uses a global-fit environment in which:

1. SMEFT operators are proposed by the RL policy.
2. Wilson coefficients are fitted to the experimental data.
3. Theoretical predictions for the observables are calculated.
4. The global \(\chi^2\) is minimized.
5. The state is updated.
6. A reward proportional to the improvement in the fit is returned to the RL agent.

The implementation described in the paper uses **flavio** for observable calculations and \(\chi^2\) evaluation, **wilson for RG running and matching** and **Minuit** for global minimization.

---

---

## Dependencies

ReinforcePhysics relies on scientific and machine-learning software including:

* Python
* PyTorch
* flavio
* Minuit
* NumPy
* SciPy

A complete dependency specification will be provided in `requirements.txt` / `pyproject.toml`.

---

## Getting started

A typical ReinforcePhysics workflow is:

```text
Experimental data
       │
       ▼
  Physics environment
       │
       ▼
   Current state
       │
       ▼
 Transformer policy
       │
       ▼
 Proposed SMEFT operator
       │
       ▼
 Wilson-coefficient fit
       │
       ▼
    χ² minimization
       │
       ▼
     Reward Δχ²
       │
       ▼
  Updated RL policy
       │
       └──────────────► repeat
```

The policy progressively learns which SMEFT operators are most effective at improving the global fit.


---

## Paper

**J. Kumar, M. Bouchard, and D. London,
"Searching for New Physics with Reinforcement Learning"**

ArXiv: [https://arxiv.org/pdf/2609.10382]

---

## Citation

If you use ReinforcePhysics in your research, please cite:

```bibtex
@article{Kumar:2026zpo,
    author = "Kumar, Jacky and Bouchard, Marianne and London, David",
    title = "{Searching for New Physics with Reinforcement Learning}",
    eprint = "2609.10382",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "UdeM-GPP-TH-26-311",
    month = "9",
    year = "2026"
}
```

---

## License

ReinforcePhysics is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

## Acknowledgements

Many thanks to @skumarudel (Sajan Kumar) for being always available for discussions and helping in writing the codes.  

---

## Contact

For questions, suggestions, or collaborations, please open an issue in this repository.

**ReinforcePhysics**
*Reinforcement learning for searching for new physics.*

