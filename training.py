import sys

from pathlib import Path

import copy

import random

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

import contextlib

sys.path.insert(

    0,

    "/home/kumarj/links/projects/def-london/kumarj/gitrepos/smeft-anomaly"

)

from smeft_new.pulls_linear_10 import (

    constraints_ewp,

    CHI2_SM,

    flavio_pull_fn,

    BASELINE_PULLS ,

)


OBS_NAMES = constraints_ewp

from smeft_new.state import SMEFTState, step

from smeft_new.helpers import print_policy_diagnostics

#from smeft_new.sensitivity_matrix import M_ij


# ------------------------------------------------------------

# LOAD OPERATOR VOCABULARY

# ------------------------------------------------------------

import argparse

import warnings

import flavio

import wilson

from wilson import wcxf

warnings.filterwarnings("ignore")

def _build_catalogue_from_wcxf() -> dict[str, list[str]]:

    """

    Query the wcxf package for every WC in the SMEFT Warsaw basis and

    organise them by their wcxf 'sector' label.

    Returns

    -------

    dict  {sector_name: [wc_name, ...]}  - ordered by sector, then WC name

    """

    try:

        basis_obj = wcxf.Basis["SMEFT", "Warsaw"]

        catalogue: dict[str, list[str]] = {}

        for sector_name, sector_data in basis_obj.sectors.items():

            if sector_name == "dB=de=dmu=dtau=0":

                wcs = sorted(sector_data.keys())

                if wcs:

                    catalogue[sector_name] = wcs

        return catalogue

    except Exception as exc:

        warnings.warn(

            f"Could not load Warsaw basis from wcxf ({exc}). "

            "Falling back to a minimal built-in catalogue."

        )

        return _fallback_catalogue()

OPERATOR_CATALOGUE: dict[str, list[str]] = _build_catalogue_from_wcxf()

operator_vocab: list[str] = [wc for ops in OPERATOR_CATALOGUE.values() for wc in ops]

print(f"Loaded {len(operator_vocab)} operators")

print(operator_vocab[:100])

# Adding a copy method to SMEFTState class

# This assumes SMEFTState is a dataclass or similar simple class

def _smeftstate_copy(self):

    # Need to also copy active_ops and step for a complete state copy

    new_state = SMEFTState(pulls=np.copy(self.pulls), chi2=self.chi2, obs_names=self.obs_names)

    new_state.active_ops = list(self.active_ops) if hasattr(self, 'active_ops') else []

    new_state.step = self.step if hasattr(self, 'step') else 0

    return new_state

SMEFTState.copy = _smeftstate_copy

def _smeftstate_to_dict(self):

    return {

        "pulls": self.pulls.tolist(),

        "chi2": self.chi2,

        "obs_names": self.obs_names,

        "active_ops": getattr(self, 'active_ops', []),

        "step": getattr(self, 'step', 0),

    }

SMEFTState.to_dict = _smeftstate_to_dict


SEED = 42

print(f"Using seed = {SEED}")

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)


n_ops = len(operator_vocab)

n_obs = len(constraints_ewp)

M_ij = np.zeros(

    (n_ops,n_obs),

    dtype=np.float32

)

print("M_ij shape =", M_ij.shape)

print("operator_vocab =", len(operator_vocab))

print("constraints =", len(constraints_ewp))

# ============================================================

# SMEFT ACTOR-CRITIC

#

# IMPORTANT STATE DEFINITION

# ============================================================

#

# The Markov state is explicitly

#

#   s_t = (

#       pulls_t,

#       chi2_t,

#       active_operator_mask_t,

#       fired_operator_mask_t,

#       step_t

#   )

#

# BOTH actor and critic receive this SAME state.

#

# Actor:

#

#   pi_theta(a | s_t)

#

# Critic:

#

#   V_phi(s_t)

#

# The sampled return-to-go

#

#   G_t = r_t + gamma r_{t+1} + ...

#

# is used as the Monte-Carlo estimate of

#

#   Q^pi(s_t, a_t)

#

# and

#

#   A_t = G_t - V_phi(s_t)

#

# ============================================================


import os

import json

import random

import numpy as np

import scipy.stats

import torch

import torch.nn as nn

import torch.nn.functional as F


# ============================================================

# DEVICE

# ============================================================

DEVICE = torch.device(

    "cuda" if torch.cuda.is_available() else "cpu"

)


# ============================================================

# MODEL

# ============================================================

class SMEFTActorCritic(nn.Module):

    """

    Actor-Critic model with an explicitly shared state definition.

    STATE:

        s_t = (

            pulls,

            chi2,

            active operator mask,

            fired/evaluated operator mask,

            step

        )

    Both actor and critic are functions of exactly this state.

    Actor:

        pi(a | s)

    Critic:

        V(s)

    """

    def __init__(

        self,

        op_vocab,

        obs_names,

        sensitivity_matrix,

        chi2_reference,

        max_steps=10,

        embed_dim=32,

        n_heads=4,

        alpha_init=1.0,

        pull_gain=2.5,

    ):

        super().__init__()

        self.op_vocab = op_vocab

        self.obs_names = obs_names

        self.num_ops = len(op_vocab)

        self.num_obs = len(obs_names)

        self.embed_dim = embed_dim

        self.max_steps = max_steps

        # ----------------------------------------------------

        # Operator -> integer index

        # ----------------------------------------------------

        self.op_to_idx = {

            op: i

            for i, op in enumerate(op_vocab)

        }

        # ----------------------------------------------------

        # Sensitivity matrix

        # ----------------------------------------------------

        sm_shape = tuple(

            sensitivity_matrix.shape

        )

        expected = (

            self.num_obs,

            self.num_ops

        )

        expected_T = (

            self.num_ops,

            self.num_obs

        )

        if sm_shape == expected:

            M = sensitivity_matrix.T

        elif sm_shape == expected_T:

            M = sensitivity_matrix

        else:

            raise ValueError(

                f"sensitivity_matrix shape {sm_shape} does not match "

                f"(num_obs, num_ops)={expected} or "

                f"(num_ops, num_obs)={expected_T}"

            )

        self.register_buffer(

            "M",

            torch.as_tensor(

                M,

                dtype=torch.float32

            )

        )

        # ----------------------------------------------------

        # Reference chi2

        #

        # We normalize chi2 so the network does not receive

        # a potentially large raw number.

        # ----------------------------------------------------

        self.chi2_reference = float(

            max(abs(chi2_reference), 1.0) #FIXME

        )

        # ====================================================

        # STATE ENCODING

        # ====================================================

        # Operator identity

        self.op_embed = nn.Embedding(

            self.num_ops,

            embed_dim

        )

        # Observable pulls

        self.pull_proj = nn.Linear(

            1,

            embed_dim

        )

        # Active operator status:

        #

        #   0 = not active

        #   1 = active

        #

        self.active_embed = nn.Embedding(

            2,

            embed_dim

        )

        # Fired/evaluated status:

        #

        #   0 = not fired

        #   1 = fired

        #

        self.fired_embed = nn.Embedding(

            2,

            embed_dim

        )

        # Cross-attention between operators and observable pulls

        self.cross_attn = nn.MultiheadAttention(

            embed_dim=embed_dim,

            num_heads=n_heads,

            batch_first=True

        )

        # ----------------------------------------------------

        # Global state features

        #

        # chi2 + step

        # ----------------------------------------------------

        self.global_state_encoder = nn.Sequential(

            nn.Linear(

                2,

                embed_dim

            ),

            nn.Tanh(),

            nn.Linear(

                embed_dim,

                embed_dim

            ),

        )

        # ====================================================

        # ACTOR

        # ====================================================

        # Each operator receives:

        #

        #   operator identity

        #   pull/attention information

        #   active status

        #   fired status

        #   global state

        #

        # ALL of these come from the same s_t.

        self.actor_head = nn.Sequential(

            nn.Linear(

                embed_dim * 5,

                embed_dim

            ),

            nn.Tanh(),

            nn.Linear(

                embed_dim,

                1

            ),

        )

        # ====================================================

        # CRITIC

        # ====================================================

        # Critic receives a pooled representation of the SAME

        # operator-level state representation used by actor,

        # together with the global state.

        self.critic_head = nn.Sequential(

            nn.Linear(

                embed_dim * 6,

                embed_dim

            ),

            nn.ReLU(),

            nn.Linear(

                embed_dim,

                embed_dim // 2

            ),

            nn.ReLU(),

            nn.Linear(

                embed_dim // 2,

                1

            ),

        )

        # Trainable scaling parameters

        self.alpha = nn.Parameter(

            torch.tensor(

                alpha_init,

                dtype=torch.float32

            )

        )

        self.beta = nn.Parameter(

            torch.tensor(

                pull_gain,

                dtype=torch.float32

            )

        )

    # ========================================================

    # STATE ENCODER

    # ========================================================

    def encode_state(

        self,

        state,

        fired_ops=None,

    ):

        """

        Construct ONE explicit representation of s_t.

        This function is the central guarantee that actor and

        critic receive the SAME state.

        Returns

        -------

        operator_state : (num_ops, embed_dim * 5)

            Per-operator representation containing:

                operator identity

                pull/attention representation

                active status

                fired status

        global_state : (1, embed_dim)

            Global state containing:

                chi2

                step

        """

        device = self.M.device

        # ----------------------------------------------------

        # Pulls

        # ----------------------------------------------------

        pulls_data = (

            state.pulls

            if hasattr(state, "pulls")

            else state["pulls"]

        )

        if not isinstance(

            pulls_data,

            torch.Tensor

        ):

            pulls_tensor = torch.tensor(

                pulls_data,

                dtype=torch.float32,

                device=device

            )

        else:

            pulls_tensor = pulls_data.to(

                device=device,

                dtype=torch.float32

            )

        if pulls_tensor.ndim != 1:

            pulls_tensor = pulls_tensor.flatten()

        if pulls_tensor.numel() != self.num_obs:

            raise ValueError(

                f"Expected {self.num_obs} pulls, "

                f"got {pulls_tensor.numel()}"

            )

        # ----------------------------------------------------

        # Current chi2

        # ----------------------------------------------------

        chi2 = float(

            getattr(

                state,

                "chi2",

                state.get("chi2", 0.0)

                if isinstance(state, dict)

                else 0.0

            )

        )

        chi2_normalized = (

            chi2 / self.chi2_reference

        )

        # ----------------------------------------------------

        # Current step

        # ----------------------------------------------------

        step_value = float(

            getattr(

                state,

                "step",

                state.get("step", 0)

                if isinstance(state, dict)

                else 0

            )

        )

        step_fraction = (

            step_value

            / float(

                max(self.max_steps, 1)

            )

        )

        # ----------------------------------------------------

        # Operator identity

        # ----------------------------------------------------

        op_ids = torch.arange(

            self.num_ops,

            device=device

        )

        op_repr = self.op_embed(

            op_ids

        )

        # ----------------------------------------------------

        # Pull representation

        # ----------------------------------------------------

        pull_tokens = self.pull_proj(

            pulls_tensor.unsqueeze(-1)

        )

        # MultiheadAttention expects:

        #

        # query = operator tokens

        # key/value = observable pull tokens

        #

        # Shape:

        #

        #   (1, num_ops, embed_dim)

        #   (1, num_obs, embed_dim)

        # ----------------------------------------------------

        op_query = op_repr.unsqueeze(0)

        pull_tokens = pull_tokens.unsqueeze(0)

        attn_out, attn_weights = self.cross_attn(

            query=op_query,

            key=pull_tokens,

            value=pull_tokens

        )

        attn_out = attn_out.squeeze(0)

        # ----------------------------------------------------

        # ACTIVE OPERATOR MASK

        # ----------------------------------------------------

        active_mask = torch.zeros(

            self.num_ops,

            dtype=torch.long,

            device=device

        )

        active_ops = getattr(

            state,

            "active_ops",

            []

        )

        for op in active_ops:

            if op in self.op_to_idx:

                active_mask[

                    self.op_to_idx[op]

                ] = 1

        # ----------------------------------------------------

        # FIRED / EVALUATED MASK

        # ----------------------------------------------------

        fired_mask = torch.zeros(

            self.num_ops,

            dtype=torch.long,

            device=device

        )

        if fired_ops is not None:

            for idx in fired_ops:

                idx = int(idx)

                if 0 <= idx < self.num_ops:

                    fired_mask[idx] = 1

        # ----------------------------------------------------

        # Embed masks

        # ----------------------------------------------------

        active_repr = self.active_embed(

            active_mask

        )

        fired_repr = self.fired_embed(

            fired_mask

        )

        # ----------------------------------------------------

        # GLOBAL STATE

        #

        # EXACT SAME chi2 and step information is available

        # to both actor and critic.

        # ----------------------------------------------------

        global_scalar_state = torch.tensor(

            [

                chi2_normalized,

                step_fraction,

            ],

            dtype=torch.float32,

            device=device

        ).unsqueeze(0)

        global_repr = self.global_state_encoder(

            global_scalar_state

        )

        # ----------------------------------------------------

        # Broadcast global state to every operator

        # ----------------------------------------------------

        global_per_operator = (

            global_repr

            .expand(

                self.num_ops,

                -1

            )

        )

        # ----------------------------------------------------

        # ONE SHARED STATE REPRESENTATION

        # ----------------------------------------------------

        operator_state = torch.cat(

            [

                op_repr,

                attn_out,

                active_repr,

                fired_repr,

                global_per_operator,

            ],

            dim=-1

        )

        return (

            operator_state,

            global_repr,

            attn_weights,

            pulls_tensor,

            active_mask,

            fired_mask,

        )

    # ========================================================

    # FORWARD

    # ========================================================

    def forward(

        self,

        state,

        fired_ops=None,

    ):

        """

        Returns:

            logits

            V(s)

            attention

            diagnostics

        Both actor and critic are computed from the SAME

        state encoding produced by encode_state().

        """

        (

            operator_state,

            global_repr,

            attn_weights,

            pulls_tensor,

            active_mask,

            fired_mask,

        ) = self.encode_state(

            state,

            fired_ops=fired_ops

        )

        # ====================================================

        # ACTOR

        # ====================================================

        neural_logits = (

            self.actor_head(

                operator_state

            )

            .squeeze(-1)

            * self.alpha

        )

        # Optional direct physics term

        #

        # We keep this available but disabled, as in your

        # current implementation.

        pulls_column = pulls_tensor.unsqueeze(-1)

        direct_physics_logits = (

            torch.matmul(

                self.M,

                pulls_column

            )

            .squeeze(-1)

            * self.beta

        )

        # Current policy

        final_logits = neural_logits

        # ----------------------------------------------------

        # Mask operators that have already been fired

        # ----------------------------------------------------

        final_logits = final_logits.masked_fill(

            fired_mask.bool(),

            -1e9

        )

        # ====================================================

        # CRITIC

        # ====================================================

        # Pool the EXACT SAME operator_state that the actor

        # received.

        #

        # Therefore:

        #

        # actor = f(s)

        # critic = g(s)

        #

        # with the same s.

        # ----------------------------------------------------

        pooled_operator_state = (

            operator_state.mean(

                dim=0,

                keepdim=True

            )

        )

        critic_input = torch.cat(

            [

                pooled_operator_state,

                global_repr,

            ],

            dim=-1

        )

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

        state_value = self.critic_head(

            critic_input

        ).squeeze()

        diagnostics = {

            "base_logits":

                neural_logits.detach(),

            "direct_logits":

                direct_physics_logits.detach(),

            "pulls":

                pulls_tensor.detach()

                .cpu()

                .numpy(),

            "active_mask":

                active_mask.detach()

                .cpu()

                .numpy(),

            "fired_mask":

                fired_mask.detach()

                .cpu()

                .numpy(),

        }

        return (

            final_logits,

            state_value,

            attn_weights.squeeze(0),

            diagnostics,

        )


# ============================================================

# STEP-LEVEL REWARD

# ============================================================

def compute_step_rewards(

    history,

    alpha_step=0.1,

    dead_step_penalty=0.5,

    min_delta_chi2=1e-3,

    solved_bonus=10.0,

    chi2_threshold=1e-6,

):

    """

    One reward for every action.

    delta_chi2 =

        trial_chi2 - old_chi2

    Hence

        chi2_drop = -delta_chi2

    is positive when the action improves the fit.

    """

    rewards = []

    for step in history:

        delta = float(

            step["delta_chi2"]

        )

        chi2_drop = -delta

        reward = (

            chi2_drop

            - alpha_step

        )

        if chi2_drop < min_delta_chi2:

            reward -= dead_step_penalty

        if (

            step["trial_chi2"]

            < chi2_threshold

        ):

            reward += solved_bonus

        rewards.append(

            float(reward)

        )

    return rewards


# ============================================================

# RETURN-TO-GO

# ============================================================

def compute_returns_to_go(

    step_rewards,

    gamma=0.99,

):

    """

    G_t = r_t + gamma r_{t+1}

              + gamma^2 r_{t+2} + ...

    This is the Monte-Carlo estimate used as:

        G_t ~ Q^pi(s_t, a_t)

    """

    returns = [

        0.0

        for _ in step_rewards

    ]

    running = 0.0

    for t in reversed(

        range(len(step_rewards))

    ):

        running = (

            step_rewards[t]

            + gamma * running

        )

        returns[t] = running

    return returns


# ============================================================

# TRAJECTORY REWARD

# ============================================================

def compute_rollout_reward(

    final_state,

    baseline_chi2,

    trajectory_length=None,

    history=None,

    alpha_step=0.1,

    dead_step_penalty=0.5,

    min_delta_chi2=1e-3,

    solved_bonus=10.0,

    chi2_threshold=1e-6,

    clip_reward=False,

    clip_range=(-20.0, 20.0),

):

    """

    Logging / best-trajectory reward.

    NOT used for policy gradient.

    """

    if isinstance(

        final_state,

        dict

    ):

        if "chi2_total" in final_state:

            final_chi2 = final_state[

                "chi2_total"

            ]

        else:

            final_chi2 = final_state[

                "chi2"

            ]

    elif hasattr(

        final_state,

        "chi2"

    ):

        final_chi2 = final_state.chi2

    else:

        raise ValueError(

            "final_state does not contain chi2"

        )

    delta_chi2 = (

        baseline_chi2

        - final_chi2

    )

    reward = delta_chi2

    if (

        abs(delta_chi2)

        <= min_delta_chi2

    ):

        reward -= 30.0

    if final_chi2 < chi2_threshold:

        reward += solved_bonus

    if trajectory_length is not None:

        reward -= (

            alpha_step

            * trajectory_length

        )

    if history is not None:

        for item in history:

            delta = float(

                item.get(

                    "delta_chi2",

                    0.0

                )

            )

            chi2_drop = -delta

            if (

                chi2_drop

                < min_delta_chi2

            ):

                reward -= dead_step_penalty

    if clip_reward:

        reward = np.clip(

            reward,

            clip_range[0],

            clip_range[1]

        )

    return (

        float(reward),

        float(delta_chi2)

    )


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

    def __init__(

        self,

        momentum=0.01,

        eps=1e-6,

    ):

        self.momentum = momentum

        self.eps = eps

        self.mean = 0.0

        self.var = 1.0

        self.initialized = False

    def update(

        self,

        returns_t

    ):

        batch_mean = (

            returns_t

            .mean()

            .item()

        )

        batch_var = (

            returns_t

            .var(

                unbiased=False

            )

            .item()

        )

        if not self.initialized:

            self.mean = batch_mean

            self.var = max(

                batch_var,

                self.eps

            )

            self.initialized = True

        else:

            self.mean = (

                (1.0 - self.momentum)

                * self.mean

                + self.momentum

                * batch_mean

            )

            self.var = (

                (1.0 - self.momentum)

                * self.var

                + self.momentum

                * batch_var

            )

            self.var = max(

                self.var,

                self.eps

            )

    def normalize(

        self,

        returns_t

    ):

        std = (

            self.var ** 0.5

        ) + self.eps

        return (

            returns_t

            - self.mean

        ) / std

    def update_and_normalize(

        self,

        returns_t

    ):

        self.update(

            returns_t

        )

        return self.normalize(

            returns_t

        )


# ============================================================

# ROLLOUT

# ============================================================

def rollout(

    policy,

    pull_fn,

    obs_names,

    baseline_pulls,

    baseline_chi2,

    max_steps=10,

    deterministic=False,

    verbose=True,

    trajectory_idx=1,

    total_trajectories=30,

):

    device = policy.M.device

    # --------------------------------------------------------

    # INITIAL STATE

    # --------------------------------------------------------

    current_state = SMEFTState(

        pulls=np.array(

            baseline_pulls,

            dtype=np.float32

        ),

        chi2=float(

            baseline_chi2

        ),

        obs_names=obs_names,

    )

    current_state.active_ops = []

    current_state.step = 0

    # --------------------------------------------------------

    # Operators that have already been evaluated

    # --------------------------------------------------------

    all_evaluated_op_indices = set()

    accepted_op_indices = []

    # --------------------------------------------------------

    # Training quantities

    # --------------------------------------------------------

    log_probs = []

    entropies = []

    attentions = []

    diagnostics = []

    # V(s_t)

    state_values = []

    state_trajectory = [

        current_state.copy()

    ]

    history = []

    if verbose:

        print(

            f"\n--- Trajectory "

            f"[{trajectory_idx:02d} / "

            f"{total_trajectories:02d}] ---"

        )

    # ========================================================

    # TRAJECTORY

    # ========================================================

    for step_idx in range(

        max_steps

    ):

        # ----------------------------------------------------

        # IMPORTANT:

        #

        # The policy receives the COMPLETE state:

        #

        # pulls

        # chi2

        # active_ops

        # fired_ops

        # step

        #

        # ----------------------------------------------------

        (

            logits,

            state_val,

            attn_weights,

            diag,

        ) = policy(

            current_state,

            fired_ops=all_evaluated_op_indices

        )

        # ----------------------------------------------------

        # Policy distribution

        # ----------------------------------------------------

        dist = torch.distributions.Categorical(

            logits=logits

        )

        if deterministic:

            action = torch.argmax(

                logits

            )

        else:

            action = dist.sample()

        action_idx = int(

            action.item()

        )

        # ----------------------------------------------------

        # Store action information

        # ----------------------------------------------------

        log_probs.append(

            dist.log_prob(action)

        )

        entropies.append(

            dist.entropy()

        )

        attentions.append(

            attn_weights.detach()

        )

        diagnostics.append(

            diag

        )

        # IMPORTANT:

        #

        # This is V(s_t), i.e. the value BEFORE taking

        # action a_t.

        #

        state_values.append(

            state_val

        )

        # ----------------------------------------------------

        # Mark action as evaluated

        # ----------------------------------------------------

        all_evaluated_op_indices.add(

            action_idx

        )

        proposed_op = (

            policy.op_vocab[

                action_idx

            ]

        )

        # ----------------------------------------------------

        # Environment transition

        #

        # The active operators define the current SMEFT

        # theory.

        # ----------------------------------------------------

        trial_ops = (

            list(

                current_state.active_ops

            )

            + [proposed_op]

        )

        (

            _,

            trial_pulls,

            trial_chi2,

            _,

            _

        ) = pull_fn(

            trial_ops

        )

        trial_chi2 = float(

            trial_chi2

        )

        old_chi2 = float(

            current_state.chi2

        )

        chi2_reduction = (

            old_chi2

            - trial_chi2

        )

        delta_raw = (

            trial_chi2

            - old_chi2

        )

        # ----------------------------------------------------

        # Accept/reject

        # ----------------------------------------------------

        accepted = False

        if (

            chi2_reduction

            > deltachi2_min

        ):

            accepted = True

            accepted_op_indices.append(

                action_idx

            )

            current_state = SMEFTState(

                pulls=np.array(

                    trial_pulls,

                    dtype=np.float32

                ),

                chi2=trial_chi2,

                obs_names=obs_names,

            )

            current_state.active_ops = (

                list(trial_ops)

            )

            current_state.step = (

                step_idx + 1

            )

        else:

            # Rejected operator does not change

            # physics state, but DOES change:

            #

            #   fired mask

            #   step

            #

            current_state.step = (

                step_idx + 1

            )

        state_trajectory.append(

            current_state.copy()

        )

        # ----------------------------------------------------

        # Print

        # ----------------------------------------------------

        if verbose:

            status = (

                "ACCEPT"

                if accepted

                else "REJECT"

            )

            print(

                f"STEP {step_idx:02d}: "

                f"ADD {proposed_op:20s} "

                f"Deltachi^2={delta_raw:+8.3f} "

                f"(Drop: "

                f"{chi2_reduction:8.3f}) "

                f"{status}"

            )

        # ----------------------------------------------------

        # History

        # ----------------------------------------------------

        history.append({

            "step":

                step_idx,

            "proposed_operator":

                proposed_op,

            "accepted":

                accepted,

            "state_before":

                state_trajectory[-2].to_dict(),

            "state_after":

                current_state.to_dict(),

            "trial_chi2":

                float(trial_chi2),

            "delta_chi2":

                float(delta_raw),

        })

    # ========================================================

    # FINAL

    # ========================================================

    if verbose:

        if current_state.active_ops:

            final_ops_str = (

                " -> ".join(

                    current_state.active_ops

                )

            )

        else:

            final_ops_str = (

                "None (Pure SM)"

            )

        total_delta_chi2 = (

            float(baseline_chi2)

            - current_state.chi2

        )

        print(

            f"FINAL SEQUENCE "

            f"[{len(current_state.active_ops)} ops]: "

            f"{final_ops_str}"

        )

        print(

            f"FINAL RESULTS: "

            f"Total Deltachi^2 = "

            f"{total_delta_chi2:+.3f} | "

            f"Final chi^2 = "

            f"{current_state.chi2:.3f}"

        )

    return {

        "final_state":

            current_state,

        "state_trajectory":

            state_trajectory,

        "ops":

            accepted_op_indices,

        "op_names":

            current_state.active_ops,

        "log_probs":

            torch.stack(

                log_probs

            ),

        "entropies":

            torch.stack(

                entropies

            ),

        "state_values":

            torch.stack(

                state_values

            ),

        "attention":

            attentions,

        "diagnostics":

            diagnostics,

        "history":

            history,

    }


# ============================================================

# TRAINING PARAMETERS

# ============================================================

MAX_STEPS = 10

ROLLOUTS_PER_BATCH = 25

N_BATCHES = 300

deltachi2_min = 1e-2

GRAD_CLIP = 0.5

GAMMA = 0.99

CRITIC_LOSS_COEF = 0.5

ADV_NORMALIZE = True

RETURN_NORM_MOMENTUM = 0.01


# ============================================================

# MODEL INITIALIZATION

# ============================================================

M_tensor = torch.tensor(

    M_ij,

    dtype=torch.float32

)

M_mean = M_tensor.mean()

M_std = (

    M_tensor.std()

    + 1e-6

)

M_normalized = (

    M_tensor - M_mean

) / M_std


model = SMEFTActorCritic(

    op_vocab=operator_vocab,

    obs_names=OBS_NAMES,

    sensitivity_matrix=

        M_normalized,

    chi2_reference=

        CHI2_SM,

    max_steps=

        MAX_STEPS,

    embed_dim=32,

    n_heads=4,

    alpha_init=1.0,

    pull_gain=2.5,

).to(DEVICE)


optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=1e-3,

    weight_decay=1e-4,

)


return_normalizer = (

    RunningReturnNormalizer(

        momentum=

            RETURN_NORM_MOMENTUM

    )

)


# ============================================================

# OUTPUT

# ============================================================

OUTPUT_DIR = (

    "/home/kumarj/links/scratch/tmp_real/full_critic/"

)

os.makedirs(

    OUTPUT_DIR,

    exist_ok=True

)

METRICS_LOG_PATH = os.path.join(

    OUTPUT_DIR,

    "metrics_log.txt"

)

THEORY_ARCHIVE_PATH = os.path.join(

    OUTPUT_DIR,

    "theory_archive.json"

)

REWARD_DISTS_PATH = os.path.join(

    OUTPUT_DIR,

    "reward_distributions.json"

)


# ============================================================

# METRICS

# ============================================================

with open(

    METRICS_LOG_PATH,

    "w"

) as f:

    f.write(

        "batch\t"

        "mean_traj_reward\t"

        "max_traj_reward\t"

        "mean_delta_chi2\t"

        "actor_loss\t"

        "critic_loss\t"

        "entropy_loss\t"

        "total_loss\t"

        "alpha\t"

        "beta\t"

        "mean_advantage\t"

        "raw_advantage_mean\t"

        "raw_advantage_std\t"

        "mean_abs_raw_advantage\t"

        "max_abs_raw_advantage\t"

        "explained_variance\t"

        "return_mean\t"

        "return_std\t"

        "grad_norm\n"

    )


if not os.path.exists(

    THEORY_ARCHIVE_PATH

):

    with open(

        THEORY_ARCHIVE_PATH,

        "w"

    ) as f:

        json.dump(

            [],

            f,

            indent=2

        )


if not os.path.exists(

    REWARD_DISTS_PATH

):

    with open(

        REWARD_DISTS_PATH,

        "w"

    ) as f:

        json.dump(

            {},

            f,

            indent=2

        )


theory_archive = []

reward_distributions = {}

best_reward = -float("inf")

best_ops = None


# ============================================================

# TRAINING LOOP

# ============================================================

for batch_idx in range(

    N_BATCHES

):

    model.train()

    progress = (

        batch_idx

        / float(N_BATCHES)

    )

    entropy_coef = (

        0.01 * (1.0 - progress)

        + 0.0001 * progress

    )

    batch_traj_rewards = []

    batch_delta_chi2 = []

    batch_ops = []

    # --------------------------------------------------------

    # Step-level quantities

    # --------------------------------------------------------

    all_log_probs = []

    all_entropies = []

    all_state_values = []

    all_returns = []

    # --------------------------------------------------------

    # Diagnostics

    # --------------------------------------------------------

    batch_attn_correlations = []

    batch_base_norms = []

    batch_direct_norms = []

    print(

        f"\n==================== "

        f"BATCH {batch_idx:03d} "

        f"===================="

    )

    # ========================================================

    # ROLLOUTS

    # ========================================================

    for rollout_idx in range(

        ROLLOUTS_PER_BATCH

    ):

        result = rollout(

            policy=model,

            pull_fn=flavio_pull_fn,

            obs_names=OBS_NAMES,

            baseline_pulls=

                BASELINE_PULLS,

            baseline_chi2=

                CHI2_SM,

            max_steps=

                MAX_STEPS,

            deterministic=False,

            verbose=True,

            trajectory_idx=

                rollout_idx + 1,

            total_trajectories=

                ROLLOUTS_PER_BATCH,

        )

        final_state = result[

            "final_state"

        ]

        hist = result[

            "history"

        ]

        traj_ops = result[

            "ops"

        ]

        traj_len = len(

            traj_ops

        )

        # ====================================================

        # TRAJECTORY REWARD

        # ====================================================

        (

            traj_reward,

            delta_chi2

        ) = compute_rollout_reward(

            final_state=

                final_state,

            baseline_chi2=

                CHI2_SM,

            trajectory_length=

                traj_len,

            history=

                hist,

            alpha_step=0.1,

            dead_step_penalty=0.5,

        )

        # ====================================================

        # STEP REWARDS

        # ====================================================

        step_rewards = (

            compute_step_rewards(

                hist,

                alpha_step=0.1,

                dead_step_penalty=0.5,

            )

        )

        # ====================================================

        # RETURN-TO-GO

        # ====================================================

        returns_to_go = (

            compute_returns_to_go(

                step_rewards,

                gamma=GAMMA

            )

        )

        returns_t = torch.tensor(

            returns_to_go,

            dtype=torch.float32,

            device=DEVICE

        )

        # ====================================================

        # STORE ACTOR / CRITIC DATA

        # ====================================================

        all_log_probs.append(

            result["log_probs"]

        )

        all_entropies.append(

            result["entropies"]

        )

        all_state_values.append(

            result["state_values"]

        )

        all_returns.append(

            returns_t

        )

        # ====================================================

        # ARCHIVE

        # ====================================================

        theory_archive.append({

            "batch":

                batch_idx,

            "rollout":

                rollout_idx,

            "traj_reward":

                float(traj_reward),

            "step_rewards":

                [

                    float(r)

                    for r in step_rewards

                ],

            "returns_to_go":

                [

                    float(r)

                    for r in returns_to_go

                ],

            "delta_chi2":

                float(delta_chi2),

            "chi2_min":

                float(

                    final_state.chi2

                ),

            "operators":

                result["op_names"],

            "pulls":

                np.asarray(

                    final_state.pulls

                ).tolist(),

            "history":

                hist,

        })

        with open(

            THEORY_ARCHIVE_PATH,

            "w"

        ) as f:

            json.dump(

                theory_archive,

                f,

                indent=2

            )

        # ====================================================

        # BATCH LOGGING

        # ====================================================

        batch_traj_rewards.append(

            float(traj_reward)

        )

        batch_delta_chi2.append(

            float(delta_chi2)

        )

        batch_ops.append(

            traj_ops

        )

        # ====================================================

        # ATTENTION DIAGNOSTICS

        # ====================================================

        for (

            attn_matrix,

            diag

        ) in zip(

            result["attention"],

            result["diagnostics"]

        ):

            abs_pulls = np.abs(

                diag["pulls"]

            )

            if np.sum(

                abs_pulls

            ) > 0:

                step_corrs = []

                attn_np = (

                    attn_matrix

                    .cpu()

                    .numpy()

                )

                for op_idx in range(

                    attn_np.shape[0]

                ):

                    row_attn = (

                        attn_np[op_idx]

                    )

                    corr, _ = (

                        scipy.stats

                        .spearmanr(

                            row_attn,

                            abs_pulls

                        )

                    )

                    if not np.isnan(

                        corr

                    ):

                        step_corrs.append(

                            corr

                        )

                if step_corrs:

                    batch_attn_correlations.append(

                        np.mean(

                            step_corrs

                        )

                    )

            batch_base_norms.append(

                torch.norm(

                    diag[

                        "base_logits"

                    ]

                )

                .cpu()

                .item()

            )

            batch_direct_norms.append(

                torch.norm(

                    diag[

                        "direct_logits"

                    ]

                )

                .cpu()

                .item()

            )

        # ====================================================

        # BEST TRAJECTORY

        # ====================================================

        if (

            traj_reward

            > best_reward

        ):

            best_reward = (

                traj_reward

            )

            best_ops = list(

                traj_ops

            )

    # ========================================================

    # REWARD DISTRIBUTION

    # ========================================================

    reward_distributions[

        f"batch_{batch_idx}"

    ] = batch_traj_rewards

    # ========================================================

    # FLATTEN ALL STEPS

    # ========================================================

    flat_log_probs = torch.cat(

        all_log_probs

    )

    flat_entropies = torch.cat(

        all_entropies

    )

    flat_values = torch.cat(

        all_state_values

    )

    flat_returns = torch.cat(

        all_returns

    )

    # ========================================================

    # NORMALIZED RETURNS

    # ========================================================

    flat_returns_norm = (

        return_normalizer

        .update_and_normalize(

            flat_returns

        )

    )

    # ========================================================

    # RAW ADVANTAGE

    #

    # G_t is the Monte-Carlo Q estimate.

    #

    # V(s_t) is the critic.

    #

    # Therefore:

    #

    # A_t = G_t - V(s_t)

    #

    # ========================================================

    raw_advantages = (

        flat_returns_norm

        - flat_values.detach()

    )

    raw_advantage_mean = (

        raw_advantages.mean()

    )

    raw_advantage_std = (

        raw_advantages.std()

    )

    mean_abs_raw_advantage = (

        raw_advantages.abs().mean()

    )

    max_abs_raw_advantage = (

        raw_advantages.abs().max()

    )

    # ========================================================

    # EXPLAINED VARIANCE

    # ========================================================

    target_var = (

        flat_returns_norm

        .detach()

        .var()

    )

    if target_var.item() > 1e-8:

        explained_variance = (

            1.0

            - (

                flat_returns_norm.detach()

                - flat_values.detach()

            )

            .var()

            / (

                target_var

                + 1e-8

            )

        )

    else:

        explained_variance = torch.tensor(

            0.0,

            device=DEVICE

        )

    # ========================================================

    # ADVANTAGE NORMALIZATION

    # ========================================================

    advantages = raw_advantages

    if (

        ADV_NORMALIZE

        and advantages.numel() > 1

        and advantages.std().item() > 1e-6

    ):

        advantages = (

            (

                advantages

                - advantages.mean()

            )

            / (

                advantages.std()

                + 1e-8

            )

        )

    # ========================================================

    # ACTOR LOSS

    # ========================================================

    actor_loss = -(

        advantages.detach()

        * flat_log_probs

    ).mean()

    # ========================================================

    # CRITIC LOSS

    # ========================================================

    critic_loss = F.mse_loss(

        flat_values,

        flat_returns_norm.detach()

    )

    # ========================================================

    # ENTROPY

    # ========================================================

    entropy_loss = (

        -entropy_coef

        * flat_entropies.mean()

    )

    # ========================================================

    # TOTAL LOSS

    # ========================================================

    total_loss = (

        actor_loss

        + CRITIC_LOSS_COEF

        * critic_loss

        + entropy_loss

    )

    # ========================================================

    # BACKPROP

    # ========================================================

    optimizer.zero_grad(

        set_to_none=True

    )

    total_loss.backward()

    # ========================================================

    # GRADIENT CLIPPING

    # ========================================================

    raw_grad_norm = (

        torch.nn.utils

        .clip_grad_norm_(

            model.parameters(),

            GRAD_CLIP

        )

        .item()

    )

    optimizer.step()

    # ========================================================

    # BATCH METRICS

    # ========================================================

    batch_mean_traj_reward = float(

        np.mean(

            batch_traj_rewards

        )

    )

    batch_max_traj_reward = float(

        np.max(

            batch_traj_rewards

        )

    )

    batch_mean_delta_chi2 = float(

        np.mean(

            batch_delta_chi2

        )

    )

    return_std = (

        return_normalizer.var

        ** 0.5

    )

    # ========================================================

    # SAVE METRICS

    # ========================================================

    with open(

        METRICS_LOG_PATH,

        "a"

    ) as f:

        f.write(

            f"{batch_idx}\t"

            f"{batch_mean_traj_reward:.6f}\t"

            f"{batch_max_traj_reward:.6f}\t"

            f"{batch_mean_delta_chi2:.6f}\t"

            f"{actor_loss.item():.6f}\t"

            f"{critic_loss.item():.6f}\t"

            f"{entropy_loss.item():.6f}\t"

            f"{total_loss.item():.6f}\t"

            f"{model.alpha.item():.6f}\t"

            f"{model.beta.item():.6f}\t"

            f"{advantages.mean().item():.6f}\t"

            f"{raw_advantage_mean.item():.6f}\t"

            f"{raw_advantage_std.item():.6f}\t"

            f"{mean_abs_raw_advantage.item():.6f}\t"

            f"{max_abs_raw_advantage.item():.6f}\t"

            f"{explained_variance.item():.6f}\t"

            f"{return_normalizer.mean:.6f}\t"

            f"{return_std:.6f}\t"

            f"{raw_grad_norm:.6f}\n"

        )

    # ========================================================

    # PRINT SUMMARY

    # ========================================================

    print(

        f"\n>>> SUMMARY "

        f"BATCH {batch_idx:03d} <<<"

    )

    print(

        f"Actor: "

        f"{actor_loss.item():.4f} | "

        f"Critic: "

        f"{critic_loss.item():.4f} | "

        f"Entropy: "

        f"{entropy_loss.item():.4f} | "

        f"Mean Deltachi^2: "

        f"{batch_mean_delta_chi2:.2f}"

    )

    print(

        f"Return mean: "

        f"{return_normalizer.mean:.4f} | "

        f"Return std: "

        f"{return_std:.4f} | "

        f"Mean |Raw Adv|: "

        f"{mean_abs_raw_advantage.item():.4f} | "

        f"Raw Adv Std: "

        f"{raw_advantage_std.item():.4f} | "

        f"Explained Var: "

        f"{explained_variance.item():.4f}"

    )

    print(

        f"Gradient norm: "

        f"{raw_grad_norm:.4f}"

    )


