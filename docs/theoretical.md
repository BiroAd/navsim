# RL on the codebook policy: theory

Background for the RL fine-tuning of `vision_llm_lora` (RL-1 to RL-4). Implementation:
`navsim/agents/vision_llm_lora/vision_llm_lora_rl.py`, `scripts/training/rl_policy_head.py`; results:
[checkpoint_evaluation.md](checkpoint_evaluation.md) ("Shortlist headroom for RL", "RL on the policy head"). Written
2026-11-08.

## 1. Setting: a contextual bandit with known rewards

For every scene s the model outputs a distribution π(·|s) over the K = 16,384 codebook trajectories and drives one of
them (argmax, or the most likely entry of the heads' shortlist). There is one decision per scene and no sequence of
actions, so in RL terms the problem is a **contextual bandit**: context = scene, action = codebook entry, reward =
score of that entry.

Unusually, **the reward of every action is known in advance.** `exp/codebook_scores_navtrain` holds the PDM score of
all K entries in every navtrain scene, so a reward is a table lookup:

    r_k(s) = NC · DAC · DDC · TLC · (5·EP + 5·TTC + 2·LK + 2·HC) / 14 ∈ [0, 1]

i.e. the v2 EPDMS of entry k in scene s without two-frame comfort (EC needs the plan of the previous frame), with
log-replay (non-reactive) traffic and without the human-penalty filter. This is the same table as the offline proxy.
Below, s is dropped: everything holds per scene, and the training objective averages over scenes.

Two consequences:
- The classic offline-RL problem (the data shows only one action per state, so values of other actions must be
  extrapolated) does not arise.
- "Online" RL with log-replay traffic needs no simulator: simulating a sampled trajectory would recompute the table
  entry exactly. Only reactive traffic (navhard) or EC would add information the table lacks.

## 2. Objective: expected reward with a KL anchor

    J(π) = Σ_k π(k) · r_k  −  β · KL(π ‖ π_start)
         = E_π[r]  −  β · Σ_k π(k) · log( π(k) / π_start(k) )

π_start is the policy the RL run starts from (frozen copy). The first term pushes probability towards high-reward
trajectories; the KL term keeps π close to π_start. β ≥ 0 sets the trade-off.

**What β restricts.** The KL anchor limits how far the RL policy may move from the starting policy, and so protects
what the reward does not measure: the human-likeness and smoothness of the imitation-trained policy (two-frame comfort
EC, which the reward lacks), and robustness to errors of the reward table (log replay, no human filter). β = 0 is pure
reward maximization.

## 3. The optimal policy

Maximizing J over all distributions π (Lagrange multiplier for Σ_k π(k) = 1; set ∂/∂π(k) = 0):

    r_k − β · ( log π(k) − log π_start(k) + 1 ) − λ = 0

    ⇒  π*(k) = π_start(k) · exp(r_k / β) / Z,      Z = Σ_j π_start(j) · exp(r_j / β)

So **π\* ∝ π_start · exp(r / β)**: the starting policy, reweighted by the exponentiated reward. β acts as a temperature
on the reward.

**Reward difference.** For two trajectories k and j of the same scene, Δ = r_k − r_j:

    π*(k) / π*(j)  =  π_start(k) / π_start(j)  ·  exp(Δ / β)

The optimal policy keeps the starting policy's preference between k and j and multiplies it by exp(Δ/β). Examples for
Δ = 0.1 (a 10-point better score):

| β | factor exp(0.1 / β) |
|---|---|
| 0.5 | 1.2 |
| 0.2 | 1.6 |
| 0.1 | 2.7 |
| 0.05 | 7.4 |
| 0.02 | 148 |
| 0.005 | 4.9 · 10⁸ |

## 4. KL of the optimum and its bounds

Inserting π\* into the KL:

    KL(π* ‖ π_start) = Σ_k π*(k) · ( r_k / β − log Z ) = E_π*[r] / β − log Z

Properties (λ = 1/β):
- **Monotone:** d KL / dλ = λ · Var_π\*(r) ≥ 0, so the KL of the optimum grows as β shrinks.
- **Upper bound from the objective:** π\* is at least as good as π_start, E_π\*[r] − β·KL ≥ E_π_start[r], hence
  KL(π\* ‖ π_start) ≤ ( E_π\*[r] − E_π_start[r] ) / β ≤ ( max_k r_k − E_π_start[r] ) / β
  (the same follows from Jensen: log Z ≥ E_π_start[r] / β).
- **Limit β → 0:** π\* becomes π_start restricted to the set B of the best-scoring entries (renormalized), and

      KL → −log π_start(B),      π_start(B) = Σ_{k ∈ B} π_start(k)

  This is the largest KL any β can require: how surprising the best trajectories are to the starting policy. If
  π_start gives them 10% of its mass the limit is −log 0.1 = 2.3 nats; if 0.01%, 9.2 nats.

**On our data** (p3_aux_policy ep 3 as π_start; π\* computed per scene from that scene's rewards, 3,000 scenes per
split from cache shard 0; π_start from the cached fused states):

| β | train: KL(π\*‖π_start) | train: E_π\*[r] | train: argmax of π\* | val: KL | val: E_π\*[r] | val: argmax of π\* |
|---|---|---|---|---|---|---|
| start (π_start) | 0 | 0.841 | 0.897 | 0 | 0.723 | 0.792 |
| 0.5 | 0.05 | 0.904 | 0.932 | 0.10 | 0.834 | 0.895 |
| 0.2 | 0.13 | 0.930 | 0.940 | 0.28 | 0.892 | 0.913 |
| 0.1 | 0.20 | 0.940 | 0.947 | 0.41 | 0.911 | 0.923 |
| 0.05 | 0.33 | 0.949 | 0.955 | 0.61 | 0.925 | 0.933 |
| 0.02 | 0.73 | 0.961 | 0.965 | 1.14 | 0.941 | 0.946 |
| 0.01 | 1.36 | 0.970 | 0.973 | 2.05 | 0.954 | 0.957 |
| 0.005 | 2.37 | 0.977 | 0.979 | 3.70 | 0.965 | 0.968 |
| β → 0 | 4.94 (median 2.00) | 0.983 | 0.983 | 10.88 (median 5.68) | 0.981 | 0.981 |

(5.3% of the codebook is tied at the best score in an average train scene, 3.0% in val.)

How to read it: the val columns are an **oracle**: π\* uses the val scenes' own rewards, which a policy never sees, so
it bounds what is reachable, not what is expected. The train columns are what training could reach if the policy could
represent the optimum on every training scene. In val the starting policy puts very little mass on the best entries
(median −log π_start(B) = 5.7, i.e. ~0.3%): reaching them requires a large change of the ranking.

## 5. Gradients: exact objective and GRPO

With logits z (π = softmax(z)), the exact gradient of J is

    ∂J / ∂z_k = π(k) · ( r_k − E_π[r] )  −  β · π(k) · ( log(π(k)/π_start(k)) − KL(π‖π_start) )

Every entry is pushed up or down in proportion to its probability and to how much better or worse than average it is.
Because all K rewards are known, this is computed **exactly** (`expected_reward_kl_loss`, "exact" runs), with no
sampling noise; its cost is one softmax over K per scene.

**GRPO** (`grpo_loss`) is the sampled version: draw G entries k_1..k_G from π, advantage A_i = r_{k_i} − mean_j r_{k_j}
(standard GRPO additionally divides by the group std), loss −(1/G) Σ_i A_i · log π(k_i) + β·KL. Only the G sampled
entries are pushed directly; the others move through the softmax normalization and through the shared parameters
(similar trajectories share embeddings). In expectation, with the group-mean baseline,
E[ (1/G) Σ_i A_i ∇log π(k_i) ] = ((G − 1)/G) · ∇E_π[r]: the exact gradient (up to a constant factor) plus sampling
noise. With looked-up rewards GRPO can therefore at best match the exact objective; it matters where rewards must be
simulated per sample (reactive traffic, stage 2), where the exact sum over K is unaffordable.

**Why std normalization hurts here.** Dividing by the group std σ makes the policy-gradient term scale-free: a group
whose rewards differ by 0.001 gets the same gradient size as one differing by 0.5. Relative to the KL term this is a
per-scene effective β_eff ≈ β · σ: with typical σ ≈ 0.05–0.1 among sampled entries, β = 0.2 acts like 0.01–0.02, and
noise between nearly equal entries is amplified. Without the division ("Dr. GRPO", `--advantage mean`), the gradient
scales with the actual reward differences, like the exact one.

## 6. What a policy-head-only run can reach

The policy logits are z_k = c · tanh( (W f + b) · e_k / √D + b_k ) (soft cap), where f is the fused scene state
(1024-d), W the scene projection, e_k the frozen trajectory embedding and b_k a per-trajectory bias. With the backbone
frozen, f is fixed per scene, and RL on the policy head (RL-1 / RL-2) can only change W, b, b_k and the scale.

Each PDM sub-score head has the same form (its own projection of the same f against the same e_k). So inside the
heads' shortlist, the policy head has no information about which entry is better that the heads did not already
have: the policy can become much better on its own (argmax), but its gain inside the shortlist is bounded by the
features. Moving π towards π\* in val scenes, where the best entries get ~0.3% of π_start's mass, needs features that
rank them differently, i.e. training the LoRA (RL-3 / RL-4), with a frozen copy of the scorer adapter so the heads and
their shortlist stay valid.

Measured on train (β = 0.02): the optimum has KL 0.73 and argmax 0.965; the trained head reached KL 0.50 and argmax
0.917 at lr 1e-4 (5 epochs), and KL 0.85 / argmax 0.935 at lr 1e-3 (10 epochs). The head cannot represent π\* even
on the scenes it trains on.
