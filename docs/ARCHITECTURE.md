# Architecture

TasteTrail v13 solves a fixed movie comparison: five liked films, three disliked films, and two unseen candidates. The selected production configuration is `qloo_hierarchical_mixture`; its matched no-Qloo control is `agent_hierarchical_mixture`.

## Components and responsibilities

| Component | Implementation | Responsibility |
|---|---|---|
| Web API | `work/culture_v13/app_v13.py` | Validate ten distinct catalog films, route requests, hide internal server errors |
| Orchestrator | `service_v13.py` | Load assets, run the selected branch, assemble evidence and cost summaries |
| Initial estimator | `work/culture_v11/baseline_snapshot.py`, `PROMPT` | One GLM request using visible history and candidate metadata |
| Planner | `hierarchical_agent.py`, `PLAN_PROMPT` | Form 1–3 testable hypotheses, support/counterevidence references, unknowns, and query groups |
| Background tools | `work/culture_v11/src/tastetrail/cluster_tools.py` | Collaborative factors and community preference margins |
| Soft groups | `mixture_tools.py` | Infer weighted membership in 12/24 taste groups and predict candidate preference |
| Qloo evidence | `hierarchical_agent.py`, `evidence_cards.py` | Retrieve exact-candidate group associations and allowed film culture descriptions |
| Reviewer | `hierarchical_agent.py`, `REVIEW_PROMPT` | Compare candidates against the same hypotheses, evidence, and contrary history |
| Recovery wrapper | `hierarchical_repair.py` | One bounded recovery for an objectively malformed or truncated review |
| Decision layer | `feature_tools_v13.py`, `inference_v13.py` | Shared serving/evaluation features, fusion, and fallback |

Paths without a prefix in this table are under `work/culture_v13`. The planner and reviewer are roles of the same model service; deterministic tools are not independent LLM agents.

## Request flow

1. The server resolves selected title/year pairs to verified catalog identities. The visible history is assigned local IDs H01–H08; candidates are C01 and C02.
2. The initial GLM call returns a ranking, a confidence value, and a short explanation. All three compared methods share this initial response.
3. The planner proposes tentative hypotheses. Each must identify supporting history, separate counterevidence, and what cannot be established. It selects two non-overlapping liked groups from H01–H05 and one disliked group from H06–H08. It must not rank candidates.
4. Local tools calculate personalized margins. In the Qloo branch, three group queries each request only the same two candidate entities. Available culture descriptions are attached as bounded evidence cards.
5. The reviewer compares both candidates, tests the planner's claims, and returns a ranking with a concise evidence explanation. It does not receive the initial GLM choice or rationale.
6. A regularized logistic fusion combines the initial estimate, matrix-factor margin, community margin, two soft-group margins, and reviewer margin. The reviewer's choice is evidence, not an override of the numerical tools.

## What soft clustering means here

From 3,640 background users, Bernoulli mixtures learn 12 and 24 overlapping patterns of likes and dislikes. Ratings 4–5 are positive, 1–2 are negative; rating 3 and unobserved films are not treated as dislikes. The models balance positive/negative observations and shrink estimates for sparsity.

For a new user, the eight visible ratings produce a posterior weight for every group. A person may resemble several groups at once. Candidate preference is a weighted sum across those groups, followed by a correction for population-wide liking. This is a numerical estimate from sparse observations, not a psychological category.

The no-Qloo branch smooths group preferences through collaborative film relationships. The Qloo branch uses the cultural relationship kernel built from allowed film metadata. Both use the same mixture mechanism and group counts. Their feature scales and fusion weights are calibrated separately on development data.

## How Qloo changes the decision

**Numerical route:** `bernoulli_k12_qloo` and `bernoulli_k24_qloo` directly enter the final selected fusion. Cultural information can therefore alter a choice even if the reviewer gives a similar explanation.

**Review route:** the planner selects seed groups; Qloo returns affinities for the exact same candidate pair. For each group the code computes `log(affinity(C01)) - log(affinity(C02))`. A rating-strength-weighted liked-group margin minus the disliked-group margin gives signed cultural evidence. This helps distinguish a specific preference from a candidate that is broadly associated with everything.

For illustration only, if the liked-group affinity ratio is 2 and the disliked-group ratio is 3, the contrast is `log(2) - log(3) < 0`. Even though C01 is associated with the likes, it is more strongly associated with the dislikes relative to C02. The sign favors C02 on this particular evidence dimension. These invented numbers explain the calculation; they are not a recorded Qloo result or a personal probability.

Available public descriptions of themes, style, tone, audience, genre, and tags provide semantic context. The code does not infer a plot fact from an affinity number. Missing descriptions remain missing.

## Prompt contract

The complete runnable prompts are in source, not generated from test answers:

- [Initial estimator](../work/culture_v11/baseline_snapshot.py): prefer personal evidence over generic popularity; use the same eight ratings and two candidates; return a complete two-item ranking.
- [Planner and reviewer](../work/culture_v13/hierarchical_agent.py): hypotheses must be tentative and refer to actual history; reviewer must consider contrary observations and must not treat tool margins as personal probabilities.
- [Format recovery](../work/culture_v13/hierarchical_repair.py): return a short valid JSON comparison from the same original visible evidence.

All roles treat input strings as data. They output concise results and evidence, rather than a reasoning transcript. Schema validation checks IDs, history group membership, disjoint positive groups, and ranking consistency.

## Reliability and accounting

The selected backend is `hierarchical_v2_repair`. It retains the original planner and first review, with at most one extra low-effort review when the first response is objectively truncated or has invalid output structure. Both Agent branches use the same rule. It never retries a valid answer because it is incorrect, uncertain, or inconvenient. The original failed request remains in the cost.

Missing required Qloo evidence invokes the selected no-Qloo fusion with initial-model review margins and no new paid call. A failed review with available Qloo evidence retains the other Qloo features while replacing the review margin with the initial estimate. Missing evidence is never interpreted as dislike.

Each production branch includes the initial GLM call, planner, its own reviewer, and any recovery. A preflight budget uses an 18× conservative allowance relative to the initial call. Unknown usage remains unknown unless a documented upper bound exists. The three-way comparison shares computations where possible; its total expense is distinct from a single final recommendation.

## Data boundaries

Qloo requests use public film UUIDs and bounded query fields. No name, email, original MovieLens user ID, or hidden candidate rating is sent. The model sees the supplied film history and ratings, public candidate metadata, and allowed evidence. Private files remain outside the repository; only aggregated research results are published.

Deployment asset construction is separate from experiment replay. Background assets are rebuilt from an authorized MovieLens download; current Qloo responses and a mutable model alias can differ from the archived experiment. A new deployment must not claim exact historical reproducibility solely because it uses the frozen fusion configuration.

## Research influences and scope of adaptation

The system is an engineering combination of established ideas, with attribution. We do not claim to reproduce the training methods or reported gains of these papers:

- [UR4Rec, COLING 2025](https://aclanthology.org/2025.coling-main.45/) motivates candidate-conditioned preference evidence instead of a generic biography. Our fixed two-candidate evidence cards use deterministic selection, not its learned retriever.
- [ARAG, 2025 preprint](https://arxiv.org/abs/2506.21931) motivates separating user understanding, evidence alignment, and ranking. We use a compact planner/reviewer sequence and local tools, rather than claiming its four-agent implementation.
- [NextQuill](https://arxiv.org/abs/2506.02368) motivates distinguishing preference-specific information from shared signals. Our population corrections and matched controls are statistical engineering choices; they do not establish causal effects.
- [AgenticRec, 2026 preprint](https://arxiv.org/abs/2603.21613) motivates tool-grounded reasoning over positive and negative evidence. We do not train GLM with its reinforcement learning procedure.
- [Automated Composition of Agents, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/39c51d0fe64c273413a4d358361b3daa-Abstract-Conference.html) motivates evaluating component utility together with cost. Our selection is development-only and frozen before confirmation; it is not an online knapsack agent.

Only the implemented components and their relevant influences are documented here. The public `agent_config.json` contains the selected Agent and Agent + Qloo configurations, with the same numerical parameters as the final confirmation experiment.

## Read-only decision receipt

`work/culture_v13/decision_receipt.py` runs after inference. It verifies each numeric input, reconstructs the selected fusion score from normalized features and frozen coefficients, and publishes grouped additive contributions only when that reconstruction matches the returned score. Contributions are oriented toward the selected candidate. They are model-score units, not percentages or causal effects. A fallback reconstructs the configured fallback calculation, without claiming Qloo improved it.

The cultural ledger treats a stronger association with liked seeds as support and with disliked seeds as counterevidence. Seed IDs must belong to the correct visible rating group. Missing or invalid affinities remain unavailable; a valid zero stays zero. This ledger describes observations and is separate from the learned fusion account. If receipt construction fails, the already completed recommendation is preserved. All visible strings use text nodes in the browser. The optional download is a local text file containing the submitted film ratings; it does not publish those ratings.
