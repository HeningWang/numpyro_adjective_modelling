# Subjectivity-Based Adjective Ordering Maximizes Communicative Success

## Metadata

- Authors: Michael Franke, Gregory Scontras, Mihael Simonic
- Year: 2019 (appears as preprint / proceedings paper; associated CogSci 2019)
- Venue: Proceedings of the 41st Annual Conference of the Cognitive Science Society (CogSci 2019)
- DOI / URL: https://github.com/michael-franke/adjective_order (code repo cited in paper)
- Local PDF path: `10-writing/literatures/Franke et al. - Subjectivity-based adjective ordering maximizes co.pdf`
- Tags: subjectivity, communicative success, Monte Carlo simulation, RSA, sequential interpretation, adjective ordering

## One-Paragraph Summary

Franke, Scontras, and Simonic provide a formal rational explanation for why placing less subjective adjectives closer to the noun (the subjectivity-based ordering) maximizes average communicative success. They extend an RSA framework with a context-dependent, threshold-based semantics for scalar adjectives (following Simonic 2018) and assume sequentially intersective composition, where each adjective restricts the interpretation context for subsequent adjectives. Through Monte Carlo simulation over many randomly sampled referential contexts, they demonstrate that ordering adjectives with decreasing subjectivity produces higher average probability of listener identifying the intended referent compared to the reverse order. The key mechanism is that more objective adjectives, when placed closer to the noun, operate over a smaller, more restricted context and thus contribute more reliable information to reference resolution.

## Research Question

- Why does subjectivity-based adjective ordering (less subjective closer to the noun) maximize communicative success?
- Can this preference be derived from first principles of rational communication without stipulating syntactic structure?

## Method / Data

- Design: Formal model + Monte Carlo simulation (10^6 samples)
- Participants/Data: No empirical data collected; predictions derived analytically and verified by simulation
- Measures: Expected probability of referential success under subjectivity-ordered vs. reverse-ordered utterances
- Analysis approach: RSA model with threshold semantics and sequential intersectivity; Monte Carlo simulation averaging over contexts, noise levels, and semantic thresholds

## Key Findings

1. The expected probability of referential success for "big brown bag" (subjectivity order: more subjective first, less subjective closer to noun) is 0.54, versus 0.49 for "brown big bag" — a significant advantage for subjectivity-based ordering (paired t-test, t ≈ 19.26, p < 10^-80).
2. The mechanism is sequential intersective interpretation: adjectives closer to the noun restrict a smaller, already-constrained set of potential referents, making objective adjectives (which agree more across speakers) more reliable at that late, restricted stage.
3. The preference for subjectivity-based ordering emerges gradually through a stochastic process across contexts, consistent with graded rather than categorical empirical ordering preferences.

## Core Takeaways For The Current Project

- This paper provides the theoretical underpinning connecting subjectivity (faultless disagreement) to communicative efficiency — the core hypothesis in our model.
- The sequential intersective composition rule (each adjective interprets against the context restricted by prior adjectives) is essentially what our incremental speaker model computes, and its expected utility calculation corresponds to our listener-based RSA utility.
- The simulation result (ordering by subjectivity is on average better) is the key motivation for why the RSA model should prefer subjectivity-ordered utterances.

## Limitations / Boundary Conditions

- The simulation assumes a specific threshold-based semantics for scalar adjectives; results may not directly transfer to non-scalar adjectives (e.g., color).
- The model predicts a preference rather than a categorical constraint, meaning it allows many counter-examples — which matches the graded nature of empirical data.
- The paper uses a simplified two-adjective setting; generalizing to three or more adjectives requires extending the sequential intersective composition.

## Useful Quotes (Short)

- "Ordering adjectives with respect to decreasing subjectivity increases the probability of communicative success." (Results section)
- "The preference for subjectivity-based orderings would evolve gradually." (Discussion)

## Relevance To Current Writing Tasks

- Supports: The rational / efficiency-based motivation for subjectivity-ordered adjective sequences in our model.
- Contrasts with: Accounts that stipulate ordering preferences via fixed syntactic hierarchy (e.g., Cinque 1994).
- Open questions for us: Our RSA model infers both alpha (rationality) and sigma (noise) — does the noise parameter in our model correspond to the "more subjective adjectives deviate more from the ground truth" assumption in Franke et al.'s formalization?
