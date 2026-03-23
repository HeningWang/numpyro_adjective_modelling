# An Incremental RSA Model for Adjective Ordering Preferences in Referential Visual Context

## Metadata

- Authors: Fabian Schlotterbeck, Hening Wang
- Year: 2023
- Venue: Proceedings of the Society for Computation in Linguistics (SCiL) 2023, pages 121–132. Amherst, Massachusetts, June 15–17, 2023.
- DOI / URL: (SCiL 2023 proceedings)
- Local PDF path: `10-writing/literatures/Schlotterbeck and Wang - 2023 - An Incremental RSA Model for Adjective Ordering Pr.pdf`
- Tags: RSA, incremental processing, adjective ordering, subjectivity, discriminatory strength, referential context, preference ratings, German

## One-Paragraph Summary

Schlotterbeck and Wang report a preference rating experiment in German testing two efficiency-based hypotheses about adjective ordering: the SUBJECTIVITY hypothesis (Scontras et al. 2017, 2019) and the DISCRIMINATORY STRENGTH hypothesis (Fukumura 2018). Participants rated orderings of adjective pairs in visual referential contexts where either the more or less subjective adjective was the more discriminating one. Results show that adjectives with lower discriminatory strength (less useful for identifying the target) are preferred later in multi-adjective expressions, consistent with DISCRIMINATORY STRENGTH; the subjectivity preference weakens or inverts when context makes the more subjective adjective more discriminating. To account for these data, the authors propose an incremental RSA model in the Cohn-Gordon et al. (2019) tradition, implementing fully incremental interpretation without requiring prediction of utterance completions, which explains both the SUBJECTIVITY and DISCRIMINATORY STRENGTH effects.

## Research Question

- Do subjectivity and discriminatory strength both influence adjective ordering in referential visual context, and how do they interact?
- Can an incremental RSA model account for both effects simultaneously?

## Method / Data

- Design: Web-based preference rating experiment; 2 (COMBINATION: dimension × color/shape, or color × shape) × 2 (RELEVANCE: first or second adjective relevant for reference) × 2 (SIZE DISTRIBUTION: sharp vs. blurred, between-participants) mixed factorial design
- Participants/Data: N = 120 German native speakers recruited via prolific.co; 81 experimental + 99 filler items per participant; total 486 experimental items across 6 lists
- Measures: Slider ratings indicating preference between two orderings of adjective pairs in visual context
- Analysis approach: Mixed-effects logistic regression; separate analyses by COMBINATION; incremental RSA model simulation

## Key Findings

1. If the communicative efficiency of an adjective is low in a given context (i.e., it does not help distinguish the target), it is preferred later in the multi-adjective expression — supporting DISCRIMINATORY STRENGTH.
2. In sharp SIZE DISTRIBUTION conditions (where size is more informative), the preference for subjective-first ordering is strengthened in dimension-relevant contexts — a complex interaction between context and subjectivity.
3. The incremental RSA model accounts for the qualitative pattern: fully incremental interpretation (without anticipating completions) correctly predicts the joint influence of subjectivity and discriminatory strength, with less-subjective adjectives that are also more discriminating preferred closer to the noun.
4. The two hypotheses make divergent predictions in certain contexts; the data favor DISCRIMINATORY STRENGTH as the dominant short-term force when context strongly discriminates one adjective.

## Core Takeaways For The Current Project

- This paper is the direct predecessor of the current project: our NumPyro model implements the incremental RSA speaker from this paper, and our slider experiment extends its preference rating paradigm to English with a full 2×2 adjective-class design (color × size × form).
- The finding that discriminatory strength (context-driven referential utility) modulates the subjectivity preference is central to our model — the RSA listener's posterior probability of identifying the target is the operationalization of discriminatory strength.
- The incremental model architecture (fully incremental, no look-ahead) is the specification of our `speaker_recursive` model.

## Limitations / Boundary Conditions

- The experiment uses German, a prenominal language; our replication in English adds cross-linguistic scope.
- The experiment uses a preference rating slider, not a production task; production data may differ.
- The model is not formally fit via Bayesian inference in this paper; our project extends it with MCMC-based parameter estimation.

## Useful Quotes (Short)

- "If the communicative efficiency of an adjective is low in a given context, it is preferred later in a multi-adjective expression." (Abstract)
- "What sets the model apart from previous approaches is that it assumes fully incremental interpretation, without the need to anticipate possible sentence completions." (Abstract)

## Relevance To Current Writing Tasks

- Supports: The incremental RSA model architecture; the dual-hypothesis experimental design; the slider preference rating paradigm.
- Contrasts with: Global RSA models; models that treat ordering preferences as context-independent.
- Open questions for us: Does the incremental model fit our English production data as well as the German preference data, and what do the inferred alpha and beta parameters tell us about rationality and cost tradeoffs?
