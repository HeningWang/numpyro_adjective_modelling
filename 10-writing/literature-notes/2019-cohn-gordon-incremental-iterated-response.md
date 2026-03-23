# An Incremental Iterated Response Model of Pragmatics

## Metadata

- Authors: Reuben Cohn-Gordon, Noah D. Goodman, Christopher Potts
- Year: 2019
- Venue: Proceedings of the Society for Computation in Linguistics (SCiL) 2019, pages 81–90
- DOI / URL: https://scholarworks.umass.edu/scil/vol2/iss1/10
- Local PDF path: `10-writing/literatures/Cohn-Gordon et al. - 2019 - An Incremental Iterated Response Model of Pragmatics.pdf`
- Tags: RSA, incremental pragmatics, iterated response, word-by-word, referring expressions, TUNA corpus

## One-Paragraph Summary

Cohn-Gordon, Goodman, and Potts develop an incremental extension of the Rational Speech Act (RSA) framework in which pragmatic reasoning proceeds word-by-word rather than over complete utterances. The key innovation is an incremental semantic function defined in terms of possible full-utterance completions of a partial string, which allows a word-level RSA speaker (SWORD) and listener (LWORD) to be defined. They show that this incremental model makes qualitatively different predictions from global RSA — specifically, an incremental speaker can arrive at globally informative utterances via locally greedy decisions — and that it accounts for (i) cross-linguistic asymmetries in over-modification between pre- and postnominal languages, and (ii) anticipatory pragmatic inferences. They also show the model produces more realistic outputs than global RSA on the TUNA referring expression corpus.

## Research Question

- Do incremental pragmatic processes (word-by-word reasoning) lead to different predictions than global RSA, and if so, can the incremental model account for empirical phenomena the global model cannot?

## Method / Data

- Design: Computational; formal definition of incremental RSA agents + qualitative comparison + TUNA corpus evaluation
- Participants/Data: No new experimental data; TUNA referring expression corpus (Gatt et al. 2009)
- Measures: Predicted utterance distributions; corpus fit
- Analysis approach: Formal derivation of incremental semantics from global semantics; comparison of speaker and listener agents across models; TUNA evaluation

## Key Findings

1. The incremental speaker (SUTT-IP) and global speaker (SUTT-GP) produce different preferred utterances even for the same referent — not just quantitatively but qualitatively (they may prefer different optimal utterances).
2. An incremental speaker can explain cross-linguistic asymmetries in over-modification: postnominal languages reduce the incremental informational benefit of early adjective production, predicting lower over-modification rates.
3. Incremental pragmatic listeners can derive anticipatory inferences from the first word of an utterance before the full utterance is heard, explaining Sedivy-style contrastive inference.
4. The model outperforms global RSA on TUNA referring expression generation.

## Core Takeaways For The Current Project

- This paper is the foundational reference for the incremental RSA speaker architecture that our project implements. Our `speaker_recursive` (incremental) model uses the same word-by-word reasoning structure.
- The derivation of SUTT-IP from SWORD via the chain rule is directly the computational logic behind our incremental speaker: each word choice is made to maximize local informational utility given the prefix produced so far.
- The contrast with global RSA maps directly to the two models we compare: global_speaker vs. speaker_recursive.

## Limitations / Boundary Conditions

- The incremental semantics defined here requires knowing the full set of possible utterance completions, which may be computationally intractable for large utterance spaces.
- The paper uses simplified toy examples and a corpus (TUNA) that is structurally different from naturalistic adjective ordering in complex DPs.
- No Bayesian parameter fitting; predictions are qualitative.

## Useful Quotes (Short)

- "A speaker that incrementally makes pragmatically informative choices arrives at an utterance which is globally informative." (Introduction)
- "SUTT-GP and SUTT-IP are not only quantitatively different, but even differ in their predictions about which utterances are optimal." (Section 2.3)
- "In natural language, speakers and listeners produce and comprehend utterances segment by segment." (Section 2.2)

## Relevance To Current Writing Tasks

- Supports: The theoretical motivation for the incremental speaker model in our paper.
- Contrasts with: The global RSA speaker, which does not capture word-order-sensitive phenomena.
- Open questions for us: Our incremental model uses the entire NP as the unit of decision rather than truly word-by-word; how does this relate to the strict word-by-word formulation here?
