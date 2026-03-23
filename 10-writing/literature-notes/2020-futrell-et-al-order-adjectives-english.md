# What Determines the Order of Adjectives in English? Comparing Efficiency-Based Theories Using Dependency Treebanks

## Metadata

- Authors: Richard Futrell, William Dyer, Gregory Scontras
- Year: 2020
- Venue: Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020)
- DOI / URL: (conference proceedings; authors at UC Irvine and Oracle Corporation)
- Local PDF path: `10-writing/literatures/Futrell et al. - 2020 - What determines the order of adjectives in English Comparing efficiency-based theories using depend.pdf`
- Tags: adjective ordering, subjectivity, information locality, integration cost, information gain, corpus, dependency treebanks, efficiency

## One-Paragraph Summary

Futrell, Dyer, and Scontras implement and test four efficiency-based quantitative theories of adjective ordering — subjectivity, information locality, integration cost, and a new measure called information gain — by evaluating each theory's ability to predict observed adjective orders in English hand-parsed and automatically-parsed dependency treebanks. They find that subjectivity, information locality, and information gain are all strong predictors of adjective order, while integration cost performs poorly alone. A two-factor account combining subjectivity with information gain shows improved predictions, suggesting that properties of adjective meaning (subjectivity) and adjective-noun semantic dependency (information gain) capture complementary aspects of ordering.

## Research Question

- Among four theoretically motivated efficiency-based theories of adjective ordering, which best predict attested orders in English corpora?
- Is there a two-factor model that outperforms any single factor?

## Method / Data

- Design: Corpus study; regression analysis
- Participants/Data: Penn Treebank (hand-parsed); automatically parsed large English web corpus; adjective pairs from multi-adjective noun phrases
- Measures: Prediction accuracy on unseen adjective pairs; R-squared; odds ratios
- Analysis approach: Logistic regression predicting observed order direction; leave-one-out cross-validation

## Key Findings

1. Subjectivity (Scontras et al. 2017) is a robust predictor of adjective ordering, replicating prior work in a large-scale corpus setting.
2. Information gain (mutual information between adjective and noun, given the other adjective) is introduced here and performs comparably to subjectivity.
3. Integration cost (entropy of the distribution over heads that an adjective can attach to) performs poorly as a solo predictor.
4. A two-factor model of subjectivity + information gain provides better fits than either alone, suggesting both meaning subjectivity and adjective-noun collocational specificity contribute.

## Core Takeaways For The Current Project

- Provides corpus-level validation for subjectivity as a predictor, important for establishing the prior plausibility of our RSA model's subjectivity-based motivation.
- Information gain as defined here relates to how specifically an adjective tends to co-occur with its head noun — a measure of adjective-noun semantic specificity that overlaps with the "less subjective adjectives name more inherent properties" intuition.
- The two-factor result (subjectivity + information gain) suggests our model may benefit from including a noun-specificity or adjective-noun dependency measure alongside the subjectivity-based RSA mechanism.

## Limitations / Boundary Conditions

- Corpus-based: observed order frequencies reflect community-wide conventions and may conflate online processing preferences with long-term conventionalization.
- English only; cross-linguistic extension would require per-language corpora and subjectivity norms.
- The information gain measure relies on corpus co-occurrence statistics that may be sparse for rare adjective-noun pairs.

## Useful Quotes (Short)

- "We find that subjectivity, information locality, and information gain are all strong predictors, with some evidence for a two-factor account, where subjectivity and information gain reflect complementary aspects of ordering preferences." (Abstract)

## Relevance To Current Writing Tasks

- Supports: Subjectivity as the primary predictor and the efficiency-based motivation for our RSA model.
- Contrasts with: Integration cost accounts; syntactic cartography accounts.
- Open questions for us: Can we define an analog of information gain within the RSA framework — e.g., as a property of how much a given adjective narrows the referential search space in a given context?
