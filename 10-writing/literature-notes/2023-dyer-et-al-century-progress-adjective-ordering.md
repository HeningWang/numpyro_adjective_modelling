# Evaluating a Century of Progress on the Cognitive Science of Adjective Ordering

## Metadata

- Authors: William Dyer, Charles Torres, Gregory Scontras, Richard Futrell
- Year: 2023
- Venue: Transactions of the Association for Computational Linguistics (TACL), doi:10.1162/tacl_a_00596
- DOI / URL: https://doi.org/10.1162/tacl_a_00596
- Local PDF path: `10-writing/literatures/Dyer et al. - 2023 - Evaluating a Century of Progress on the Cognitive .pdf`
- Tags: adjective ordering, subjectivity, information locality, integration cost, information gain, corpus study, dependency treebanks, efficiency

## One-Paragraph Summary

Dyer et al. conduct a large-scale corpus evaluation of four efficiency-based quantitative theories of adjective ordering in English: subjectivity (Scontras et al. 2017), information locality (Futrell 2019), integration cost (Dyer 2017), and information gain (a new proposal). Using hand-parsed and automatically-parsed dependency treebanks, they evaluate each theory's ability to predict the observed order of adjective pairs involving unseen combinations. They find that subjectivity, information locality, and information gain are all strong predictors, with some evidence for a two-factor model in which subjectivity and information gain capture complementary variance, while integration cost alone performs poorly.

## Research Question

- Which efficiency-based theories best predict the preferred order of adjective pairs in English corpora?
- Is there evidence for a two-factor account combining subjectivity and another information-theoretic predictor?

## Method / Data

- Design: Corpus study using dependency treebanks
- Participants/Data: Hand-parsed Penn Treebank + automatically parsed English web corpora; pairs of co-occurring prenominal adjectives extracted
- Measures: Prediction accuracy for unseen adjective pairs; correlation with observed order frequency
- Analysis approach: Regression models with single and combined predictors; evaluation on held-out data

## Key Findings

1. Subjectivity (Scontras et al. 2017) remains a strong predictor of adjective ordering across corpora and evaluation conditions.
2. Information gain — the mutual information between an adjective and the noun given a context — is a newly introduced predictor that performs comparably to subjectivity.
3. A two-factor model combining subjectivity and information gain outperforms either factor alone, suggesting that adjective-noun semantic relations and adjective subjectivity capture distinct aspects of ordering preferences.
4. Integration cost (Dyer 2017), which relies on the entropy of the set of possible head nouns, performs poorly in isolation.

## Core Takeaways For The Current Project

- Provides empirical support for subjectivity as a durable predictor in corpus data, which is the primary semantic predictor used in our RSA model for ordering.
- Information gain as defined here (mutual information between adjective and noun) is closely related to the communicative efficiency motivation in our RSA model — adjectives with higher mutual information with the noun are more useful for reference resolution and should be placed closer to the noun.
- The two-factor finding (subjectivity + information gain) suggests that the subjectivity account alone may be incomplete and that noun-specific semantic relations matter independently.

## Limitations / Boundary Conditions

- Corpus-based measures of order reflect both speaker preferences and genre/register effects; naturalistic data may conflate ordering preferences with collocational habits.
- The study is restricted to English; cross-linguistic generalization requires further work.
- The information gain measure depends on corpus-based estimates that may be noisy for infrequent adjective-noun pairs.

## Useful Quotes (Short)

- "We find that subjectivity, information locality, and information gain are all strong predictors, with some evidence for a two-factor account." (Abstract)

## Relevance To Current Writing Tasks

- Supports: The use of subjectivity as a predictor in our model; motivates including mutual information / discriminability as a secondary factor.
- Contrasts with: Models that rely on syntactic structure or integration cost alone.
- Open questions for us: Can information gain be operationalized within the RSA framework as a property of the state space, linking the corpus finding to our model's listener utility calculations?
