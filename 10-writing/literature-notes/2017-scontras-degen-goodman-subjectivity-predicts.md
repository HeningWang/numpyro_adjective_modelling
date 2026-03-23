# Subjectivity Predicts Adjective Ordering Preferences

## Metadata

- Authors: Gregory Scontras, Judith Degen, Noah D. Goodman
- Year: 2017
- Venue: Open Mind: Discoveries in Cognitive Science, 1(1), 53–65
- DOI / URL: https://doi.org/10.1162/opmi_a_00005
- Local PDF path: `10-writing/literatures/Scontras et al. - 2017 - Subjectivity Predicts Adjective Ordering Preferences.pdf`
- Tags: subjectivity, adjective ordering, faultless disagreement, corpus, behavioral experiment, cross-linguistic universals

## One-Paragraph Summary

Scontras, Degen, and Goodman test the hypothesis that adjective subjectivity — operationalized as the degree to which two speakers can faultlessly disagree about whether a property holds of an object — predicts adjective ordering preferences in English. Using behavioral experiments to measure both ordering preferences (how natural is "the big blue box" vs. "the blue big box") and subjectivity ratings (direct "how subjective is this adjective" ratings validated against faultless disagreement scores), they find that subjectivity explains nearly all variance in ordering preferences (r^2 ≈ .81). Subjectivity outperforms competing accounts based on inherentness, intersectivity, and concept formability. The finding that subjectivity — a general psychological construct — drives ordering preferences provides evidence that linguistic universals can emerge from general properties of cognition rather than language-specific grammatical machinery.

## Research Question

- Does adjective subjectivity predict ordering preferences, and does it outperform alternative predictors (inherentness, intersectivity, concept formability)?

## Method / Data

- Design: Three behavioral studies; corpus analysis
- Participants/Data: English-speaking participants recruited via Amazon Mechanical Turk for preference ratings and subjectivity norming
- Measures: Ordering preference score (proportion of A-B preferred over B-A for each adjective in two-adjective strings); subjectivity ratings (direct "how subjective" scale); faultless disagreement scores; corpus proportion measures
- Analysis approach: Linear regression; Pearson correlations; comparison of predictor R-squared values

## Key Findings

1. Adjective subjectivity is a robust predictor of ordering preferences (r^2 ≈ .81 in behavioral data; r^2 ≈ .83 when compared to corpus data), accounting for nearly all variance.
2. Faultless disagreement and direct subjectivity ratings are highly correlated (r^2 ≈ .91), validating both measures as proxies for subjectivity.
3. Subjectivity outperforms inherentness (which accounts for ~0% of variance), intersectivity, and concept formability.
4. The subjectivity gradient predicts the well-known hierarchy: color is less subjective than size; form/material is less subjective still, consistent with cross-linguistic generalizations.

## Core Takeaways For The Current Project

- This is the primary source for the subjectivity hypothesis used in our model. The faultless disagreement operationalization of subjectivity is what our RSA model encodes through the sequential intersective semantics: less subjective adjectives produce more reliable restrictions.
- The result that subjectivity (r^2 ≈ .81) outperforms all alternatives justifies using it as the primary semantic predictor.
- The corpus-behavior correlation (r^2 = .83) justifies using behavioral ordering preferences as a valid proxy for natural production.

## Limitations / Boundary Conditions

- The study is conducted entirely in English; cross-linguistic validation is important (see Trainin & Shetreet 2021 for Hebrew; Scontras 2023 for review).
- Subjectivity norms are collected in a context-free setting; ordering preferences in referential contexts may deviate (see Danks & Schwenk 1972; Fukumura 2018).
- The measure of inherentness used to compare against may not be the most sensitive operationalization available.

## Useful Quotes (Short)

- "Less subjective adjectives occur closer to the nouns they modify." (Core claim)
- "Subjectivity scores account for nearly all of the variance in the ordering preference data." (Results)
- "Subjectivity synthesizes — rather than supplants — many of the previous psychological approaches." (Discussion)

## Relevance To Current Writing Tasks

- Supports: The subjectivity hypothesis as the semantic basis for adjective ordering; faultless disagreement as the operationalization.
- Contrasts with: Cartographic/syntactic approaches; inherentness-based accounts.
- Open questions for us: Our experiment varies subjectivity across adjective classes (color, size, form) — does the model recover the correct subjectivity ordering through the inferred alpha and beta parameters?
