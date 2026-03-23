# Modeling Cross-Linguistic Production of Referring Expressions

## Metadata

- Authors: Brandon Waldon, Judith Degen
- Year: 2021
- Venue: Proceedings of the Society for Computation in Linguistics (SCiL) 2021, pages 206–215
- DOI / URL: https://scholarworks.umass.edu/scil/vol4/iss1/19
- Local PDF path: `10-writing/literatures/2021.scil-1.19v1.pdf`
- Tags: RSA, cross-linguistic, overspecification, incremental pragmatics, Spanish, English, color/size asymmetry

## One-Paragraph Summary

Waldon and Degen present a computational RSA model of referring expression production that synthesizes standard (global) RSA, incremental RSA (Cohn-Gordon et al. 2018), and continuous semantics (Degen et al. 2020) to account for two attested phenomena: the color/size asymmetry in redundant modification in English, and cross-linguistic differences in redundant modification rates between English (prenominal) and Spanish (postnominal). The model is assessed qualitatively under various parameter regimes for three hypothetical Spanish idiolects. The paper argues that neither standard RSA nor continuous RSA alone can account for the cross-linguistic data, and that incremental processing in combination with non-deterministic adjective semantics is needed.

## Research Question

- Can a unified RSA model account for (a) the asymmetry between redundant color and redundant size modification in English and (b) lower rates of redundant color modification in Spanish compared to English?
- What model components (global vs. incremental pragmatics; binary vs. continuous semantics) are individually necessary?

## Method / Data

- Design: Computational/theoretical — model comparison across parameter regimes
- Participants/Data: No new empirical data; model predictions compared to existing literature (Rubio-Fernández 2016; Degen et al. 2020)
- Measures: Predicted speaker probability of redundant modification in size-sufficient (SS) and color-sufficient (CS) scenes
- Analysis approach: Qualitative model assessment; agent-based RSA simulation for English and three hypothetical Spanish idiolects (postnom., split, postnom.-conj.)

## Key Findings

1. Standard RSA (global, binary semantics) cannot capture the color/size asymmetry in English or the cross-linguistic differences.
2. Continuous RSA (global, graded semantics) captures the color/size asymmetry but predicts equally high redundant modification in both English and Spanish-postnom., failing cross-linguistic predictions.
3. Incremental RSA combined with continuous semantics captures both the color/size asymmetry and lower redundant modification in postnominal Spanish, because the marginal benefit of early adjective mention is reduced when adjectives come after the noun.

## Core Takeaways For The Current Project

- Directly relevant: the paper motivates the same two-factor model architecture (incremental vs. global speaker) that our project implements in NumPyro.
- The finding that incremental RSA + continuous semantics is necessary to handle cross-linguistic data supports using an incremental speaker in the Schlotterbeck-Wang tradition.
- The paper uses the same SS/CS scene logic as Degen et al. (2020), providing conceptual scaffolding for our referential context design.

## Limitations / Boundary Conditions

- Model is assessed qualitatively, not fit to data with Bayesian inference; parameter regimes are explored but not estimated from empirical data.
- The Spanish data used for comparison involve single-modifier DPs; the cross-linguistic predictions for complex DPs (color + size together) are largely exploratory.
- Only predicts aggregate redundant modification rates, not adjective ordering within multi-adjective DPs.

## Useful Quotes (Short)

- "Our model makes incremental utterance choice predictions and assumes a non-deterministic semantics for adjectives in referring expressions." (Abstract)
- "Incremental RSA combined with continuous semantics captures previously attested production patterns, including English speakers' tendency to produce redundant color adjectives more frequently than redundant size adjectives, as well as Spanish speakers' tendency to employ redundant color adjectives less frequently than English speakers." (Abstract)

## Relevance To Current Writing Tasks

- Supports: The motivation for comparing global and incremental RSA speaker models in the adjective ordering task.
- Contrasts with: Standard RSA (global, binary) accounts that cannot handle the cross-linguistic picture.
- Open questions for us: The paper treats order as a fixed property of the language (prenominal vs. postnominal) rather than modeling within-language order preferences — our project extends this by modeling the ordering choice itself.
