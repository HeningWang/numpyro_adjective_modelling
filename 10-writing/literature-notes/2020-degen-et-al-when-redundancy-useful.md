# When Redundancy Is Useful: A Bayesian Approach to "Overinformative" Referring Expressions

## Metadata

- Authors: Judith Degen, Robert X. D. Hawkins, Caroline Graf, Elisa Kreiss, Noah D. Goodman
- Year: 2020
- Venue: Psychological Review, 127(6), 1006–1033
- DOI / URL: https://dx.doi.org/10.1037/rev0000186
- Local PDF path: `10-writing/literatures/Degen et al. - 2020 - When redundancy is useful A Bayesian approach to .pdf`
- Tags: RSA, overspecification, continuous semantics, color/size asymmetry, Bayesian model, referring expressions

## One-Paragraph Summary

Degen et al. propose that what appears to be "overinformative" redundant modification in referring expressions is actually rational speaker behavior under a non-deterministic, continuous semantics for scalar adjectives. They extend RSA with a continuous (graded) semantics in which adjectives like "big" have uncertain applicability to objects, formalizing this uncertainty as a speaker-internal distribution over threshold values. Because color adjectives are treated as less uncertain (more deterministic) than size adjectives, the continuous RSA model predicts that redundant color modification will be more frequent than redundant size modification — the well-known color/size asymmetry — without requiring any special pragmatic stipulations. Three experiments with English participants confirm key model predictions about when redundant color vs. size modification is produced.

## Research Question

- Can overspecification (redundant modification) be explained as rational behavior under graded, uncertain adjective semantics rather than as a violation of Gricean maxims?
- What predicts the asymmetry between redundant color and redundant size modification?

## Method / Data

- Design: 3 experiments (reference game paradigms); + RSA model fitting
- Participants/Data: English-speaking participants recruited via Mechanical Turk (N varies per experiment)
- Measures: Rate of redundant color vs. size modification in referential descriptions
- Analysis approach: Bayesian RSA model with continuous (graded) adjective semantics; model predictions compared to empirical data

## Key Findings

1. Speakers produce redundant color modifiers far more frequently than redundant size modifiers across three experiments, replicating the color/size asymmetry.
2. The continuous-semantics RSA model accounts for this asymmetry: color adjectives have higher semantic certainty (lower variance threshold distributions), making them more useful even when redundant; size adjectives are more uncertain, reducing their expected utility.
3. Standard (binary-semantics) RSA predicts that speakers should never overspecify and cannot account for the asymmetry. The continuous RSA model also captures cross-experiment variation in modification rates.

## Core Takeaways For The Current Project

- The continuous semantics framework in Degen et al. is the predecessor of our model's treatment of adjective applicability. Our model uses subjectivity (faultless disagreement) as a correlate of the semantic uncertainty that drives the continuous RSA predictions here.
- The RSA structure with graded semantics is directly instantiated in our NumPyro model; the `log_beta` parameter in our model is the analog of the cost/uncertainty tradeoff in Degen et al.
- The color/size asymmetry documented here provides a benchmark for evaluating adjective ordering models: a model that correctly handles this asymmetry must treat color and size as having different semantic properties.

## Limitations / Boundary Conditions

- The model is fit to aggregate production rates, not ordering preferences within multi-adjective DPs; it does not directly address the sequencing question.
- All experiments use English, a prenominal language; cross-linguistic generalization requires additional assumptions (Waldon & Degen 2021).
- The continuous semantics requires specifying prior distributions over adjective thresholds, which are treated as free parameters; interpretability depends on model assumptions.

## Useful Quotes (Short)

- "We propose that what appears to be overinformative modification is actually rational speaker behavior given the non-deterministic semantics of scalar adjectives." (Abstract paraphrase)
- "Color adjectives are more determinate in their applicability than size adjectives... this determines the asymmetry in redundant modification rates." (Core argument)

## Relevance To Current Writing Tasks

- Supports: The claim that adjective semantics (specifically semantic uncertainty / subjectivity) drives ordering and modification preferences; the use of RSA with continuous/graded semantics.
- Contrasts with: Gricean accounts that treat overspecification as a violation; standard RSA with binary semantics.
- Open questions for us: Can the Degen et al. continuous semantics be integrated with our incremental ordering model to simultaneously account for both modification and ordering patterns?
