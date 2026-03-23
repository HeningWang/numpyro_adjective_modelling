# An Information-Theoretic Explanation of Adjective Ordering Preferences

## Metadata

- Authors: Michael Hahn, Judith Degen, Noah Goodman, Dan Jurafsky, Richard Futrell
- Year: 2018
- Venue: Proceedings of the 40th Annual Conference of the Cognitive Science Society (CogSci 2018), pp. 1766–1771
- DOI / URL: https://escholarship.org/uc/item/6c81w2c5
- Local PDF path: `10-writing/literatures/Hahn et al. - An Information-Theoretic Explanation of Adjective Ordering Preferences.pdf`
- Tags: adjective ordering, information theory, subjectivity, mutual information, RSA, corpus study, Monte Carlo simulation

## One-Paragraph Summary

Hahn et al. present a corpus study and a formal rational model showing that adjective ordering preferences are predicted jointly by adjective subjectivity and the mutual information between adjectives. Through corpus analysis they replicate the subjectivity effect and discover a new effect: pairs of adjectives with lower mutual information (more independent meanings) show stronger ordering preferences. They then propose a rational RSA-style model that incorporates both an incremental listener with sequentially intersective semantics and a speaker whose cost function is based on a language model, deriving the prediction that ordering adjectives by decreasing subjectivity maximizes communicative success — with memory limitations (progressive noise in the listener's buffer) providing the formal mechanism by which subjectivity and mutual information both affect preferences.

## Research Question

- Why does adjective subjectivity predict ordering preferences, and what additional factors (mutual information) explain residual variance?
- Can an information-theoretic rational model derive these effects from first principles?

## Method / Data

- Design: Corpus study (ordering preferences from British National Corpus) + formal rational model + Monte Carlo simulation
- Participants/Data: Corpus of adjective pair co-occurrences extracted from BNC; no experimental participants
- Measures: Corpus-based ordering preferences; model-predicted ordering probabilities
- Analysis approach: Mixed-effects regression on corpus data; formal model derivation; Monte Carlo simulation of communicative success

## Key Findings

1. Adjective subjectivity predicts corpus ordering preferences (replicating Scontras et al. 2017).
2. Lower mutual information between adjective pairs predicts stronger ordering preferences — this is a novel corpus finding not previously documented.
3. A formal RSA model with incremental (sequentially intersective) interpretation and memory noise explains both effects: when memory is imperfect, the first adjective in a string may be lost by the time the noun is reached, and more objective adjectives (placed closer to the noun, interpreted last in the incremental composition) are more reliable as a result.
4. The model predicts that orderings respecting decreasing subjectivity produce higher expected communicative success via a Monte Carlo simulation.

## Core Takeaways For The Current Project

- This paper provides an explicit information-theoretic mechanism (memory-limited incremental interpretation) that derives the subjectivity preference — one step deeper than the Franke et al. "communicative success" argument.
- The mutual information finding is practically important: if two adjectives have high mutual information (i.e., one implies the other), ordering preferences should be weaker. Our stimuli should ideally control for adjective-adjective correlations.
- The RSA framework used here (incremental listener with memory noise) is very close to our incremental speaker model — the same sequential intersective logic applies on both the speaker and listener sides.

## Limitations / Boundary Conditions

- Corpus-based: ordering preferences extracted from corpora are influenced by genre and conventional usage, not just processing preferences.
- Memory-based mechanism is stipulated (progressive deletion probability) rather than empirically verified independently.
- The model is not fit to empirical data via Bayesian inference; it is a simulation demonstrating a plausible mechanism.

## Useful Quotes (Short)

- "Across languages, adjectives are subject to ordering restrictions. Recent research shows that these are predicted by adjective subjectivity, but the question remains open why this is the case." (Abstract)
- "We first conduct a corpus study and not only replicate the subjectivity effect, but also find a previously undocumented effect of mutual information on adjective ordering." (Abstract)

## Relevance To Current Writing Tasks

- Supports: The subjectivity-based motivation for our model; the incremental interpretation mechanism; the connection between information theory and RSA-based ordering.
- Contrasts with: Static semantic hierarchy accounts (Dixon, Cinque); accounts that do not incorporate memory limitations.
- Open questions for us: We do not model mutual information between adjectives; should our prior over adjective pairs in the experiment include adjective-adjective correlation as a control variable?
