# Speakers and Listeners Exploit Word Order for Communicative Efficiency: A Cross-Linguistic Investigation

## Metadata

- Authors: Paula Rubio-Fernández, Francis Mollica, Julian Jara-Ettinger
- Year: 2021
- Venue: Journal of Experimental Psychology: General, 150(3), 583–594
- DOI / URL: 10.1037/xge0000963
- Local PDF path: `10-writing/literatures/Rubio-Fernandez et al. - 2021 - Speakers and listeners exploit word order for comm.pdf`
- Tags: incremental efficiency hypothesis, word order, cross-linguistic, English, Spanish, color adjectives, redundancy, eye-tracking, referential communication

## One-Paragraph Summary

This paper introduces and tests the *incremental efficiency hypothesis*: the efficiency of a referential expression must be calculated incrementally, over the full visual context as it unfolds word by word, not just globally with respect to competitors of the same kind. Production data (Exp. 1) confirm that English speakers produce more redundant color adjectives than Spanish speakers, but both groups use more in denser displays — showing language-specific efficiency pressures modulated by visual demands. Two eye-tracking experiments (2a, 2b) show that English listeners establish color contrast *across categories* (guided by color before the noun), while Spanish listeners establish it *within categories* (guided by noun before color) — and Spanish listeners switch to the English strategy when tested in English. This directly demonstrates that word order determines the pragmatic affordances of color adjectives.

## Research Question

- Does adjective position (prenominal vs. postnominal) affect the efficiency of redundant color adjectives, and does this explain cross-linguistic differences in overspecification?

## Method / Data

- Design: Language production experiment (Exp. 1: English vs. Spanish; 4-item vs. 16-item displays) + two eye-tracking experiments (Exp. 2a, 2b: English vs. Spanish listeners; shape-competitor vs. color-competitor displays)
- Measures: Rate of redundant color adjectives (Exp. 1); fixation proportions to target, shape competitor, color competitor (Exps. 2a-b)
- Analysis approach: Mixed-effects models; cross-linguistic comparison; within-subjects language switch (Exp. 2b)

## Key Findings

1. English speakers produce more redundant color adjectives than Spanish speakers, but both increase use in denser displays — consistent with incremental efficiency pressure modulated by visual difficulty.
2. English listeners establish color contrast across categories (BLUE SHAPES → triangular one), while Spanish listeners do so within categories (TRIANGLES → blue one) — different search strategies from same visual display.
3. Spanish listeners switch to the English (across-category) strategy when immediately tested in English, confirming that word order, not language community, drives the effect.

## Core Takeaways For The Current Project

- The incremental efficiency hypothesis formalizes that **prenominal adjective = guide visual search by property first**, making color-first ordering more efficient in prenominal languages. This is exactly the mechanism our incremental RSA speaker implements.
- The cross-linguistic production result (English > Spanish redundancy, both increase with density) maps directly onto what our model should reproduce for ordering preference.
- "Pragmatic contrast is not a processing constraint" — incremental contrast (across categories) precedes and shapes pragmatic inference; relevant to how our model's incremental speaker differs from the global speaker.

## Limitations / Boundary Conditions

- Only two languages compared (English, Spanish) in production; cross-linguistic comprehension tested in Exps. 2a-b.
- Eye-tracking paradigm measures online processing, not static ordering preference as in our slider experiment.
- Does not test adjective ordering per se, but redundancy rate — connection to noun-phrase ordering preferences requires additional inference.

## Useful Quotes (Short)

- "The incremental efficiency hypothesis: the efficiency of a referential expression must be calculated incrementally, over the entire visual context." (p. 2)
- "Speakers and listeners of different languages exploit word order to increase communicative efficiency." (abstract)
- "Pragmatic contrast is not a processing constraint." (abstract)

## Relevance To Current Writing Tasks

- Supports: The theoretical motivation for the incremental speaker in our RSA model; the cross-linguistic production results provide a benchmark for what our model should predict
- Contrasts with: Global RSA accounts treating efficiency as computed over the full, final description
- Open questions for us: Can our model reproduce the density modulation effect (both English and Spanish increase redundancy in denser displays)?
