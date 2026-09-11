# Communicative Efficiency in Adjective Ordering

**Plan-Guided Production between Hierarchical Composition and Incremental Pragmatics**
Hening Wang, Fabian Schlotterbeck, and Michael Franke · University of Tübingen

## Editing the manuscript

The current manuscript is [paper/draft.tex](paper/draft.tex); the compiled version is
[paper/draft.pdf](paper/draft.pdf). Edit this source directly. References are in
[paper/references.bib](paper/references.bib), and all figures used in the main text
and appendices are in [paper/figures](paper/figures).

From the repository root, build with:

```sh
make draft
```

This requires a TeX installation with `latexmk`, `biber`, and the packages imported
by the manuscript. It rebuilds the source and writes `paper/draft.pdf`; temporary
files stay in `paper/build/`. Python, R, inference outputs, and Git LFS are not
needed to edit or compile the paper. `make clean` removes temporary build files.

## Contents

| Location | Purpose |
|---|---|
| [paper/](paper/) | Current manuscript, bibliography, styles, and figures |
| [paper/data/](paper/data/) | Frozen numerical summaries used in the paper and figures |
| [paper/scripts/](paper/scripts/) | Figure-generation scripts; see the [figure guide](paper/README.md) |
| [data/](data/) | Experimental data, stimulus inventory, and the production model input |
| [models/production/](models/production/) | Production models, inference, validation, and the GPT-2 order feature |
| [analysis/](analysis/) | Behavioural analysis, predictive diagnostics, and summary exports |
| [simulation/](simulation/) | Random-scene simulations and predictions on experimental displays |

The manuscript compares global, plan-guided, and fully incremental production
under context-fixed semantics and sequential context updating. All production
models predict the same fifteen utterance categories. The behavioural studies
measure slider preferences and the selection, length, and ordering of produced
adjectives.

## Computational work

The figures and numerical summaries are included so manuscript editing is
independent of computational reruns. See [analysis/README.md](analysis/README.md)
for the current analysis entry points and [data/Readme.md](data/Readme.md) for the
data inventory. Large posterior files and generated run directories are stored
separately and are not included in this checkout.

Python dependencies are listed in [requirements.txt](requirements.txt).
Figure scripts use R with `ggplot2`, `dplyr`, `tidyr`, `readr`, `scales`, and
`patchwork`. Production inference uses CUDA and 64-bit JAX arithmetic.

## License

See [LICENSE](LICENSE).
