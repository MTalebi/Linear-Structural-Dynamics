# Linear Structural Dynamics

<!-- STAGE 2 ONLY (after the Zenodo release): insert the DOI badge line below, above the license badges.
[![DOI](https://zenodo.org/badge/DOI/{{CONCEPT_DOI}}.svg)](https://doi.org/{{CONCEPT_DOI}})
-->
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Code license: MIT](https://img.shields.io/badge/Code%20license-MIT-yellow.svg)](LICENSE-CODE)

A coding-first introduction to linear structural dynamics with NumPy and OpenSeesPy, built around one numerical solver for the equation of motion.

**Read the book:** <https://mtalebi.github.io/Structural-Dynamics-Book/>

<!-- STAGE 2 ONLY: add this Citation section once the concept DOI exists (same block as the book's preface).
## Citation

If this material is useful in your teaching or research, please cite it:

```bibtex
@misc{talebi2026LSDbook,
  author       = {Talebi-Kalaleh, M.},
  year         = {2026},
  title        = {Linear Structural Dynamics [e-book]},
  publisher    = {Zenodo},
  doi          = {{{CONCEPT_DOI}}},
  url          = {https://doi.org/{{CONCEPT_DOI}}}
}
```
-->

## About the book

The book teaches the dynamics of structures by building one small numerical solver for the equation of motion and handing it every kind of load. It starts from the single-degree-of-freedom model and free vibration, builds a state-space engine with exact, Newmark and central-difference integrators, and applies it to harmonic and periodic loads, arbitrary loads and earthquakes, assembling response spectra point by point. It then moves to yielding oscillators and ductility, to modal analysis of multi-story buildings, and to response-spectrum analysis and the equivalent lateral force method of ASCE 7. Every chapter combines worked examples, diagrams and interactive figures, and the numbers in the text are printed by code that the reader can run in the browser. The code uses NumPy, SciPy and OpenSeesPy, with consistent SI units throughout.

## About this repository

This repository is the public archive record of the book. It holds the citation metadata that Zenodo uses to assign the book a DOI, the Python environment that mybinder.org builds for the live code cells on the website, and the two Python modules that every chapter imports. The manuscript of the book is not part of this repository.

| File | Purpose |
| --- | --- |
| `CITATION.cff` | Citation metadata (the "Cite this repository" button on GitHub) |
| `.zenodo.json` | Metadata for the Zenodo archive |
| `requirements.txt`, `runtime.txt` | Python 3.12 environment of the live kernel |
| `sdtools.py` | The book's numerical engine: exact, Newmark and central-difference SDOF solvers, response spectra, and the synthetic ground motion |
| `osdof.py` | OpenSeesPy helpers for the SDOF and shear-building cross-checks |
| `LICENSE` | Full text of the CC BY-NC-ND 4.0 license, which covers the text and figures |
| `LICENSE-CODE` | Full text of the MIT License, which covers the code |

## License

The text and figures of the book are licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/); see [LICENSE](LICENSE) for the full text. The code, meaning `sdtools.py`, `osdof.py` and the code examples in the book, is licensed under the MIT License; see [LICENSE-CODE](LICENSE-CODE).
