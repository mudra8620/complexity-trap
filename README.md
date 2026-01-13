# The Complexity Trap

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18228983.svg)](https://doi.org/10.5281/zenodo.18228983)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Paper:** Chaudhary, M. *The Complexity Trap: Robustness Limits of Subword Tokenization for Code*. Zenodo, 2026. https://doi.org/10.5281/zenodo.18228983

Official implementation of **"The Complexity Trap: Robustness Limits of Subword Tokenization for Code"**

## Key Finding

BPE tokenization creates a **complexity trap**: robustness degrades as code becomes more algorithmic.

| Representation | Linear | Conditional | Loop | Recursion | Average |
|----------------|--------|-------------|------|-----------|---------|
| Char-Level     | -56%   | -51%        | -50% | -58%      | -54%    |
| BPE (GPT)      | -20%   | -15%        | -19% | -28%      | -21%    |
| Positional AST | **0%** | **0%**      | **0%**| **0%**   | **0%**  |
| Pure AST       | **0%** | **0%**      | **0%**| **0%**   | **0%**  |

AST representations achieve **empirically perfect invariance (0% degradation)** under variable renaming.

## Installation

```bash
git clone https://github.com/mudra8620/complexity-trap.git
cd complexity-trap
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the main experiment (reproduces Table 1 in paper)
python experiments/run_full_experiment.py

# Run HumanEval validation (reproduces Table 2 in paper)
python experiments/run_humaneval.py

# Generate publication figures
python figures/generate_figures.py
```

## Results

### Synthetic Data (Table 1)
Accuracy drop under variable renaming:

| Tier | Char-Level | BPE (GPT) | Positional AST | Pure AST |
|------|------------|-----------|----------------|----------|
| Linear | -56% | -20% | 0% | 0% |
| Conditional | -51% | -15% | 0% | 0% |
| Loop | -50% | -19% | 0% | 0% |
| Recursion | -58% | -28% | 0% | 0% |

### HumanEval Validation (Table 2)
Real-world code from OpenAI's HumanEval benchmark:

| Problem Type | BPE Drop | AST Drop |
|--------------|----------|----------|
| Linear | -11% | 0% |
| Recursive | -18% | 0% |

**The Complexity Trap**: BPE degradation is **64% worse** on recursive vs linear code (18% vs 11%).

## Repository Structure

```
complexity-trap/
├── src/                    # Core modules
│   ├── representations/    # Tokenizers (Char, BPE, AST)
│   ├── mutations/          # Variable renaming engine
│   └── models/             # Transformer classifier
├── experiments/            # Reproducible experiments
├── results/                # Output data
├── figures/                # Publication figures
└── paper/                  # LaTeX source
```

## Citation

```bibtex
@misc{chaudhary2026complexity,
  author       = {Chaudhary, Mudra},
  title        = {The Complexity Trap: Robustness Limits of Subword Tokenization for Code},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18228983},
  url          = {https://doi.org/10.5281/zenodo.18228983}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

- **Author**: Mudra Chaudhary
- **Email**: mudra6820@gmail.com
