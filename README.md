<div align="center">

# Token-Level Robustness in AI-Generated Text Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20091838.svg)](https://doi.org/10.5281/zenodo.20091838)
[![arXiv](https://img.shields.io/badge/arXiv-2506.04050-b31b1b.svg)](https://arxiv.org/abs/2506.04050)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Using SHAP to expose — and improve — robustness in AI-generated text detection.*

</div>

## Paper

|                  |                                                                          |
| ---------------- | ------------------------------------------------------------------------ |
| **Title**        | Explainability-Based Token Replacement on LLM-Generated Text |
| **Authors**      | Hadi Mohammadi, Anastasia Giachanou, Daniel L. Oberski, Robert A. Bagheri |
| **Affiliation**  | Utrecht University, The Netherlands |
| **Venue**        | arXiv preprint |
| **arXiv**        | [2506.04050](https://arxiv.org/abs/2506.04050) |
| **Code archive** | [10.5281/zenodo.20091838](https://doi.org/10.5281/zenodo.20091838) (this repository, snapshot v1.0-thesis) |

> This repository accompanies **Chapter 4** of the PhD thesis
> *Let Me Explain! Explainable NLP for Understanding Large Language Models* (Hadi Mohammadi, Utrecht University, 2026).

## Abstract

Detectors of AI-generated text often rely on surface-level token patterns that can be circumvented by small, targeted edits. This work uses SHAP to identify the tokens most influential in detector decisions, then systematically replaces them to probe both adversarial weaknesses and pathways to greater robustness. Experiments on the CLIN33 corpus and AuTexTification data show that explainability-guided token replacement reveals concrete brittleness in current detectors and points to defensive training strategies.

## Citation

If you use this code or data, please cite **both** the paper and this code archive:

```bibtex
@article{mohammadi2025token,
  title         = {Explainability-Based Token Replacement on LLM-Generated Text},
  author        = {Mohammadi, Hadi and Giachanou, Anastasia and Oberski, Daniel L. and Bagheri, Robert A.},
  year          = {2025},
  journal       = {arXiv preprint},
  eprint        = {2506.04050},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2506.04050}
}

@software{mohammadi_token_replacement_2026,
  author    = {Mohammadi, Hadi and Giachanou, Anastasia and Oberski, Daniel L. and Bagheri, Robert A.},
  title     = {Token-Level Robustness in AI-Generated Text Detection},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0-thesis},
  doi       = {10.5281/zenodo.20091838},
  url       = {https://doi.org/10.5281/zenodo.20091838}
}
```

---

## Overview

We investigate how SHAP-based explanations can guide systematic token replacements that probe AI-text detector robustness, and we evaluate an ensemble detector designed to remain resilient against such replacements.

## Repository Structure

```
Token-Replacement/
├── README.md
├── LICENSE
├── CITATION.cff
├── CONTRIBUTING.md
├── requirements.txt
├── paper.pdf                                          # arXiv preprint
├── notebooks/
│   ├── Enhancing_AI_Generated_Text_Undetectability.ipynb  # Main analysis (HSR/PSR/GPT/GPT+Genre)
│   ├── CLIN33_Shared_Task_V6.ipynb                    # CLIN33 shared-task entry pipeline
│   ├── contestant_model_v1.ipynb                      # Submitted contestant model V1
│   ├── CLIN_Example.ipynb                             # CLIN baseline example
│   └── Example_full.ipynb                             # Extended example with full pipeline
├── code/
│   ├── human_eval_sample_selector.py                  # Stratified sampling for human-eval set
│   └── contestant_models/                             # CLIN33 contestant model variants (1–7)
│       ├── contestant_model.py                        # Submitted base model
│       ├── contestant_model_v1.py                     # Submission V1
│       ├── contestant_model_{1..6}.py                 # Iterative variants explored
│       ├── contestant_model_{7,corrected_7,modified_7}.py  # V7 family
│       ├── contestant_model_2_modified.py             # V2 modified
│       ├── contestant_model_3_1.py                    # V3.1
│       └── integrated_contestant_model_3.py           # Integrated V3
├── models/
│   ├── xgboost_model.pkl                              # Trained XGBoost classifier
│   ├── tfidf_vectorizer.pkl                           # Fitted TF-IDF vectorizer
│   └── bert_finetuned/                                # BERT tokenizer + training metadata
│       ├── config.json
│       ├── tokenizer.{json,_config.json}
│       ├── special_tokens_map.json
│       ├── vocab.txt
│       ├── trainer_state.json
│       └── training_args.bin
├── data/
│   ├── human_eval_samples.csv                         # Selected texts for human evaluation
│   ├── human_eval_texts_only.csv                      # Same texts, single-column form
│   ├── most_effective_tokens.csv                      # Per-token mean |SHAP| values
│   ├── big_comparison_table.csv                       # Cross-model comparison table
│   ├── big_experiment_results.csv                     # Aggregate experiment metrics
│   ├── clin33_dev/                                    # CLIN33 dev splits (en/nl × news/reviews/twitter)
│   │   └── dev_{en,nl}_{news,reviews,twitter}.csv
│   └── strategy_results/                              # Per-sample rewrites for the four strategies
│       ├── strategy{1..4}_results.csv                 # BERT-based detector
│       └── strategy{1..4}_results_xgb.csv             # XGBoost detector
├── evaluation/
│   └── human_eval_form.html                           # Annotation form (web)
└── results/
    ├── model_comparison_by_lang_domain.csv            # Detector accuracy by (model, language, domain)
    ├── overlap_SHAP_{HSR,PSR,GPT,GPT+Genre}.csv       # Sample-level flip overlap (SHAP-driven)
    ├── overlap_Random_{HSR,PSR,GPT,GPT+Genre}.csv     # Same, random-token baseline
    ├── figures/                                       # Generated plots
    └── tables/
        └── model_comparison_by_lang_domain.csv
```

> **Note on data:** CLIN33 / AuTexTification dev splits and per-sample rewrite results are included for reproducibility of the main notebook. Original CLIN33 source data is © the shared-task organisers; please cite the shared-task papers and respect their terms when redistributing or reusing.

> **Note on secrets:** API keys in the notebooks are placeholders (`hf_REDACTED_SET_VIA_ENV_VAR`, `sk-proj-REDACTED_SET_VIA_ENV_VAR`). Set your own via `os.environ["HF_TOKEN"]` / `os.environ["OPENAI_API_KEY"]` before running.

## Token Replacement Strategies

| Strategy | Description |
|----------|-------------|
| **HSR** | Human Similar Replacement — Word2Vec synonyms from human text |
| **PSR** | Part-of-Speech Replacement — HSR with POS tag matching |
| **GPT** | GPT-4o-mini prompted replacements |
| **GPT+Genre** | GPT with domain context (news, reviews, tweets) |

## Explainability Method

SHAP (Shapley Additive exPlanations) is used to compute global token importance, with a Random baseline for comparison.

## Installation

```bash
git clone https://github.com/mohammadi-hadi/Token-Replacement.git
cd Token-Replacement
pip install -r requirements.txt
```

## License

MIT License — see [LICENSE](LICENSE).

## Contact

- **Hadi Mohammadi** — Utrecht University
- Website: [mohammadi.cv](https://mohammadi.cv)
