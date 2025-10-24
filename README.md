# FinBERT Reproduction

This project aims to reproduce the results of the sentiment analysis model **FinBERT**, as described in the paper "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models" by Dogu Araci (2019).

## Original Paper
* **Title:** FinBERT: Financial Sentiment Analysis with Pre-trained Language Models
* **Author:** Dogu Tan Araci
* **Year:** 2019
* **Abstract Link:** [https://arxiv.org/abs/1908.10063v1](https://arxiv.org/abs/1908.10063v1)
* **PDF Link:** [https://arxiv.org/pdf/1908.10063](https://arxiv.org/pdf/1908.10063)

## Environment & Requirements
This code was developed and tested using **Google Colab** with a **T4 GPU**.

* **Key Libraries:**
    * `transformers`
    * `torch`
    * `datasets`
    * `evaluate`
    * `scikit-learn`
    * `matplotlib`
    * `numpy`

## How to Run
1.  **Upload:** Upload the Jupyter Notebook (`.ipynb`) file provided in this repository to Google Colab.
2.  **Select GPU:** Ensure a GPU runtime is selected (T4 GPU).
3.  **Run All:** Execute all cells sequentially from top to bottom.

## Reproducibility
* **Seeds:** Random seeds (`seed=42`) are set for dataset splitting (`train_test_split`) and within the `TrainingArguments` to ensure consistent results across runs.
* **Dataset:** The `atrost/financial_phrasebank` dataset is used, and the data is programmatically re-split to match the paper's 64% train / 16% validation / 20% test partition.

## Attribution
* This work is a reproduction based on the original FinBERT paper by Dogu Araci.
* The overall code structure for loading data, tokenizing, training, and evaluation follows standard examples and best practices from the Hugging Face `transformers` and `datasets` libraries documentation.

## Key Modifications for Reproduction
To accurately reproduce the paper's methodology, the following specific techniques were implemented:

1.  **Slanted Triangular Learning Rate (STLR):** Implemented using the `lr_scheduler_type="linear"` and `warmup_ratio=0.2` parameters within `TrainingArguments`, matching the paper's description.
2.  **Discriminative Fine-Tuning (DFT):** Implemented by subclassing the `Trainer` class (`DiscriminativeLRsTrainer`) to create separate parameter groups for the BERT base model and the classification head, applying a lower learning rate (0.85 discrimination rate) to the base model parameters as specified in the paper.
3.  **Gradual Unfreezing (GU):** Implemented using a custom `TrainerCallback` (`GradualUnfreezingCallback`) that unfreezes one layer of the BERT encoder every third of a training epoch, starting from the top layer, as described in the paper.
4.  **Dataset Handling:**
    * Used the `atrost/financial_phrasebank` dataset from Hugging Face as the originally cited versions (`takala/financial_phrasebank`) gave loading issues with current library versions.
    * Combined the predefined splits (`train`, `validation`, `test`) from `atrost/financial_phrasebank` and then re-split the combined data using `train_test_split` with `test_size=0.2` twice (seed=42) to achieve the 64% Train/16% Validation/20% Test partition described in the paper.