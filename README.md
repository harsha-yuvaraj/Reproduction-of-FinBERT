# FinBERT Optimization

This project aims to improve the results of the sentiment analysis model **FinBERT**, as described in the paper "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models" by Dogu Araci (2019).

By engineering a custom Multi-Layer Fusion architecture with RoBERTa, improved at 87.7% accuracy and 94.49% negative recall compared to the original FinBERT baseline. The new architecture aims to address the information bottleneck in standard BERT models by preserving both low-level syntactic cues and high-level semantic meaning.

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

## Dataset Handling
Used the `atrost/financial_phrasebank` dataset from Hugging Face as the originally cited versions (`takala/financial_phrasebank`) gave loading issues with current library versions.

Combined the predefined splits (`train`, `validation`, `test`) from `atrost/financial_phrasebank` and then re-split the combined data using `train_test_split` with `test_size=0.2` twice (seed=42) to achieve the 64% Train/16% Validation/20% Test partition described in the paper.
