# Deep-Learning-Algorithms-for-Mixed-Classification-Sentiment-Analysis-in-Norwegian-Text

This repository contains the code and results of a Bachelor's thesis project focused on mixed-classification sentiment analysis in Norwegian text. We explore various deep learning models, including RNNs, CNNs, and transformer-based models, specifically tuned for Norwegian datasets.

**Authors**: Isak Killingrød, Jon Ingvar Jonassen Skånøy

**Supervisor**: Mohammad Arif Payenda
## Evaluation 
Grade: A

Justification: The candidates have implemented and tested several machine learning models and ensembles of models on data that either label a given sentence as positive/negative/neutral for "ternary" labels or positive/negative/neutral/mixed for "mixed" labels. They have optimized the results for the same dataset's ternary and mixed-label variants by fine-tuning singular models and creating and testing various ensemble models.


The candidates' work is characterized by its exceptional quality, and they have achieved commendable results. They have adeptly employed complex algorithms and conducted exemplary scientific research, demonstrating remarkable self-sufficiency. The practical applications of their findings carry considerable significance.


The thesis is well-written and academic. The problem statement and objective are clearly articulated. The background section provides detailed descriptions. The experiments are conducted to a high standard, with comprehensive documentation of the research process and results. The results and conclusion are presented in great detail, and there is a strong element of self-reflection throughout.
## Introduction
This project investigates the impact of introducing a mixed sentiment label in Norwegian text datasets and evaluates the performance of various machine learning models under this configuration. The goal is to improve sentiment analysis by accurately classifying nuanced sentiments that traditional binary or ternary classification might overlook.

## Background and Motivation
Sentiment analysis is essential in natural language processing, with applications in areas such as market analysis and social media monitoring. While most research has concentrated on languages like English, Norwegian sentiment analysis remains underexplored, particularly in the context of mixed sentiment labels. This project aims to bridge that gap by evaluating how deep learning models handle mixed sentiments in Norwegian text.

## Problem Statement
The primary challenge addressed in this project is the effect of introducing a mixed sentiment label (positive/negative) into a Norwegian sentiment analysis dataset. The project assesses how this affects the performance of various models, including both individual models and ensembles.

## Objectives
- **Data Relabeling**: Create a Norwegian dataset labeled with positive, negative, neutral, and mixed sentiments.
- **Model Training**: Implement and fine-tune various deep learning models, including GRUs, BiGRUs, LSTMs, BiLSTMs, CNNs, and transformers.
- **Performance Evaluation**: Compare the performance of models on datasets with and without the mixed label.
- **Ensembling**: Investigate the impact of ensemble methods on model performance.

## Methodology
The project began with relabeling an existing Norwegian dataset to include mixed sentiment labels, followed by implementing several deep learning models. The architectures explored include RNN-based models, CNNs, and transformers like NorBERT and NorT5. Each model was trained on both the relabeled mixed dataset and a traditional ternary dataset for comparison.

## Implementation Details

### Models
- **NorBERT**: A fine-tuned BERT model specifically trained on Norwegian text.
- **NorT5**: A T5 model adapted for Norwegian, used for text-to-text sentiment classification.
- **RNN Variants**: Includes GRU, BiGRU, LSTM, and BiLSTM models.
- **CNN**: A convolutional neural network embedded with NorBERT.

### Training
Training was conducted on GPUs with PyTorch, using hyperparameter optimization and patience factors to prevent overfitting. Models were evaluated using precision, recall, accuracy, and F1 score.

**Note**: The datasets used in this project are sourced from a closed Hugging Face dataset.

## Results
The results indicate that models generally perform worse on datasets with mixed labels due to increased complexity. However, ensemble methods significantly improved performance, with diverse models contributing to more accurate classifications.

## Conclusion
The introduction of mixed sentiment labels presents challenges for deep learning models in sentiment analysis. Nevertheless, leveraging ensemble techniques and advanced models like NorT5 can mitigate some difficulties and improve performance.





