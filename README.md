# Multilingual Named Entity Recognition (NER) using XLM-RoBERTa

This project explores multilingual Named Entity Recognition (NER) using [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base), focusing on the performance across **German (de)**, **French (fr)**, **Italian (it)**, and **English (en)** — the four official Swiss languages. 

The study compares different fine-tuning strategies to evaluate cross-lingual generalization, single-language specialization, and language-family transferability.

---

## 🌍 Languages & Motivation

Switzerland has four national languages with different representation proportions:

- **German (de):** 62.9% of spoken language 
- **French (fr):** 22.9% of spoken language
- **Italian (it):** 8.4% of spoken language
- **English (en):** 5.9% of spoken language

The goal is to build an efficient and generalizable multilingual NER system — particularly beneficial for low-resource languages like **Italian** and **English** (in the Swiss context).

---

## 🧱 Tokenization Pipeline

Understanding XLM-R's tokenization was critical:

- **Normalization**: Clean and normalize the input text.
- **Pre-tokenization**: Define how the text is split (e.g., whitespace, punctuation).
- **Tokenization**: Use [SentencePiece](https://github.com/google/sentencepiece) to split text into subwords.
- **Post-processing**: Add special tokens like `<s>`, `</s>` for sequence boundaries.

To ensure accurate loss computation:
- Special tokens like `<s>`, `</s>` were masked using **`-100`** so they're excluded from the loss.
- If a word is split into subwords, **only the first token** is considered — the rest receive `-100`.

---

## ⚙️ Data Handling

Used Hugging Face’s `DataCollatorForTokenClassification` to:
- Dynamically pad sequences within each batch to the longest sequence.
- Maintain alignment between tokens and their entity labels after tokenization.

---

## 🧪 Experimentation Strategies

Three fine-tuning approaches were explored:

### 1. **Zero-Shot Transfer**
- Fine-tuned on **German (de)** only.
- Evaluated on all 4 languages.
- Tests cross-lingual generalization.

### 2. **Language-Specific Fine-Tuning**
- Trained a **separate model for each language**.
- Evaluated each model on its own language.

### 3. **Multilingual Fine-Tuning**
- Fine-tuned a **single model across all 4 languages**.
- Evaluated performance on each individual language.

---

## 📊 Results

| Evaluation Language | Model Trained on de | Model Trained Separately | Model Trained on All |
|---------------------|---------------------|---------------------------|-----------------------|
| **de**              | 0.8677              | 0.8677                    | 0.8682                |
| **fr**              | 0.7141              | 0.8505                    | 0.8647                |
| **it**              | 0.6923              | 0.8192                    | 0.8575                |
| **en**              | 0.5890              | 0.7068                    | 0.7870                |

> Evaluation Metric: **Test F1 Score**

---

## 🔍 Error Analysis

Conducted deep analysis across all three methods:

### ❌ Common Errors
- Identified which **tokens** were most frequently misclassified.
- Found **specific sentences** with consistently incorrect predictions.

### 🧾 Labeling Issues
- Some samples were **mislabeled** in the dataset, introducing noise.
- Especially critical in low-resource languages where each sample has higher weight.

### 🌐 Language Family Transfer
- **Fine-tuning on same language family (e.g., Romance: fr, it)** improved performance significantly.
- Shows that language similarity helps compensate for smaller dataset sizes.

---

## 🧠 Insights & Recommendations

- **Trade-off between data size and language similarity**:
  - If collecting more data is hard, prioritize fine-tuning on similar languages.
  - A well-finetuned **single multilingual model** can outperform individual models.
  
- Future improvements:
  - Manual **label correction** in training data.
  - Expand support to other multilingual NER datasets (e.g., WikiAnn).
  - Investigate **adapter layers** or **language-specific heads** for better language control.

---

## 🚀 **Final Model (Multilingual Fine-tuning)**:  
Achieved test F1-scores:  
- German (de): **0.8682**  
- French (fr): **0.8647**  
- Italian (it): **0.8575**  
- English (en): **0.7870**

---

## 🛠️ Tools & Frameworks

- [Transformers by Hugging Face](https://github.com/huggingface/transformers)
- [Datasets by Hugging Face](https://github.com/huggingface/datasets)
- PyTorch
- XLM-RoBERTa (`xlm-roberta-base`)
- SentencePiece
- `DataCollatorForTokenClassification`
