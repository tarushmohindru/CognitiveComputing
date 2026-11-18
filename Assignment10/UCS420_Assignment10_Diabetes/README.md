# UCS420 Cognitive Computing — Assignment 10
## Diabetes Care Information Assistant (odd roll number)

A Python healthcare cognitive assistant. It understands diabetes questions, works out their intent,
reasons over rules and a knowledge graph, and answers from a domain knowledge base.

### Files
| File | Purpose |
|---|---|
| `Diabetes_Care_Assistant.ipynb` | **Main submission.** Open it in Google Colab and run all cells. |
| `diabetes_assistant.py` | The same code as a plain Python script, in the same cell order. |
| `results/sample_output.txt` | Full output of a complete run: evaluation, test cases and responses. |
| `results/confusion_matrix.png` | Confusion matrix of the deployed model. |
| `results/knowledge_graph.png` | Picture of the domain knowledge graph. |

### How to run (Google Colab)
1. Go to <https://colab.research.google.com> → **File → Upload notebook** → choose `Diabetes_Care_Assistant.ipynb`.
2. **Runtime → Run all**. Nothing needs installing, because Colab already includes every library used.
3. In §10, type a question in the chat box and press **Enter** or click **Ask**. You can ask as many
   questions as you want. If widgets don't show, run `chat_loop()` in the next cell instead (type `exit` to stop).

To run locally: `pip install scikit-learn nltk pandas matplotlib networkx ipywidgets`, then `python diabetes_assistant.py`.

### How each requirement is covered
| # | Requirement | Implementation |
|---|---|---|
| 1 | Corpus ≥ 40 queries, ≥ 5 intents | **189 labelled queries, 7 intents** (27 each): symptoms & diagnosis, types & causes, diet & nutrition, medication & treatment, monitoring, exercise & lifestyle, complications |
| 2 | Taxonomy ≥ 4 categories × 2 subcategories | 5 major categories, each with 2 subcategories and leaf topics (Understanding Diabetes, Clinical Management, Lifestyle Management, Complications, Safety & Escalation) |
| 3 | ≥ 8 concepts, ≥ 10 relationships | **20 concepts, 24 typed relations** stored as triples. A BFS over the graph finds the chain of relations linking concepts mentioned in a query |
| 4 | Lowercase, tokenize, stop-words, lemmatize | NLTK `word_tokenize`, English stop-word list (negations kept), WordNet lemmatizer |
| 5 | TF-IDF + NB / LR / SVM | Word (1–2-gram) + character (3–5-gram) TF-IDF. Naive Bayes, Logistic Regression and Linear SVM are all trained and compared |
| 6 | Accuracy, precision, recall, F1, confusion matrix | Held-out test set (25%) plus 5-fold stratified CV. Deployed LR: **accuracy 0.79, macro F1 0.79, CV F1 0.77** |
| 7 | Knowledge base per intent | 7 intents × (overview + 4–9 topic entries). The topic is picked by keyword overlap with the lemmatized query |
| 8 | ≥ 5 rules incl. emergency, low-confidence, unsupported | **8 rules:** R1 emergency escalation, R2 critical glucose reading, R3 greeting, R4 unsupported query, R5 low confidence (p < 0.40 or margin < 0.10), R6 glucose-reading interpretation, R7 medication-change safety, R8 pregnancy context, plus a disclaimer on every answer |
| 9 | Integrated assistant | `DiabetesCareAssistant.respond()`: NLP → classifier → rule engine → KB retrieval + KG reasoning → response |
| 10 | Interactive Colab interface | ipywidgets chat UI (text box, Ask and Clear buttons, scrolling history) plus an `input()` loop as a fallback |

### Test results (§11)
| Test | Query | Result |
|---|---|---|
| Each intent (×7) | e.g. "Is it okay for a diabetic to eat mango?" | Correct intent for all 7 (confidence 52–100%) |
| Differently worded (×3) | "My mouth is always dry and I keep running to the toilet at night — should I worry?" | Symptoms & Diagnosis ✔ |
| Unsupported (×2) | "How do I treat a migraine headache?" | R4: out of scope ✔ |
| Low confidence (×2) | "Is this normal for a diabetic?" | R5: asks the user to clarify between the top two intents ✔ |
| Emergency (×2) | "My diabetic father has fainted and is not responding!" | R1: emergency escalation (112 / 108) ✔ |

*This assistant gives general educational information only. It is not medical advice.*
