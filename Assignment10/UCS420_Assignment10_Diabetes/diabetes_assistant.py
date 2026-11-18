# %% [markdown]
# # UCS420: Cognitive Computing — Assignment 10
# ## Diabetes Care Information Assistant (Odd Roll Number)
#
# A Python-based **Healthcare Cognitive Assistant** that understands user queries about diabetes,
# identifies their intent, reasons over a domain knowledge graph and rule base, retrieves relevant
# information from a knowledge base and generates an appropriate response.
#
# **Pipeline:** `User query → NLP preprocessing → TF-IDF → Intent classifier → Rule engine (reasoning) → Knowledge retrieval + Knowledge-graph inference → Response`
#
# | # | Requirement | Section |
# |---|---|---|
# | 1 | Labelled corpus ≥ 40 queries, ≥ 5 intents | §1 |
# | 2 | Hierarchical taxonomy ≥ 4 categories × 2 sub-categories | §2 |
# | 3 | ≥ 8 domain concepts, ≥ 10 relationships | §3 |
# | 4 | Lowercasing, tokenization, stop-word removal, lemmatization | §4 |
# | 5 | TF-IDF + Naive Bayes / Logistic Regression / SVM | §5 |
# | 6 | Accuracy, precision, recall, F1, confusion matrix | §6 |
# | 7 | Domain knowledge base for every intent | §7 |
# | 8 | ≥ 5 rules incl. emergency, low-confidence, unsupported | §8 |
# | 9 | Integrated assistant | §9 |
# | 10 | Interactive Colab interface (multiple queries) | §10 |
# | — | Required test cases | §11 |
#
# > ⚠️ *This assistant provides general educational information only and is not a substitute for professional medical advice.*

# %% [markdown]
# ### §0 Setup — imports and NLTK resources
# All libraries used here (`nltk`, `scikit-learn`, `matplotlib`, `networkx`, `ipywidgets`, `pandas`) are pre-installed in Google Colab.

# %%
import re
import sys
import string
import warnings
from collections import Counter, deque

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# Download the NLTK resources needed for tokenization, stop words and lemmatization.
for _pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass  # offline: the fallbacks below are used instead

print("Setup complete.")

# %% [markdown]
# ## §1 Labelled Corpus (Requirement 1)
# **189 labelled queries across 7 intent categories** (27 per intent, balanced).

# %%
CORPUS = {
    "symptoms_diagnosis": [
        "What are the common symptoms of diabetes?",
        "How do I know if I have diabetes?",
        "Is frequent urination a sign of diabetes?",
        "Why am I always thirsty and tired all the time?",
        "What are the early warning signs of high blood sugar?",
        "How is diabetes diagnosed by doctors?",
        "Which blood test confirms diabetes?",
        "I have blurred vision and sudden weight loss, could it be diabetes?",
        "What are the signs of type 1 diabetes in children?",
        "Do slow healing wounds mean I might have diabetes?",
        "What symptoms does prediabetes show?",
        "I feel very hungry and urinate a lot at night, is that diabetes?",
        "My skin has dark patches on the neck, is it a symptom of diabetes?",
        "Which signs show that my blood sugar might be too high?",
        "Can diabetes be present without any symptoms?",
        "What is the fasting glucose test used for diagnosis?",
        "I feel tired, thirsty and lose weight without trying, what could it be?",
        "Should I get tested for diabetes?",
        "What are the classic symptoms of high blood sugar like thirst and frequent urination?",
        "Is excessive thirst a symptom of diabetes?",
        "Can blurred vision be an early sign of diabetes?",
        "What tests are done to diagnose diabetes?",
        "What are the symptoms of type 2 diabetes in adults?",
        "How do doctors confirm a diabetes diagnosis?",
        "Are frequent infections and itching signs of diabetes?",
        "What are the warning symptoms of diabetes in women?",
        "What level of fasting sugar is used to diagnose diabetes?",
    ],
    "types_causes": [
        "What are the different types of diabetes?",
        "What is the difference between type 1 and type 2 diabetes?",
        "What causes diabetes?",
        "Is diabetes hereditary or genetic?",
        "What is gestational diabetes?",
        "What are the risk factors for type 2 diabetes?",
        "Can eating too much sugar cause diabetes?",
        "What is prediabetes and what does it mean?",
        "Is type 1 diabetes an autoimmune disease?",
        "Can obesity lead to diabetes?",
        "What is insulin resistance?",
        "What is diabetes and how does it affect the body?",
        "Why do people develop type 2 diabetes?",
        "Does family history increase my chance of getting diabetes?",
        "What role does the pancreas play in diabetes?",
        "Can stress or lack of activity cause diabetes?",
        "Who is at higher risk of becoming diabetic?",
        "What is type 1 diabetes?",
        "What causes type 1 diabetes?",
        "What causes type 2 diabetes?",
        "Is diabetes genetic if my parents have it?",
        "What are the main risk factors for developing diabetes?",
        "Why does the body become resistant to insulin?",
        "Can being overweight cause type 2 diabetes?",
        "Who gets gestational diabetes during pregnancy?",
        "What is the cause of prediabetes?",
        "How many types of diabetes are there?",
    ],
    "diet_nutrition": [
        "What foods should a diabetic eat?",
        "Which foods should I avoid with diabetes?",
        "Can a diabetic person eat rice?",
        "Are fruits safe to eat if I have diabetes?",
        "What is the glycemic index of food?",
        "What is a good breakfast for diabetics?",
        "How do I count carbohydrates in my meals?",
        "Can diabetics eat sweets and desserts?",
        "Is honey or jaggery better than sugar for diabetics?",
        "Suggest a diet plan for type 2 diabetes",
        "Can I drink alcohol if I have diabetes?",
        "Are artificial sweeteners safe for diabetic patients?",
        "Is it okay to eat potatoes or bread with diabetes?",
        "What snacks can a diabetic have between meals?",
        "Should I eat roti or rice for dinner?",
        "How much sugar can a diabetic consume in a day?",
        "Is milk or tea with sugar bad for diabetics?",
        "Which vegetables and proteins are best for controlling sugar?",
        "What foods are good for lowering blood sugar?",
        "Which fruits can a diabetic eat?",
        "Is brown rice better than white rice for diabetics?",
        "How many carbs should a diabetic eat per meal?",
        "What should I eat for lunch and dinner with diabetes?",
        "Which foods have a low glycemic index?",
        "Can diabetics eat chocolate or ice cream?",
        "Is a low carb or keto diet good for diabetes?",
        "What drinks are safe for diabetics?",
    ],
    "medication_treatment": [
        "How does metformin work?",
        "What are the side effects of metformin?",
        "When should I take my insulin injection?",
        "What are the different types of insulin?",
        "How do I inject insulin correctly?",
        "Can I stop taking my diabetes medicine if my sugar is normal?",
        "What should I do if I miss a dose of my diabetes tablet?",
        "How should insulin be stored?",
        "Which tablets are used to treat type 2 diabetes?",
        "What is an insulin pump?",
        "Does insulin cause weight gain?",
        "What treatment options are available for diabetes?",
        "Where should I give my insulin shot on the body?",
        "Is it safe to take metformin for a long time?",
        "What is the difference between basal and bolus insulin?",
        "Can I take my diabetes pills on an empty stomach?",
        "What are GLP-1 injections like semaglutide used for?",
        "How many units of insulin should I take?",
        "What medicines are prescribed for type 2 diabetes?",
        "What is the right way to take metformin?",
        "How does insulin therapy work?",
        "What is the difference between rapid acting and long acting insulin?",
        "I forgot to take my diabetes medicine, what should I do?",
        "Can insulin be kept outside the fridge?",
        "Do diabetes tablets have side effects?",
        "Should metformin be taken with food?",
        "Will I need insulin injections for type 2 diabetes?",
    ],
    "monitoring": [
        "What is a normal blood sugar level?",
        "How often should I check my blood sugar?",
        "What is HbA1c test?",
        "What should my target HbA1c be?",
        "How do I use a glucometer at home?",
        "What is the normal fasting blood sugar range?",
        "What should my sugar level be after meals?",
        "What is a continuous glucose monitor?",
        "What is the best time to test my glucose?",
        "My sugar reading is 180, what does it mean?",
        "How should I record and track my glucose readings?",
        "Why is my morning fasting sugar high?",
        "Is a glucose reading of 140 after eating normal?",
        "How often should I get my HbA1c checked?",
        "What range should my blood glucose stay in?",
        "How accurate are home glucose meters?",
        "Should I check sugar before or after food?",
        "What does a high glucose reading on my meter mean?",
        "What should my blood sugar be before breakfast?",
        "What is a good HbA1c level?",
        "How do I check my blood glucose with a meter?",
        "What is the normal range for random blood sugar?",
        "How many times a day should I test my sugar?",
        "Is a fasting sugar of 110 normal?",
        "What does an HbA1c of 8 percent mean?",
        "How do CGM sensors measure glucose?",
        "What are the target blood glucose levels for diabetics?",
    ],
    "exercise_lifestyle": [
        "What exercise is good for diabetes?",
        "How long should a diabetic walk every day?",
        "Can yoga help control diabetes?",
        "Is it safe to exercise with diabetes?",
        "Does stress affect blood sugar levels?",
        "How does sleep affect diabetes?",
        "Will losing weight help my diabetes?",
        "Should I quit smoking if I am diabetic?",
        "Can I work out when my sugar is high?",
        "Is strength training at the gym good for diabetics?",
        "How can lifestyle changes reverse prediabetes?",
        "How many minutes of physical activity per week do diabetics need?",
        "Does walking after meals lower blood sugar?",
        "What lifestyle habits help manage diabetes?",
        "Is cycling or swimming good for people with diabetes?",
        "How does being active improve insulin sensitivity?",
        "Can meditation reduce my sugar levels?",
        "How can I stay fit and active with type 2 diabetes?",
        "Which exercises lower blood sugar the most?",
        "How much exercise should a diabetic do every week?",
        "Is brisk walking enough exercise for diabetes?",
        "How do I avoid low sugar during exercise?",
        "Does smoking make diabetes worse?",
        "How does lack of sleep affect blood sugar?",
        "Can managing stress help control diabetes?",
        "How much weight should I lose to improve diabetes?",
        "Are there yoga poses that help diabetics?",
    ],
    "complications": [
        "What are the long term complications of diabetes?",
        "How does diabetes affect the eyes?",
        "How should I take care of my feet with diabetes?",
        "Can diabetes damage the kidneys?",
        "Why do I have nerve pain and numbness in my feet?",
        "What is hypoglycemia and how do I treat low sugar?",
        "What is diabetic ketoacidosis?",
        "Does diabetes increase the risk of heart disease?",
        "What happens if blood sugar stays high for years?",
        "What is diabetic retinopathy?",
        "Why do diabetics get infections easily?",
        "What is diabetic neuropathy?",
        "Can diabetes cause blindness?",
        "Why do diabetic foot ulcers lead to amputation?",
        "What are the signs of kidney damage in diabetics?",
        "Can uncontrolled sugar cause a heart attack or stroke?",
        "What are the symptoms of low blood sugar?",
        "Why do my feet burn and tingle at night?",
        "What eye problems can diabetes cause?",
        "How do I prevent diabetic retinopathy?",
        "What are the symptoms of diabetic kidney disease?",
        "How do I prevent nerve damage from diabetes?",
        "What are the signs of diabetic ketoacidosis?",
        "How do I treat hypoglycemia quickly?",
        "Why are diabetics at risk of heart attack?",
        "How can I prevent foot ulcers with diabetes?",
        "What are the complications of uncontrolled diabetes?",
    ],
}

INTENT_LABELS = {
    "symptoms_diagnosis": "Symptoms & Diagnosis",
    "types_causes": "Types & Causes",
    "diet_nutrition": "Diet & Nutrition",
    "medication_treatment": "Medication & Treatment",
    "monitoring": "Blood Glucose Monitoring",
    "exercise_lifestyle": "Exercise & Lifestyle",
    "complications": "Complications",
}

df = pd.DataFrame([(q, intent) for intent, qs in CORPUS.items() for q in qs],
                  columns=["query", "intent"])
print(f"Total labelled queries: {len(df)} | Intent categories: {df.intent.nunique()}")
print(df.intent.value_counts().to_string())
df.sample(8, random_state=1)

# %% [markdown]
# ## §2 Hierarchical Taxonomy (Requirement 2)
# 5 major categories, each with 2+ sub-categories and leaf topics. Every intent is mapped to a node in the taxonomy.

# %%
TAXONOMY = {
    "Diabetes Care": {
        "1. Understanding Diabetes": {
            "1.1 Types & Causes": ["Type 1", "Type 2", "Gestational", "Prediabetes", "Risk factors"],
            "1.2 Symptoms & Diagnosis": ["Classic symptoms", "Warning signs", "Diagnostic tests"],
        },
        "2. Clinical Management": {
            "2.1 Medication & Insulin": ["Oral medicines (Metformin)", "Insulin types", "Injection & storage"],
            "2.2 Blood Glucose Monitoring": ["Target ranges", "HbA1c", "Glucometer / CGM"],
        },
        "3. Lifestyle Management": {
            "3.1 Diet & Nutrition": ["Plate method", "Carbohydrates & GI", "Sweets, fruit & alcohol"],
            "3.2 Physical Activity & Wellbeing": ["Exercise", "Weight, sleep & stress", "Smoking"],
        },
        "4. Complications": {
            "4.1 Acute Complications": ["Hypoglycemia", "Diabetic ketoacidosis (DKA)"],
            "4.2 Chronic Complications": ["Retinopathy", "Nephropathy", "Neuropathy & foot care", "Heart disease"],
        },
        "5. Safety & Escalation (rule-based)": {
            "5.1 Medical Emergencies": ["Unconsciousness / seizures", "Critical glucose readings"],
            "5.2 Out-of-scope Queries": ["Non-diabetes health topics", "Non-health topics"],
        },
    }
}

# Mapping of each classifier intent to its taxonomy path
INTENT_TO_TAXONOMY = {
    "types_causes": ["Diabetes Care", "1. Understanding Diabetes", "1.1 Types & Causes"],
    "symptoms_diagnosis": ["Diabetes Care", "1. Understanding Diabetes", "1.2 Symptoms & Diagnosis"],
    "medication_treatment": ["Diabetes Care", "2. Clinical Management", "2.1 Medication & Insulin"],
    "monitoring": ["Diabetes Care", "2. Clinical Management", "2.2 Blood Glucose Monitoring"],
    "diet_nutrition": ["Diabetes Care", "3. Lifestyle Management", "3.1 Diet & Nutrition"],
    "exercise_lifestyle": ["Diabetes Care", "3. Lifestyle Management", "3.2 Physical Activity & Wellbeing"],
    "complications": ["Diabetes Care", "4. Complications"],  # sub-category chosen by retrieved topic
}


def print_tree(node, prefix=""):
    items = list(node.items()) if isinstance(node, dict) else [(leaf, None) for leaf in node]
    for i, (name, child) in enumerate(items):
        last = i == len(items) - 1
        print(prefix + ("└── " if last else "├── ") + name)
        if child is not None:
            print_tree(child, prefix + ("    " if last else "│   "))


root = next(iter(TAXONOMY))
print(root)
print_tree(TAXONOMY[root])

# %% [markdown]
# ## §3 Domain Concepts & Relationships — Knowledge Graph (Requirement 3)
# **20 domain concepts** and **24 typed relationships** stored as `(subject, relation, object)` triples.
# The graph is used at run time for *reasoning*: when a query mentions two concepts, the assistant
# searches for a chain of relations that connects them and explains it.

# %%
CONCEPTS = {
    # concept           : aliases searched in the user's text
    "Diabetes Mellitus":   ["diabetes", "diabetic"],
    "Type 1 Diabetes":     ["type 1", "type one", "type-1", "t1d"],
    "Type 2 Diabetes":     ["type 2", "type two", "type-2", "t2d"],
    "Gestational Diabetes": ["gestational", "pregnancy", "pregnant"],
    "Pancreas":            ["pancreas", "beta cell"],
    "Insulin":             ["insulin"],
    "Blood Glucose":       ["blood sugar", "sugar level", "glucose", "sugar"],
    "Insulin Resistance":  ["insulin resistance", "resistance"],
    "Obesity":             ["obesity", "obese", "overweight", "weight", "belly fat"],
    "Carbohydrates":       ["carbohydrate", "carb", "rice", "bread", "starch"],
    "Exercise":            ["exercise", "walk", "walking", "workout", "yoga", "gym", "physical activity"],
    "Metformin":           ["metformin"],
    "HbA1c":               ["hba1c", "a1c", "glycated"],
    "Hypoglycemia":        ["hypoglycemia", "low sugar", "low blood sugar", "hypo"],
    "Hyperglycemia":       ["hyperglycemia", "high sugar", "high blood sugar"],
    "Diabetic Ketoacidosis": ["ketoacidosis", "dka", "ketone"],
    "Neuropathy":          ["neuropathy", "nerve", "numbness", "tingling"],
    "Retinopathy":         ["retinopathy", "eye", "vision"],
    "Nephropathy":         ["nephropathy", "kidney"],
    "Cardiovascular Disease": ["heart", "cardiovascular", "stroke"],
}

RELATIONS = [
    ("Type 1 Diabetes", "is_a", "Diabetes Mellitus"),
    ("Type 2 Diabetes", "is_a", "Diabetes Mellitus"),
    ("Gestational Diabetes", "is_a", "Diabetes Mellitus"),
    ("Diabetes Mellitus", "characterized_by", "Hyperglycemia"),
    ("Pancreas", "produces", "Insulin"),
    ("Insulin", "lowers", "Blood Glucose"),
    ("Type 1 Diabetes", "destroys_beta_cells_of", "Pancreas"),
    ("Type 1 Diabetes", "treated_with", "Insulin"),
    ("Type 2 Diabetes", "caused_by", "Insulin Resistance"),
    ("Insulin Resistance", "raises", "Blood Glucose"),
    ("Obesity", "increases", "Insulin Resistance"),
    ("Obesity", "risk_factor_for", "Type 2 Diabetes"),
    ("Exercise", "reduces", "Insulin Resistance"),
    ("Exercise", "lowers", "Blood Glucose"),
    ("Carbohydrates", "raise", "Blood Glucose"),
    ("Metformin", "treats", "Type 2 Diabetes"),
    ("Metformin", "reduces", "Insulin Resistance"),
    ("HbA1c", "measures_3_month_average_of", "Blood Glucose"),
    ("Insulin", "overdose_causes", "Hypoglycemia"),
    ("Hyperglycemia", "leads_to", "Retinopathy"),
    ("Hyperglycemia", "leads_to", "Nephropathy"),
    ("Hyperglycemia", "leads_to", "Neuropathy"),
    ("Hyperglycemia", "leads_to", "Cardiovascular Disease"),
    ("Diabetic Ketoacidosis", "acute_complication_of", "Type 1 Diabetes"),
]

print(f"Concepts: {len(CONCEPTS)} | Relationships: {len(RELATIONS)}\n")
for s, r, o in RELATIONS:
    print(f"  {s:24s} --{r}--> {o}")

# %%
# Visualise the knowledge graph
try:
    import networkx as nx
    G = nx.DiGraph()
    for s, r, o in RELATIONS:
        G.add_edge(s, o, label=r)
    groups = {  # colour nodes by concept type
        "#ffd6d6": ["Diabetes Mellitus", "Type 1 Diabetes", "Type 2 Diabetes", "Gestational Diabetes"],
        "#d6ecff": ["Pancreas", "Insulin", "Blood Glucose", "Insulin Resistance", "HbA1c"],
        "#d9f7d9": ["Obesity", "Carbohydrates", "Exercise", "Metformin"],
        "#fff1c9": ["Hypoglycemia", "Hyperglycemia", "Diabetic Ketoacidosis", "Neuropathy",
                    "Retinopathy", "Nephropathy", "Cardiovascular Disease"],
    }
    node_color = {n: c for c, ns in groups.items() for n in ns}
    plt.figure(figsize=(16, 11))
    pos = {  # fixed, hand-placed layout for readability
        "Carbohydrates": (0, 8), "Exercise": (0, 5.5), "Obesity": (0, 3), "Metformin": (0, 0.5),
        "HbA1c": (1.5, 10.3), "Blood Glucose": (4, 7), "Insulin Resistance": (3.5, 4), "Type 2 Diabetes": (4, 1),
        "Hypoglycemia": (10, 9.5), "Insulin": (7, 6), "Pancreas": (12, 7), "Type 1 Diabetes": (9.5, 3.2),
        "Diabetic Ketoacidosis": (12.8, 3.2), "Gestational Diabetes": (11.5, 0.6), "Diabetes Mellitus": (7.5, 0),
        "Hyperglycemia": (7.5, -2.8), "Retinopathy": (3, -5), "Nephropathy": (6, -5),
        "Neuropathy": (9, -5), "Cardiovascular Disease": (12, -5),
    }
    nx.draw_networkx_nodes(G, pos, node_size=3800, node_color=[node_color[n] for n in G.nodes],
                           edgecolors="#555")
    nx.draw_networkx_labels(G, pos, labels={n: n.replace(" ", "\n", 1) for n in G.nodes},
                            font_size=8, font_weight="bold")
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=18, edge_color="#777", node_size=3800,
                           connectionstyle="arc3,rad=0.06")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"),
                                 font_size=7, font_color="#a33", label_pos=0.5)
    from matplotlib.patches import Patch
    plt.legend(handles=[Patch(color=c, label=l) for c, l in zip(groups, [
        "Disease types", "Physiology / measures", "Risk factors & interventions", "Complications"])],
        loc="lower left", fontsize=9)
    plt.title("Diabetes Domain Knowledge Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("knowledge_graph.png", dpi=120)
    plt.show()
except ImportError:
    print("networkx not installed — skipping graph drawing.")

# %% [markdown]
# ## §4 NLP Preprocessing (Requirement 4)
# 1. **Lowercase conversion**  2. **Tokenization** (NLTK `word_tokenize`)  3. **Stop-word removal** (NLTK English list,
# keeping negations such as *not/no* and time words such as *before/after* which matter in this domain)
# 4. **Lemmatization** (WordNet lemmatizer, verb form first then noun form).
#
# Small fallbacks are included so the code still runs if NLTK data cannot be downloaded.

# %%
_FALLBACK_STOPWORDS = set("""a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself of off on once only or
other our ours ourselves out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where which while who whom why
will with would you your yours yourself yourselves""".split())

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    STOPWORDS = set(_FALLBACK_STOPWORDS)

KEEP_WORDS = {"not", "no", "nor", "before", "after", "during", "high", "low", "very"}
STOPWORDS -= KEEP_WORDS

try:
    from nltk.stem import WordNetLemmatizer
    _lemmatizer = WordNetLemmatizer()
    _lemmatizer.lemmatize("tests")          # triggers LookupError if wordnet is missing
except LookupError:
    _lemmatizer = None


def to_lowercase(text):
    return text.lower()


def tokenize(text):
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)
    tokens = ["not" if t in ("n't", "nt") else t for t in tokens]
    return [t for t in tokens if re.search(r"[a-z0-9]", t)]          # drop pure punctuation


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]


def lemmatize_token(tok):
    if _lemmatizer is None:                                        # crude fallback
        for suf in ("ing", "ed", "es", "s"):
            if tok.endswith(suf) and len(tok) - len(suf) >= 3:
                return tok[: -len(suf)]
        return tok
    verb = _lemmatizer.lemmatize(tok, "v")
    return verb if verb != tok else _lemmatizer.lemmatize(tok, "n")


def lemmatize(tokens):
    return [lemmatize_token(t) for t in tokens]


def preprocess_tokens(text):
    return lemmatize(remove_stopwords(tokenize(to_lowercase(text))))


def preprocess(text):
    return " ".join(preprocess_tokens(text))


# Demonstrate every step on one query
demo = "Why am I always feeling THIRSTY and urinating so often at nights?"
step1 = to_lowercase(demo)
step2 = tokenize(step1)
step3 = remove_stopwords(step2)
step4 = lemmatize(step3)
print("Original        :", demo)
print("1. Lowercase    :", step1)
print("2. Tokens       :", step2)
print("3. No stopwords :", step3)
print("4. Lemmas       :", step4)

df["clean"] = df["query"].apply(preprocess)
df[["query", "clean", "intent"]].head(10)

# %% [markdown]
# ## §5 TF-IDF Feature Extraction + Classifier Training (Requirement 5)
# Features = **TF-IDF of word uni/bi-grams** + **TF-IDF of character 3–5-grams** (the character n-grams make the model
# robust to spelling variations and differently-worded queries).
# Three classifiers are trained and compared: **Multinomial Naive Bayes, Logistic Regression and Linear SVM**.

# %%
def make_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ])


MODELS = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(C=50, max_iter=3000, random_state=RANDOM_STATE),
    "Linear SVM": LinearSVC(C=1.0, random_state=RANDOM_STATE),
}

X, y = df["clean"], df["intent"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# Peek at the TF-IDF feature space
_feat = make_features().fit(X_train)
print("Word-level TF-IDF vocabulary size :", len(_feat.transformer_list[0][1].vocabulary_))
print("Char-level TF-IDF vocabulary size :", len(_feat.transformer_list[1][1].vocabulary_))

pipelines = {}
for name, clf in MODELS.items():
    pipe = Pipeline([("tfidf", make_features()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pipelines[name] = pipe
print("Trained:", ", ".join(pipelines))

# %% [markdown]
# ## §6 Model Evaluation (Requirement 6)
# Accuracy, macro precision, recall and F1 on the held-out test set, plus 5-fold stratified cross-validation
# (more reliable on a small corpus). The full classification report and confusion matrix are shown for the
# deployed model.

# %%
rows = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for name, pipe in pipelines.items():
    pred = pipe.predict(X_test)
    p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    cv_f1 = cross_val_score(Pipeline([("tfidf", make_features()), ("clf", MODELS[name])]),
                            X, y, cv=cv, scoring="f1_macro")
    rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred), "Precision": p,
                 "Recall": r, "F1-score": f, "5-fold CV F1 (mean)": cv_f1.mean(),
                 "5-fold CV F1 (std)": cv_f1.std()})
results = pd.DataFrame(rows).set_index("Model").round(3)
print(results.to_string())

# %%
# Logistic Regression is deployed: it is among the best performers and, unlike Linear SVM,
# outputs calibrated class probabilities that the low-confidence rule needs.
DEPLOYED_NAME = "Logistic Regression"
best = pipelines[DEPLOYED_NAME]
y_pred = best.predict(X_test)

print(f"Deployed model: {DEPLOYED_NAME}\n")
print(classification_report(y_test, y_pred, zero_division=0))

labels = list(CORPUS.keys())
cm = confusion_matrix(y_test, y_pred, labels=labels)
fig, ax = plt.subplots(figsize=(9, 7))
ConfusionMatrixDisplay(cm, display_labels=[INTENT_LABELS[l] for l in labels]).plot(
    ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
ax.set_title(f"Confusion Matrix — {DEPLOYED_NAME} (test set)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.show()

# %%
# Retrain the deployed model on the full corpus for use in the assistant
intent_model = Pipeline([("tfidf", make_features()), ("clf", MODELS[DEPLOYED_NAME])])
intent_model.fit(X, y)
WORD_VOCAB = set(intent_model.named_steps["tfidf"].transformer_list[0][1].vocabulary_)
WORD_VOCAB = {w for w in WORD_VOCAB if " " not in w}       # unigram vocabulary
print("Final model trained on all", len(X), "queries. Unigram vocabulary:", len(WORD_VOCAB))

# %% [markdown]
# ## §7 Domain Knowledge Base (Requirement 7)
# Each intent has an **overview** answer plus several **topic entries**. A topic is selected by keyword overlap
# between the (lemmatized) query and the topic's keywords — this is the knowledge-retrieval step.
# Each topic also records the taxonomy leaf it belongs to.

# %%
KNOWLEDGE_BASE = {
    "symptoms_diagnosis": {
        "overview": ("Common symptoms of diabetes include frequent urination (polyuria), excessive thirst "
                     "(polydipsia), increased hunger, unexplained weight loss, fatigue, blurred vision, slow-"
                     "healing sores and frequent infections. Type 2 diabetes often develops slowly with mild or "
                     "no symptoms, so regular screening is important."),
        "topics": [
            {"keywords": ["diagnose", "diagnosis", "test", "confirm", "blood test", "doctor"],
             "answer": ("Diabetes is diagnosed with blood tests:\n"
                        "• Fasting plasma glucose ≥ 126 mg/dL (prediabetes 100–125)\n"
                        "• HbA1c ≥ 6.5% (prediabetes 5.7–6.4%)\n"
                        "• 2-hour oral glucose tolerance test ≥ 200 mg/dL (prediabetes 140–199)\n"
                        "• Random glucose ≥ 200 mg/dL with classic symptoms\n"
                        "Usually an abnormal result is repeated on another day to confirm.")},
            {"keywords": ["child", "children", "kid", "type 1", "sudden"],
             "answer": ("Type 1 diabetes in children often appears quickly over a few weeks: extreme thirst, "
                        "frequent urination or new bed-wetting, weight loss despite eating more, tiredness, "
                        "irritability and fruity-smelling breath. Vomiting, heavy breathing or drowsiness can mean "
                        "diabetic ketoacidosis — seek emergency care.")},
            {"keywords": ["prediabetes", "borderline"],
             "answer": ("Prediabetes usually causes no symptoms — it is found by blood tests (fasting glucose "
                        "100–125 mg/dL or HbA1c 5.7–6.4%). Some people notice darkened skin patches on the neck or "
                        "armpits (acanthosis nigricans). Prediabetes can often be reversed with lifestyle change.")},
            {"keywords": ["wound", "heal", "sore", "infection", "cut"],
             "answer": ("High blood sugar damages blood vessels and weakens the immune response, so cuts and sores "
                        "heal slowly and infections recur. Slow healing together with thirst or frequent urination "
                        "is a reason to get a blood glucose test.")},
        ],
    },
    "types_causes": {
        "overview": ("Diabetes mellitus is a chronic condition in which blood glucose stays too high because the "
                     "body does not make enough insulin or cannot use it well. Main types: Type 1 (autoimmune, no "
                     "insulin production), Type 2 (insulin resistance — about 90% of cases), Gestational diabetes "
                     "(during pregnancy) and Prediabetes (above-normal glucose, not yet diabetes)."),
        "topics": [
            {"keywords": ["difference", "versus", "vs", "compare", "type 1 type 2", "how many type"],
             "answer": ("Type 1 vs Type 2:\n"
                        "• Type 1 — the immune system destroys insulin-producing beta cells of the pancreas; usually "
                        "starts in childhood/young adults; needs insulin for life.\n"
                        "• Type 2 — the body becomes resistant to insulin and the pancreas cannot keep up; linked to "
                        "excess weight, inactivity, age and family history; managed with lifestyle, oral medicines "
                        "and sometimes insulin.")},
            {"keywords": ["cause", "risk", "factor", "hereditary", "genetic", "family", "parent", "obesity",
                          "overweight", "develop", "get", "likely", "chance", "sugar"],
             "answer": ("Risk factors for Type 2 diabetes: overweight/obesity (especially belly fat), physical "
                        "inactivity, family history, age over 35–45, South-Asian ethnicity, high blood pressure, "
                        "abnormal cholesterol, history of gestational diabetes and PCOS. Eating sugar does not "
                        "directly cause diabetes, but a diet high in sugar and calories leads to weight gain, which "
                        "raises risk. Type 1 is caused by an autoimmune reaction plus genetic factors — not by diet.")},
            {"keywords": ["gestational", "pregnancy", "pregnant"],
             "answer": ("Gestational diabetes is high blood sugar that first appears during pregnancy (usually "
                        "24–28 weeks) because pregnancy hormones cause insulin resistance. It usually goes away after "
                        "delivery but raises the mother's future Type 2 risk, so screening after pregnancy is advised.")},
            {"keywords": ["insulin resistance", "resistance", "resistant"],
             "answer": ("Insulin resistance means muscle, fat and liver cells respond poorly to insulin, so glucose "
                        "cannot enter cells easily and blood sugar rises. The pancreas compensates by making more "
                        "insulin until it can no longer keep up — leading to prediabetes and Type 2 diabetes. Weight "
                        "loss and exercise improve insulin sensitivity.")},
            {"keywords": ["autoimmune", "immune"],
             "answer": ("Yes — Type 1 diabetes is an autoimmune disease: the immune system attacks and destroys the "
                        "insulin-producing beta cells in the pancreas. It is not caused by lifestyle.")},
            {"keywords": ["prediabetes", "borderline"],
             "answer": ("Prediabetes means blood glucose is higher than normal but below the diabetes threshold "
                        "(fasting 100–125 mg/dL, HbA1c 5.7–6.4%). Losing 5–7% of body weight and 150 minutes of "
                        "weekly activity can cut the risk of progressing to Type 2 diabetes by about half.")},
        ],
    },
    "diet_nutrition": {
        "overview": ("Use the Diabetes Plate Method: fill ½ the plate with non-starchy vegetables, ¼ with lean "
                     "protein (dal, eggs, fish, chicken, paneer, tofu) and ¼ with complex carbohydrates (whole "
                     "grains, millets, brown rice). Prefer high-fibre, low-glycemic-index foods; limit sugary drinks, "
                     "refined flour, fried foods and processed snacks; drink water and keep meal times regular."),
        "topics": [
            {"keywords": ["avoid", "not eat", "bad"],
             "answer": ("Foods to limit or avoid: sugary drinks and juices, sweets, white bread/maida, large "
                        "portions of white rice, fried snacks, processed meats, trans fats and excess salt. Choose "
                        "whole grains, vegetables, pulses, nuts and lean proteins instead.")},
            {"keywords": ["rice", "roti", "wheat", "grain"],
             "answer": ("Rice can be eaten in controlled portions (about ¼ of the plate). Prefer brown rice, "
                        "parboiled rice or millets, and combine with vegetables, dal or protein to slow the sugar "
                        "rise. Cooling and reheating cooked rice slightly increases resistant starch.")},
            {"keywords": ["fruit", "mango", "banana", "apple"],
             "answer": ("Whole fruits are healthy for diabetics because of their fibre and vitamins. Prefer low-GI "
                        "fruits (apple, guava, pear, berries, orange) and eat 1 small portion at a time. Limit very "
                        "sweet fruits (mango, grapes, chikoo) to small amounts, and avoid fruit juices.")},
            {"keywords": ["glycemic", "gi", "index"],
             "answer": ("The glycemic index (GI) ranks foods by how quickly they raise blood glucose: low (≤55, e.g. "
                        "oats, lentils, most vegetables), medium (56–69) and high (≥70, e.g. white bread, white rice, "
                        "glucose). Low-GI foods give a slower, steadier rise.")},
            {"keywords": ["count", "carbohydrate", "carb", "gram"],
             "answer": ("Carbohydrate counting: read food labels for total carbs, and aim for a consistent amount at "
                        "each meal (commonly 45–60 g per meal for adults — your dietitian will personalise this). "
                        "People on mealtime insulin match the dose to carbs using an insulin-to-carb ratio.")},
            {"keywords": ["sweet", "dessert", "sugar", "honey", "jaggery", "sweetener", "artificial"],
             "answer": ("Honey and jaggery raise blood sugar just like table sugar — they are not safe substitutes. "
                        "Small portions of sweets can occasionally fit if counted within your meal's carbs. Approved "
                        "non-nutritive sweeteners (stevia, sucralose, aspartame) are considered safe in moderation.")},
            {"keywords": ["alcohol", "drink", "beer", "wine"],
             "answer": ("If you drink alcohol, do so in moderation (max 1 drink/day for women, 2 for men), never on "
                        "an empty stomach, and check your sugar — alcohol can cause delayed hypoglycemia, especially "
                        "with insulin or sulfonylureas.")},
            {"keywords": ["breakfast", "plan", "meal", "diet"],
             "answer": ("Sample diabetes-friendly day: Breakfast — vegetable oats/besan chilla with curd; Lunch — 2 "
                        "multigrain rotis, dal, salad, sabzi; Snack — handful of nuts or roasted chana; Dinner — "
                        "grilled fish/paneer with vegetables and small portion of brown rice. Keep portions "
                        "consistent and avoid skipping meals.")},
        ],
    },
    "medication_treatment": {
        "overview": ("Diabetes treatment depends on the type: Type 1 needs insulin; Type 2 starts with lifestyle "
                     "change and usually metformin, with other medicines (SGLT2 inhibitors, GLP-1 receptor agonists, "
                     "DPP-4 inhibitors, sulfonylureas) or insulin added as needed. Always take medicines exactly as "
                     "prescribed."),
        "topics": [
            {"keywords": ["metformin", "work", "side effect", "effect"],
             "answer": ("Metformin is usually the first-line medicine for Type 2 diabetes. It reduces glucose "
                        "production by the liver and improves insulin sensitivity, and does not usually cause "
                        "hypoglycemia on its own. Common side effects: nausea, diarrhoea, stomach upset (reduced by "
                        "taking it with meals or using extended-release); long-term use can lower vitamin B12.")},
            {"keywords": ["type", "insulin type", "rapid", "long acting", "basal"],
             "answer": ("Types of insulin: Rapid-acting (lispro, aspart — taken just before meals), Short-acting "
                        "(regular insulin), Intermediate (NPH), Long-acting/basal (glargine, detemir, degludec — "
                        "once or twice daily) and premixed combinations.")},
            {"keywords": ["inject", "injection", "needle", "pen", "site", "shot", "body", "give", "when"],
             "answer": ("Insulin is injected under the skin of the abdomen, thighs, upper arms or buttocks. Rotate "
                        "sites to avoid lumps, use a new needle, and inject at the times your doctor prescribed — "
                        "mealtime insulin usually 0–15 minutes before eating, basal insulin at the same time each day.")},
            {"keywords": ["store", "storage", "fridge", "keep"],
             "answer": ("Store unopened insulin in the refrigerator (2–8 °C), never in the freezer. The pen/vial in "
                        "use can be kept at room temperature (below ~30 °C) for about 28 days. Keep it away from "
                        "direct sunlight and heat; discard if frozen, cloudy (when it should be clear) or expired.")},
            {"keywords": ["miss", "forgot", "skip", "dose"],
             "answer": ("If you miss a dose of a diabetes tablet, take it when you remember unless it is almost time "
                        "for the next dose — then skip it. Never double the dose. For missed insulin, follow the plan "
                        "given by your doctor and check your blood sugar more often.")},
            {"keywords": ["stop", "quit", "normal"],
             "answer": ("Do not stop diabetes medicines on your own, even if your readings are normal — the normal "
                        "readings are often because of the medicine. Discuss any dose reduction with your doctor.")},
            {"keywords": ["tablet", "pill", "oral", "drug"],
             "answer": ("Oral medicines for Type 2 diabetes include metformin, SGLT2 inhibitors (empagliflozin, "
                        "dapagliflozin), DPP-4 inhibitors (sitagliptin, vildagliptin), sulfonylureas (glimepiride, "
                        "gliclazide) and pioglitazone. GLP-1 receptor agonists (semaglutide) are mostly injectable.")},
            {"keywords": ["pump", "device"],
             "answer": ("An insulin pump is a small device that delivers rapid-acting insulin continuously under the "
                        "skin through a thin tube (basal rate) plus extra doses at meals (bolus). Often combined with "
                        "a CGM in Type 1 diabetes.")},
            {"keywords": ["weight gain", "weight", "gain"],
             "answer": ("Insulin can cause some weight gain because the body stores glucose that was previously "
                        "lost in urine. Healthy eating, regular activity and correct dosing minimise this.")},
        ],
    },
    "monitoring": {
        "overview": ("General glucose targets for most adults with diabetes (ADA): before meals 80–130 mg/dL, "
                     "1–2 hours after meals below 180 mg/dL, and HbA1c below 7%. For people without diabetes, "
                     "fasting glucose is 70–99 mg/dL. Your doctor may personalise these targets."),
        "topics": [
            {"keywords": ["hba1c", "a1c", "glycated", "target"],
             "answer": ("HbA1c measures the percentage of haemoglobin coated with glucose and reflects average blood "
                        "sugar over the last 2–3 months. Normal < 5.7%, prediabetes 5.7–6.4%, diabetes ≥ 6.5%. Most "
                        "adults with diabetes aim for < 7%. It is usually checked every 3–6 months.")},
            {"keywords": ["glucometer", "meter", "use", "finger", "strip"],
             "answer": ("Using a glucometer: wash and dry hands, insert a test strip, prick the side of a fingertip "
                        "with the lancet, touch the blood drop to the strip and read the result. Record the value with "
                        "the time and whether it was before/after a meal.")},
            {"keywords": ["often", "time", "when", "frequency", "best"],
             "answer": ("How often to test depends on treatment: people on multiple insulin injections usually test "
                        "before meals and at bedtime (3–4+ times/day); those on oral medicines may test a few times "
                        "a week, varying times (fasting and 2 hours after meals). Test more when ill or when "
                        "symptoms of low sugar occur.")},
            {"keywords": ["continuous", "cgm", "sensor", "monitor"],
             "answer": ("A continuous glucose monitor (CGM) is a small sensor worn on the arm or abdomen that measures "
                        "glucose in tissue fluid every few minutes, shows trends and alarms for highs and lows.")},
            {"keywords": ["morning", "fasting high", "dawn"],
             "answer": ("High morning (fasting) sugar can be due to the dawn phenomenon (early-morning hormones), a "
                        "late heavy dinner, insufficient night-time medication, or rebound after a night-time low. "
                        "Checking at 2–3 AM for a few nights helps your doctor find the cause.")},
            {"keywords": ["record", "track", "log", "diary"],
             "answer": ("Keep a logbook or app with date, time, reading, meal/exercise and medicine taken. Bring it "
                        "to every appointment — patterns help your doctor adjust treatment.")},
            {"keywords": ["after", "meal", "post", "eat"],
             "answer": ("After meals (1–2 hours) blood sugar should ideally be below 180 mg/dL for people with "
                        "diabetes and below 140 mg/dL for people without diabetes.")},
        ],
    },
    "exercise_lifestyle": {
        "overview": ("Adults with diabetes should aim for at least 150 minutes per week of moderate aerobic activity "
                     "(e.g. brisk walking 30 min × 5 days), not skipping more than 2 days in a row, plus resistance "
                     "training 2–3 times a week. Reduce long periods of sitting. Exercise improves insulin "
                     "sensitivity and lowers blood glucose."),
        "topics": [
            {"keywords": ["safe", "high", "before", "work out", "precaution"],
             "answer": ("Exercise safety: check sugar before exercise if on insulin or sulfonylureas; if below "
                        "100 mg/dL eat a small carb snack first. If above 250 mg/dL with ketones, avoid vigorous "
                        "exercise. Carry glucose tablets, stay hydrated and wear proper footwear.")},
            {"keywords": ["yoga", "meditation"],
             "answer": ("Yoga can help diabetes by improving flexibility, reducing stress hormones and modestly "
                        "lowering glucose and HbA1c. It works best combined with aerobic activity such as walking.")},
            {"keywords": ["stress", "anxiety", "tension"],
             "answer": ("Stress releases cortisol and adrenaline, which raise blood sugar. Relaxation techniques, "
                        "adequate sleep, physical activity and talking to someone help keep stress and glucose in check.")},
            {"keywords": ["sleep"],
             "answer": ("Poor or short sleep increases insulin resistance and appetite. Aim for 7–8 hours of good "
                        "quality sleep; get screened for sleep apnoea if you snore heavily.")},
            {"keywords": ["weight", "lose", "losing", "belly", "fat"],
             "answer": ("Losing 5–10% of body weight improves blood sugar, blood pressure and cholesterol, and can "
                        "sometimes put Type 2 diabetes into remission. Combine portion control with regular activity.")},
            {"keywords": ["smoke", "smoking", "tobacco", "quit"],
             "answer": ("Yes — quit smoking. Smoking increases insulin resistance and greatly raises the risk of heart "
                        "attack, stroke, kidney disease and foot amputation in people with diabetes.")},
            {"keywords": ["strength", "gym", "weight training", "resistance"],
             "answer": ("Strength (resistance) training 2–3 times a week builds muscle, which uses glucose and improves "
                        "insulin sensitivity. Start with light weights and good technique.")},
            {"keywords": ["walk", "minute", "long", "week", "activity"],
             "answer": ("Aim for at least 150 minutes of moderate activity per week — e.g. a 30-minute brisk walk on 5 "
                        "days. A 10–15 minute walk after each meal is especially effective at lowering post-meal sugar.")},
            {"keywords": ["reverse", "prediabetes", "prevent"],
             "answer": ("Lifestyle change can reverse prediabetes: lose 5–7% of body weight, do 150 min/week of "
                        "activity, eat more fibre and fewer refined carbs and sugary drinks. This reduces progression "
                        "to Type 2 diabetes by about 58%.")},
        ],
    },
    "complications": {
        "overview": ("Long-term high blood sugar damages blood vessels and nerves, causing chronic complications: "
                     "eye disease (retinopathy), kidney disease (nephropathy), nerve damage (neuropathy), foot ulcers, "
                     "heart attack and stroke. Acute complications include hypoglycemia and diabetic ketoacidosis. "
                     "Good control of sugar, blood pressure and cholesterol plus yearly check-ups prevent most of them."),
        "topics": [
            {"keywords": ["hypoglycemia", "low", "low sugar", "hypo"], "leaf": "4.1 Acute Complications",
             "answer": ("Hypoglycemia is blood glucose below 70 mg/dL. Symptoms: shaking, sweating, fast heartbeat, "
                        "hunger, dizziness, confusion. Treat with the 15-15 rule: take 15 g of fast-acting sugar "
                        "(½ cup juice, 3–4 glucose tablets, 1 tbsp sugar), recheck after 15 minutes and repeat if still "
                        "below 70, then eat a snack. If the person is unconscious, do not give food by mouth — call "
                        "emergency services.")},
            {"keywords": ["ketoacidosis", "dka", "ketone"], "leaf": "4.1 Acute Complications",
             "answer": ("Diabetic ketoacidosis (DKA) is a life-threatening emergency, mostly in Type 1 diabetes, caused "
                        "by severe lack of insulin: the body burns fat and produces acidic ketones. Signs: very high "
                        "sugar, nausea/vomiting, abdominal pain, deep fast breathing, fruity breath, confusion. It needs "
                        "immediate hospital treatment.")},
            {"keywords": ["eye", "vision", "retinopathy", "blind"], "leaf": "4.2 Chronic Complications",
             "answer": ("Diabetic retinopathy is damage to the blood vessels of the retina. It may have no early "
                        "symptoms but can cause blindness. Get a dilated eye examination every year; good sugar and "
                        "blood-pressure control and early laser/injection treatment protect vision.")},
            {"keywords": ["kidney", "nephropathy", "renal"], "leaf": "4.2 Chronic Complications",
             "answer": ("Diabetes is the leading cause of kidney disease (nephropathy). Get urine albumin and blood "
                        "creatinine (eGFR) tests every year; control sugar and blood pressure and avoid unnecessary "
                        "painkillers (NSAIDs).")},
            {"keywords": ["nerve", "neuropathy", "numbness", "tingling", "pain", "burn"], "leaf": "4.2 Chronic Complications",
             "answer": ("Diabetic neuropathy is nerve damage caused by long-term high sugar, usually starting in the "
                        "feet: numbness, tingling, burning pain or loss of sensation. Good glucose control slows it; "
                        "doctors can prescribe medicines for nerve pain. Loss of feeling makes daily foot checks vital.")},
            {"keywords": ["foot", "feet", "ulcer", "toe"], "leaf": "4.2 Chronic Complications",
             "answer": ("Diabetic foot care: inspect your feet daily for cuts, blisters or redness; wash and dry well "
                        "(especially between toes); moisturise but not between toes; never walk barefoot; wear "
                        "well-fitting shoes; cut nails straight; see a doctor promptly for any wound.")},
            {"keywords": ["heart", "cardiovascular", "stroke", "blood pressure", "cholesterol"], "leaf": "4.2 Chronic Complications",
             "answer": ("People with diabetes have 2–4× higher risk of heart attack and stroke. Protect your heart by "
                        "controlling blood pressure (< 130/80), LDL cholesterol, blood sugar, not smoking and staying "
                        "active; your doctor may prescribe a statin.")},
            {"keywords": ["infection", "immune"], "leaf": "4.2 Chronic Complications",
             "answer": ("High glucose weakens white blood cells and reduces blood flow, so people with diabetes get "
                        "more frequent skin, urinary and fungal infections. Better sugar control reduces this risk.")},
        ],
    },
}

# Pre-compute lemmatized keyword sets so retrieval matches the preprocessed query
for intent, entry in KNOWLEDGE_BASE.items():
    for topic in entry["topics"]:
        topic["kw_lemmas"] = [set(preprocess_tokens(k)) for k in topic["keywords"]]

print("Knowledge base entries per intent:")
for intent, entry in KNOWLEDGE_BASE.items():
    print(f"  {INTENT_LABELS[intent]:26s}: 1 overview + {len(entry['topics'])} topics")


def retrieve_knowledge(intent, query_lemmas):
    """Return (answer, taxonomy leaf) for the best-matching topic of the intent."""
    entry = KNOWLEDGE_BASE[intent]
    q = set(query_lemmas)
    best_topic, best_score = None, 0
    for topic in entry["topics"]:
        # a keyword phrase matches if all its lemmas are in the query
        score = sum(len(kw) for kw in topic["kw_lemmas"] if kw and kw <= q)
        if score > best_score:
            best_topic, best_score = topic, score
    if best_topic is None:
        return entry["overview"], None
    return best_topic["answer"], best_topic.get("leaf")

# %% [markdown]
# ## §8 Rule-Based Reasoning Engine (Requirement 8)
# Eight rules are applied in priority order. Rules **R1–R5** can *override* the classifier's answer; **R6–R8** *augment* it.
#
# | Rule | Name | Condition | Action |
# |---|---|---|---|
# | R1 | **Emergency escalation** | Emergency keywords (unconscious, seizure, chest pain, fruity breath + vomiting …) | Stop, give first-aid steps and tell user to call 112 / 108 |
# | R2 | Critical glucose reading | A glucose value < 54 or ≥ 400 mg/dL (or ≥ 300 with vomiting/confusion) | Emergency / urgent advice |
# | R3 | Greeting / empty input | Only greetings, thanks or empty text | Friendly help message |
# | R4 | **Unsupported query** | Out-of-scope health topic with no diabetes term, or no known domain vocabulary | Polite refusal + suggest proper specialist |
# | R5 | **Low confidence** | Top intent probability < 0.40 **or** margin between top-2 intents < 0.10 | Ask for clarification, offer top-2 interpretations |
# | R6 | Glucose reading interpretation | A non-critical glucose value is present | Append interpretation (low / in range / high) |
# | R7 | Medication-change safety | User wants to stop / skip / change dose | Append "consult your doctor" caution |
# | R8 | Pregnancy context | Mentions pregnancy | Append gestational-diabetes care note |
# | — | Disclaimer | Always | Append medical disclaimer |

# %%
CONFIDENCE_THRESHOLD = 0.40
MARGIN_THRESHOLD = 0.10
EMERGENCY_NUMBER = "112 (or 108 for ambulance in India)"

EMERGENCY_PATTERNS = [
    r"\bunconscious", r"\bpassed out", r"\bfaint(ed|ing)?\b", r"\bcollaps(e|ed|ing)\b",
    r"\bnot (responding|waking|breathing)", r"\bunresponsive", r"\bseizure", r"\bconvuls", r"\bfits?\b",
    r"\bchest pain", r"\b(can'?t|cannot|difficulty|trouble) breath", r"\bshortness of breath",
    r"\bslurred speech", r"\bstroke\b(?!.*risk)", r"\bfruity breath", r"\bvomit\w*.*(sugar|ketone|diabet)",
    r"\b(sugar|ketone|diabet)\w*.*vomit", r"\bsevere(ly)? (low|hypo)", r"\btook too much insulin",
    r"\binsulin overdose", r"\boverdose", r"\bsuicid", r"\bkill myself", r"\bemergency\b",
]

GREETING_WORDS = {"hi", "hello", "hey", "thanks", "thank", "bye", "good", "morning", "evening",
                  "afternoon", "ok", "okay", "goodbye", "namaste"}

DIABETES_ANCHORS = {"diabetes", "diabetic", "sugar", "glucose", "insulin", "metformin", "hba1c", "a1c",
                    "glycemic", "hypoglycemia", "hyperglycemia", "ketoacidosis", "dka", "pancreas",
                    "carb", "carbohydrate", "glucometer", "cgm", "prediabetes", "gestational",
                    "neuropathy", "retinopathy", "nephropathy", "thirsty", "urination", "urinate"}

OUT_OF_SCOPE_TERMS = {"migraine", "headache", "cancer", "tumor", "tumour", "covid", "coronavirus", "fracture",
                      "broken", "asthma", "malaria", "dengue", "tuberculosis", "tb", "flu", "influenza",
                      "cold", "cough", "fever", "acne", "rash", "hair", "tooth", "teeth", "dental", "eczema",
                      "allergy", "arthritis", "thyroid", "hiv", "hepatitis", "vaccine", "vaccination",
                      "sprain", "piles", "constipation", "diarrhea", "diarrhoea", "appendicitis", "ear",
                      "depression", "pimple", "typhoid", "jaundice", "measles", "chickenpox", "back"}
OUT_OF_SCOPE_PHRASES = ["kidney stone", "back pain", "sore throat", "blood pressure medicine"]

_MEDS = r"(dose|insulin|medicine|medication|tablet|pill|drug|metformin|units)"
MED_CHANGE_PATTERNS = [r"\b(stop|quit|skip|discontinue)\w*\b.*\b" + _MEDS, r"\bskip",
                       r"\b(increase|decrease|reduce|double|change|adjust)\b.*\b" + _MEDS,
                       r"\bhow many units", r"\bcome off"]


def extract_glucose_mgdl(text):
    """Find a blood-glucose reading in the text; returns value in mg/dL or None."""
    t = text.lower()
    if not re.search(r"sugar|glucose|reading|mg|mmol|level|bgl", t):
        return None
    if re.search(r"hba1c|a1c", t):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*mmol", t)
    if m:
        return round(float(m.group(1)) * 18)
    for m in re.finditer(r"(?<![\d.])(\d{2,3})(?![\d.%])(?!\s*(?:year|yr|kg|min|hour|day|week|month|%))", t):
        val = int(m.group(1))
        if 20 <= val <= 800:
            return val
    return None


class RuleEngine:
    def __init__(self):
        self.rules = [
            ("R1 Emergency escalation", self.r1_emergency),
            ("R2 Critical glucose reading", self.r2_critical_reading),
            ("R3 Greeting / empty input", self.r3_greeting),
            ("R4 Unsupported query", self.r4_unsupported),
            ("R5 Low confidence", self.r5_low_confidence),
        ]
        self.augmenters = [
            ("R6 Glucose reading interpretation", self.r6_reading_note),
            ("R7 Medication-change safety", self.r7_med_safety),
            ("R8 Pregnancy context", self.r8_pregnancy),
        ]

    # ---------- overriding rules ----------
    def r1_emergency(self, ctx):
        t = ctx["text"].lower()
        if any(re.search(p, t) for p in EMERGENCY_PATTERNS):
            return ("🚨 POSSIBLE MEDICAL EMERGENCY\n"
                    f"• Call emergency services now: {EMERGENCY_NUMBER}.\n"
                    "• If the person is conscious and may have LOW sugar, give 15–20 g of fast sugar "
                    "(glucose tablets, juice, sugar in water). Do NOT give anything by mouth if they are unconscious or "
                    "having a seizure — lay them on their side.\n"
                    "• If HIGH sugar with vomiting, deep breathing or fruity breath (possible DKA), go to the nearest "
                    "hospital immediately.\n"
                    "• Stay with the person and do not drive yourself if you feel unwell.")
        return None

    def r2_critical_reading(self, ctx):
        v = ctx["glucose"]
        if v is None:
            return None
        t = ctx["text"].lower()
        severe_symptoms = re.search(r"vomit|confus|drows|dizz|breath|shak|sweat", t)
        if v < 54:
            return (f"🚨 URGENT: A reading of {v} mg/dL is a seriously LOW blood sugar (level 2 hypoglycemia).\n"
                    "• Take 15–20 g of fast-acting sugar now (½ cup juice, 3–4 glucose tablets, 1 tbsp sugar).\n"
                    "• Recheck in 15 minutes and repeat if still below 70 mg/dL, then eat a snack.\n"
                    f"• If the person becomes confused, drowsy or unconscious, call {EMERGENCY_NUMBER} immediately.")
        if v >= 400 or (v >= 300 and severe_symptoms):
            return (f"🚨 URGENT: A reading of {v} mg/dL is dangerously HIGH"
                    + (" and the symptoms you describe may indicate DKA/HHS" if severe_symptoms else "") + ".\n"
                    "• Check urine/blood ketones if you can and drink water.\n"
                    f"• Contact your doctor immediately or go to the emergency department / call {EMERGENCY_NUMBER}, "
                    "especially with vomiting, abdominal pain, deep breathing or confusion.")
        return None

    def r3_greeting(self, ctx):
        toks = re.findall(r"[a-z]+", ctx["text"].lower())
        if not toks or set(toks) <= GREETING_WORDS:
            return ("👋 Hello! I am the Diabetes Care Information Assistant. Ask me about diabetes symptoms, types "
                    "and causes, diet, medicines and insulin, blood-sugar monitoring, exercise or complications.")
        return None

    def r4_unsupported(self, ctx):
        lemmas, t = set(ctx["lemmas"]), ctx["text"].lower()
        has_anchor = bool(lemmas & DIABETES_ANCHORS) or bool(re.search(r"diabet|sugar|glucose|insulin", t))
        oos = (lemmas & OUT_OF_SCOPE_TERMS) or any(p in t for p in OUT_OF_SCOPE_PHRASES)
        known = [w for w in lemmas if w in WORD_VOCAB]
        if (oos and not has_anchor) or (not known and not has_anchor):
            topic = ", ".join(sorted(oos)) if oos else "this topic"
            return (f"❌ Sorry, '{topic}' is outside my scope. I am a Diabetes Care Information Assistant and can only "
                    "answer diabetes-related questions (symptoms, types & causes, diet, medication, monitoring, "
                    "exercise and complications). Please consult a general physician or the appropriate specialist "
                    "for other health concerns.")
        return None

    def r5_low_confidence(self, ctx):
        top, second = ctx["ranked"][0], ctx["ranked"][1]
        if top[1] < CONFIDENCE_THRESHOLD or (top[1] - second[1]) < MARGIN_THRESHOLD:
            return (f"🤔 I'm not confident I understood your question (confidence {top[1]:.0%}). Did you mean:\n"
                    f"  1) {INTENT_LABELS[top[0]]} ({top[1]:.0%}), or\n"
                    f"  2) {INTENT_LABELS[second[0]]} ({second[1]:.0%})?\n"
                    "Please rephrase with more detail — e.g. 'What foods should I eat?' or "
                    "'What is a normal sugar level?'")
        return None

    # ---------- augmenting rules ----------
    def r6_reading_note(self, ctx):
        v = ctx["glucose"]
        if v is None:
            return None
        if v < 70:
            return (f"📊 Your reading of {v} mg/dL is LOW (hypoglycemia < 70). Follow the 15-15 rule: 15 g fast "
                    "sugar, recheck in 15 min.")
        if v <= 130:
            return f"📊 Your reading of {v} mg/dL is within the usual target range (80–130 fasting/pre-meal)."
        if v < 180:
            return (f"📊 Your reading of {v} mg/dL is above the fasting target (80–130) but within the after-meal "
                    "target (< 180 at 1–2 h).")
        if v < 300:
            return (f"📊 Your reading of {v} mg/dL is HIGH (hyperglycemia). Drink water, take medicines as prescribed, "
                    "and inform your doctor if readings stay above 180–250.")
        return (f"📊 Your reading of {v} mg/dL is VERY HIGH. Check for ketones and contact your doctor today.")

    def r7_med_safety(self, ctx):
        t = ctx["text"].lower()
        if any(re.search(p, t) for p in MED_CHANGE_PATTERNS):
            return ("💊 Safety note: never stop, skip or change the dose of diabetes medicines or insulin on your "
                    "own — discuss any change with your doctor.")
        return None

    def r8_pregnancy(self, ctx):
        if re.search(r"pregnan", ctx["text"].lower()):
            return ("🤰 Pregnancy note: diabetes in pregnancy needs tighter targets and close follow-up with your "
                    "obstetrician/diabetologist; many oral medicines are replaced by insulin during pregnancy.")
        return None

    def apply_override_rules(self, ctx):
        for name, rule in self.rules:
            out = rule(ctx)
            if out:
                return name, out
        return None, None

    def apply_augmenters(self, ctx):
        notes, fired = [], []
        for name, rule in self.augmenters:
            out = rule(ctx)
            if out:
                notes.append(out)
                fired.append(name)
        return fired, notes


rule_engine = RuleEngine()
print("Override rules :", [n for n, _ in rule_engine.rules])
print("Augment rules  :", [n for n, _ in rule_engine.augmenters])

# %% [markdown]
# ## §9 Integrated Cognitive Assistant (Requirement 9)
# `DiabetesCareAssistant.respond()` combines every component:
# 1. **NLP** — preprocess the query (§4)
# 2. **Intent classification** — TF-IDF + Logistic Regression probabilities (§5)
# 3. **Reasoning** — rule engine (§8) and knowledge-graph inference (§3)
# 4. **Knowledge retrieval** — best topic from the knowledge base (§7) + taxonomy path (§2)
# 5. **Response generation**

# %%
def find_concepts(text):
    t = text.lower()
    found = []
    for concept, aliases in CONCEPTS.items():
        if any(re.search(r"\b" + re.escape(a) + r"s?\b", t) for a in aliases):
            found.append(concept)
    if "Insulin Resistance" in found and not re.search(r"\binsulin\b(?! resistance)", t):
        found = [c for c in found if c != "Insulin"]
    if "Hypoglycemia" in found or "Hyperglycemia" in found:
        found = [c for c in found if c != "Blood Glucose"]
    return found


_ADJ = {}
for s, r, o in RELATIONS:
    _ADJ.setdefault(s, []).append((r, o))


def kg_path(src, dst, max_len=4):
    """Breadth-first search for a chain of relations from src to dst in the knowledge graph."""
    queue, seen = deque([(src, [])]), {src}
    while queue:
        node, path = queue.popleft()
        if node == dst:
            return path
        if len(path) >= max_len:
            continue
        for rel, nxt in _ADJ.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, path + [(node, rel, nxt)]))
    return None


def kg_reasoning(concepts):
    """Explain how the concepts mentioned in the query are connected."""
    chains = []
    for i, a in enumerate(concepts):
        for b in concepts[i + 1:]:
            path = kg_path(a, b) or kg_path(b, a)
            if path:
                chain = path[0][0] + "".join(f" --{r}--> {o}" for _, r, o in path)
                if chain not in chains:
                    chains.append(chain)
    specific = [c for c in concepts if c != "Diabetes Mellitus"]
    if not chains and specific:                            # no chain: show direct facts of the concept
        c = specific[0]
        chains = [f"{s} --{r}--> {o}" for s, r, o in RELATIONS if c in (s, o)][:3]
    return chains[:3]


class DiabetesCareAssistant:
    DISCLAIMER = ("ℹ️ This is general health information, not a diagnosis. Please consult your doctor for "
                  "personal medical advice.")

    def __init__(self, model, rules):
        self.model = model
        self.rules = rules
        self.history = []

    def classify(self, clean_text):
        probs = self.model.predict_proba([clean_text])[0]
        ranked = sorted(zip(self.model.classes_, probs), key=lambda x: -x[1])
        return [(c, float(p)) for c, p in ranked]

    def respond(self, query):
        lemmas = preprocess_tokens(query)
        clean = " ".join(lemmas)
        ranked = self.classify(clean) if clean else [("symptoms_diagnosis", 0.0), ("types_causes", 0.0)]
        ctx = {"text": query, "lemmas": lemmas, "ranked": ranked, "glucose": extract_glucose_mgdl(query)}

        result = {"query": query, "preprocessed": clean, "intent": ranked[0][0],
                  "confidence": ranked[0][1], "rules_fired": [], "taxonomy": "", "kg": [], "type": "answer"}

        rule_name, override = self.rules.apply_override_rules(ctx)
        if override:
            result["rules_fired"].append(rule_name)
            result["type"] = rule_name.split(" ", 1)[1]
            result["answer"] = override
            if rule_name.startswith(("R1", "R2")):
                result["taxonomy"] = "Diabetes Care > 5. Safety & Escalation > 5.1 Medical Emergencies"
            elif rule_name.startswith("R4"):
                result["taxonomy"] = "Diabetes Care > 5. Safety & Escalation > 5.2 Out-of-scope Queries"
            if not rule_name.startswith(("R3", "R4")):
                result["answer"] += "\n\n" + self.DISCLAIMER
        else:
            intent = ranked[0][0]
            answer, leaf = retrieve_knowledge(intent, lemmas)
            path = list(INTENT_TO_TAXONOMY[intent])
            if intent == "complications":
                path.append(leaf or "4.2 Chronic Complications")
            result["taxonomy"] = " > ".join(path)
            result["kg"] = kg_reasoning(find_concepts(query))
            fired, notes = self.rules.apply_augmenters(ctx)
            result["rules_fired"] += fired
            parts = [answer]
            if notes:
                parts.append("\n".join(notes))
            if result["kg"]:
                parts.append("🔗 Knowledge-graph reasoning:\n" + "\n".join("  " + c for c in result["kg"]))
            parts.append(self.DISCLAIMER)
            result["answer"] = "\n\n".join(parts)

        self.history.append(result)
        return result

    @staticmethod
    def format(result):
        lines = [f"🧑 You: {result['query']}"]
        if result["type"] == "answer":
            lines.append(f"🎯 Intent: {INTENT_LABELS[result['intent']]}  |  confidence: {result['confidence']:.0%}")
        else:
            lines.append(f"⚙️ Handled by rule: {result['rules_fired'][0]}"
                         f"  (classifier guess: {INTENT_LABELS[result['intent']]}, {result['confidence']:.0%})")
        if result["taxonomy"]:
            lines.append(f"🗂️ Taxonomy: {result['taxonomy']}")
        if result["type"] == "answer" and result["rules_fired"]:
            lines.append(f"📐 Rules applied: {', '.join(result['rules_fired'])}")
        lines.append("🤖 Assistant:\n" + result["answer"])
        return "\n".join(lines)

    def ask(self, query):
        print(self.format(self.respond(query)))
        print("-" * 100)


assistant = DiabetesCareAssistant(intent_model, rule_engine)
assistant.ask("What are the symptoms of diabetes?")

# %% [markdown]
# ## §10 Interactive Interface in Google Colab (Requirement 10)
# Run the cell below, type a question in the text box and press **Enter** or click **Ask**. You can ask as many
# questions as you like — the conversation history stays on screen. **Clear** resets the chat.
#
# *(If widgets are unavailable, the next cell provides a text-based `input()` loop — type `exit` to stop.)*

# %%
import html as _html


def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None and "IPKernelApp" in get_ipython().config
    except Exception:
        return False


def _to_html(result):
    color = {"answer": "#e8f5e9", "Emergency escalation": "#ffebee", "Critical glucose reading": "#ffebee",
             "Unsupported query": "#fff3e0", "Low confidence": "#fffde7"}.get(result["type"], "#e3f2fd")
    body = _html.escape(DiabetesCareAssistant.format(result).split("🤖 Assistant:\n", 1)[0]).replace("\n", "<br>")
    ans = _html.escape(result["answer"]).replace("\n", "<br>")
    return (f"<div style='border:1px solid #ccc;border-radius:10px;padding:10px;margin:8px 0;"
            f"background:{color};color:#111;font-family:sans-serif;font-size:14px'>{body}"
            f"<b>🤖 Assistant:</b><br>{ans}</div>")


def launch_chat_ui():
    import ipywidgets as widgets
    from IPython.display import display, HTML
    try:
        from google.colab import output as _colab_output
        _colab_output.enable_custom_widget_manager()
    except Exception:
        pass

    header = widgets.HTML("<h3 style='margin:4px 0'>🩺 Diabetes Care Information Assistant</h3>"
                          "<i>Ask any diabetes question. Type below and press Enter or click Ask.</i>")
    box = widgets.Text(placeholder="e.g. What should my fasting sugar level be?",
                       layout=widgets.Layout(width="70%"))
    ask_btn = widgets.Button(description="Ask", button_style="primary", icon="paper-plane")
    clear_btn = widgets.Button(description="Clear", button_style="warning")
    out = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", padding="6px",
                                               max_height="600px", overflow="auto"))

    def on_ask(_=None):
        q = box.value.strip()
        if not q:
            return
        box.value = ""
        with out:
            display(HTML(_to_html(assistant.respond(q))))

    def on_clear(_):
        out.clear_output()
        assistant.history.clear()

    ask_btn.on_click(on_ask)
    clear_btn.on_click(on_clear)
    try:
        box.on_submit(on_ask)                  # Enter key
    except Exception:
        pass
    display(widgets.VBox([header, widgets.HBox([box, ask_btn, clear_btn]), out]))


if _in_notebook():
    launch_chat_ui()

# %%
def chat_loop():
    """Text-based fallback interface: accepts multiple queries until the user types 'exit'."""
    print("🩺 Diabetes Care Information Assistant — type 'exit' to quit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "bye", "q"}:
            print("Assistant: Take care! Goodbye. 👋")
            break
        assistant.ask(q)

# Uncomment the next line to use the text-based interface instead of the widget:
# chat_loop()

# %% [markdown]
# ## §11 Testing the Assistant
# The assistant is tested with:
# 1. **A query for each defined intent** (7 intents)
# 2. **Differently-worded queries** (paraphrases not present in the corpus)
# 3. **An unsupported healthcare query**
# 4. **A low-confidence query**
# 5. **An emergency-related query**

# %%
TEST_CASES = [
    # 1. One query per intent
    ("1. Intent: Symptoms & Diagnosis", "What are the warning signs that someone has diabetes?"),
    ("1. Intent: Types & Causes", "Am I more likely to get diabetes if I am overweight?"),
    ("1. Intent: Diet & Nutrition", "Is it okay for a diabetic to eat mango?"),
    ("1. Intent: Medication & Treatment", "What are the side effects of metformin tablets?"),
    ("1. Intent: Blood Glucose Monitoring", "What is the target HbA1c for a diabetic?"),
    ("1. Intent: Exercise & Lifestyle", "Can yoga and walking help me manage my diabetes?"),
    ("1. Intent: Complications", "How can diabetes damage my kidneys?"),
    # 2. Differently-worded queries
    ("2. Differently worded", "My mouth is always dry and I keep running to the toilet at night — should I worry?"),
    ("2. Differently worded", "I checked my glucose after lunch and it showed 165 — is that fine?"),
    ("2. Differently worded", "I'm scared of needles — how do I give myself insulin properly?"),
    # 3. Unsupported healthcare query
    ("3. Unsupported query", "How do I treat a migraine headache?"),
    ("3. Unsupported query", "What is the best vaccine for covid?"),
    # 4. Low-confidence query
    ("4. Low-confidence query", "Is this normal for a diabetic?"),
    ("4. Low-confidence query", "I have a question about diabetes"),
    # 5. Emergency-related query
    ("5. Emergency query", "My diabetic father has fainted and is not responding!"),
    ("5. Emergency query", "My sugar is 45 and I am shaking and sweating"),
]

summary = []
for label, q in TEST_CASES:
    print(f"\n#### TEST — {label}")
    r = assistant.respond(q)
    print(DiabetesCareAssistant.format(r))
    print("-" * 100)
    summary.append({"Test": label, "Query": q,
                    "Handled as": r["type"] if r["type"] != "answer" else INTENT_LABELS[r["intent"]],
                    "Confidence": f"{r['confidence']:.0%}",
                    "Rules fired": ", ".join(r["rules_fired"]) or "—"})

# %%
pd.set_option("display.max_colwidth", 80)
test_summary = pd.DataFrame(summary)
print(test_summary.to_string(index=False))
test_summary

# %% [markdown]
# ## Conclusion
# The Diabetes Care Information Assistant integrates:
# * a **labelled corpus** of 189 queries over 7 intents and a **5-level taxonomy**;
# * a **knowledge graph** of 20 concepts / 24 relations used for chain-of-relation reasoning;
# * an **NLP pipeline** (lowercasing, tokenization, stop-word removal, lemmatization);
# * a **TF-IDF + Logistic Regression** intent classifier (compared against Naive Bayes and Linear SVM) evaluated with
#   accuracy, precision, recall, F1 and a confusion matrix;
# * a **domain knowledge base** with topic-level retrieval;
# * an **8-rule reasoning engine** for emergency escalation, critical readings, greetings, unsupported and
#   low-confidence queries, glucose interpretation, medication safety and pregnancy context;
# * an **interactive Colab chat interface** that accepts multiple queries.
#
# It correctly answers in-domain queries (including paraphrases), escalates emergencies, rejects out-of-scope questions
# and asks for clarification when unsure.
