import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

class Entry:
    def __init__(self, cid, claim, explanation, label, sentences, rationales, order):
        self.cid = cid
        self.claim = claim
        self.explanation = explanation
        self.label = label
        self.sentences = sentences
        self.rationales = rationales
        self.order = order

df = pd.read_csv("healthFC_annotated.csv")
entries = list()

for idx, row in df.iterrows():
    claim = row['en_claim']
    explanation = row['en_explanation']
    label = row['label']
    sentences = row['en_sentences']
    ids = row['en_ids']
    
    e = Entry(idx, claim, explanation, label, sentences, ids, ids)
    entries.append(e)
    

X = list()
y = list()
for idx in range(len(entries)):
    X.append(entries[idx])
    y.append(int(entries[idx].label))
    
X = np.array(X)
y = np.array(y)

skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
skf.get_n_splits(X, y)

entries = np.array(entries)
folds = list()

explanations = np.array([e.explanation for e in entries])
expl_folds = list()

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    
    fold_train = entries[train_index]
    fold_test = entries[test_index]
    folds.append((fold_train, fold_test))
    
    expl_fold_train = explanations[train_index]
    expl_fold_test = explanations[test_index]
    expl_folds.append((expl_fold_train, expl_fold_test))
    #google_fold_train = full_evidence[train_index]
    #google_fold_test = full_evidence[test_index]
    #google_folds.append((google_fold_train, google_fold_test))

len(folds)

from ast import literal_eval

for idx in range(len(entries)):
    entries[idx].sentences = literal_eval(entries[idx].sentences)
    
for idx in range(len(entries)):
    entries[idx].order = [int(i) for i in entries[idx].rationales[1:-1].split(", ")]



