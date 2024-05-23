import gc
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    return {"accuracy": acc, "precision" : prec, "recall" : recall, "f1": f1}


class CtDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

for fold_idx in range(len(folds)):
    print("Fold ", fold_idx)
        
    fold = folds[fold_idx]
    fold_train = fold[0]
    fold_test = fold[1]
    
    m = "microsoft/deberta-v3-large"
    models = list()

    model = None
    tokenizer = None

    torch.cuda.empty_cache()
    gc.collect()

    ## Test with gold evidence
    
    joint_train = list()
    labels_train = list()
    for cid in range(len(fold_train)):
        entry = fold_train[cid]
        claim = entry.claim
        evs_p = entry.sentences
        ids = entry.order
        string = claim + " [SEP] "

        for sid in ids:
            candidate_sentence = evs_p[sid]
            string += candidate_sentence + " "
        
        joint_train.append(string)  
        labels_train.append(entry.label)
    #print(joint_train[:3], labels_train[:3])

    joint_dev = list()
    labels_dev = list()
    for cid in range(len(fold_test)):
        entry = fold_test[cid]
        claim = entry.claim
        evs_p = entry.sentences
        ids = entry.order
        string = claim + " [SEP] "

        for sid in ids:
            candidate_sentence = evs_p[sid]
            string += candidate_sentence + " "
        
        joint_dev.append(string)    
        labels_dev.append(entry.label)
    #print(joint_dev[:3], labels_dev[:3])

    tokenizer = AutoTokenizer.from_pretrained(m, model_max_length=256)
    model = AutoModelForSequenceClassification.from_pretrained(m, num_labels=3, ignore_mismatched_sizes=True)

    trains = tokenizer(joint_train, return_tensors='pt',
                         truncation_strategy='only_first', add_special_tokens=True, padding=True)
    tests = tokenizer(joint_dev, return_tensors='pt',
                         truncation_strategy='only_first', add_special_tokens=True, padding=True)
    
    #Convert data into datasets
    train_dataset = CtDataset(trains, labels_train)
    test_dataset = CtDataset(tests, labels_dev)
    

    batch_size = 4
    logging_steps = len(fold_train) // batch_size

    model_name = f"finetuned-model"

    training_args = TrainingArguments(output_dir=model_name,
                                 dataloader_pin_memory=True, dataloader_num_workers=4,
                                fp16=True,
                                  warmup_ratio=0.06,
                                   gradient_accumulation_steps=4,
                                num_train_epochs=7,
                                learning_rate=1e-5,
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                weight_decay=0.01,
                                evaluation_strategy="epoch",
                                   save_strategy="no",
                                disable_tqdm=False,
                                logging_steps=logging_steps,
                                push_to_hub=False)

    trainer = Trainer(model=model, args=training_args,
                    compute_metrics=compute_metrics,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    tokenizer=tokenizer)

    print(m)
    trainer.train();
    
    
    ## Test with selected evidence
    
    fold_evidence = selected_evidence[fold_idx]
    
    unsorted_fold_test = unsorted_folds[fold_idx][1]
    joint_test = list()
    for cid in range(len(unsorted_fold_test)):
        entry = unsorted_fold_test[cid]
        claim = entry.claim
        evs_p = entry.sentences

        for sid in range(len(evs_p)):
            candidate_sentence = evs_p[sid]
            joint = candidate_sentence + " [SEP] " + claim
            joint_test.append(joint)


    nli_test = list()
    idx = 0
    for cid in range(len(fold_test)):
        entry = fold_test[cid]
        claim = entry.claim
        string = claim + " [SEP] "

        for i in range(idx, len(joint_test)):
            j = joint_test[i]
            if claim in j:
                if i in fold_evidence:
                    second = j.split(" [SEP] ")[0]
                    string += second
                    string += " "
            else:
                idx = i
                break
        nli_test.append(string)


    from torch.utils.data import DataLoader

    nli_encoded = tokenizer(nli_test, return_tensors='pt',
                             truncation_strategy='only_first', add_special_tokens=True, padding=True)

    nli_dataset = CtDataset(nli_encoded, np.zeros(len(nli_test)))

    test_loader = DataLoader(nli_dataset, batch_size=8,
                             drop_last=False, shuffle=False, num_workers=4)

    model.eval()
    model = model.to("cuda")

    result = np.zeros(len(test_loader.dataset))    
    index = 0

    with torch.no_grad():
        for batch_num, instances in enumerate(test_loader):
            print(batch_num)
            input_ids = instances["input_ids"].to("cuda")
            attention_mask = instances["attention_mask"].to("cuda")
            logits = model(input_ids=input_ids,
                                          attention_mask=attention_mask)[0]
            probs = logits.softmax(dim=1)

            pred = probs.argmax(-1).to("cpu")
            #pp = probs[:,1]
            #ones = torch.where(pp > 0.2)[0] 
            #pred = torch.zeros(len(pp))
            #pred[ones] = 1

            result[index : index + pred.shape[0]] = pred.flatten().to("cpu")
            index += pred.shape[0]

    y_pred = result
    y_true = list()
    for e in fold_test:
        y_true.append(e.label)
    y_true = np.array(y_true)
    
    from sklearn.metrics import f1_score, precision_score, recall_score

    f1 = f1_score(y_true,y_pred,average="macro")
    precision = precision_score(y_true,y_pred,average="macro")
    recall = recall_score(y_true,y_pred,average="macro")
    print("Selected evidence results")
    print("F1 ", f1, " P ", precision, " R ", recall)
    