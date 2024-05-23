import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, AdamW

from model import ModelForSequenceClassification
from prepare_joint import generate_joint_data, generate_masks


DEBERTA_PATH = "microsoft/deberta-v3-large"
device = torch.device('cuda:0')

'''
Torch dataset used for the model. 

encoded: DeBERTa-encoded representation of a training instance (claim + all sentences)
labels: evidence labels
nlis: NLI/entailment labels 
'''
class CtDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, nlis):
        self.encoded = encodings
        self.labels = labels
        self.nlis = nlis

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encoded.items()}
        item['labels'] = self.labels[idx]
        item['nli'] = self.nlis[idx]
        return item

    def __len__(self):
        return len(self.labels)     
  

def batch_evidence_label(labels, padding_idx = 2):
    max_sent_len = max([len(label) for label in labels])
    label_matrix = torch.ones(len(labels), max_sent_len) * padding_idx
    label_list = []
    for i, label in enumerate(labels):
        for j, evid in enumerate(label):
            label_matrix[i,j] = int(evid)
        label_list.append([int(evid) for evid in label])
    return label_matrix.long(), label_list


def batch_sentence_mask(masks):
    batch_mask = masks
    padded_batch_mask = list()
    max_shape = -1
    for m in batch_mask:
        if m.size(0) > max_shape:
            max_shape = m.size(0)

    padded_batch_mask = list()
    for m in batch_mask:
        if m.size(0) < max_shape:
            expanded = torch.cat((m, torch.zeros((max_shape - m.size(0), m.size(1)))))
        else:
            expanded = m

        expanded = expanded.view(1, expanded.size(0), expanded.size(1))
        padded_batch_mask.append(expanded)

    padded_batch_mask = torch.cat(padded_batch_mask)
    return padded_batch_mask
    

def token_idx_by_sentence(input_ids, sep_token_id, model_name):
    """
    Compute the token indices matrix of the BERT output.
    input_ids: (batch_size, paragraph_len)
    batch_indices, indices_by_batch, mask: (batch_size, N_sentence, N_token)
    bert_out: (batch_size, paragraph_len,BERT_dim)
    bert_out[batch_indices,indices_by_batch,:]: (batch_size, N_sentence, N_token, BERT_dim)
    """
    padding_idx = -1
    sep_tokens = (input_ids == sep_token_id).bool()
    paragraph_lens = torch.sum(sep_tokens,1).numpy().tolist()
    indices = torch.arange(sep_tokens.size(-1)).unsqueeze(0).expand(sep_tokens.size(0),-1)
    sep_indices = torch.split(indices[sep_tokens],paragraph_lens)
    paragraph_lens = []
    all_word_indices = []
    for paragraph in sep_indices:
        if "large" in model_name:
            paragraph = paragraph[1:]
        word_indices = [torch.arange(paragraph[i]+1, paragraph[i+1]+1) for i in range(paragraph.size(0)-2)]
        paragraph_lens.append(len(word_indices))
        all_word_indices.extend(word_indices)

    indices_by_sentence = nn.utils.rnn.pad_sequence(all_word_indices, batch_first=True, padding_value=padding_idx)
    indices_by_sentence_split = torch.split(indices_by_sentence,paragraph_lens)
    indices_by_batch = nn.utils.rnn.pad_sequence(indices_by_sentence_split, batch_first=True, padding_value=padding_idx)
    batch_indices = torch.arange(sep_tokens.size(0)).unsqueeze(-1).unsqueeze(-1).expand(-1,indices_by_batch.size(1),indices_by_batch.size(-1))
    mask = (indices_by_batch>=0) 

    return batch_indices.long(), indices_by_batch.long(), mask.long()

#Function for evaluating the model output.
def evaluation(model, dataset, data_masks):
    model.eval()
    evidence_predictions = list()
    evidence_labels = list()
    nli_preds = list()
    nli_labels = list()
    batch_size = 4
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(DataLoader(dataset, batch_size = 4, shuffle=False))):
            #encoded = batch["encodings"]
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            transformation_indices = token_idx_by_sentence(input_ids, 102, "bert")
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            padded_evidence_label, evidence_label = batch_evidence_label(batch["labels"], padding_idx = 2)
            sentence_masks = batch_sentence_mask(data_masks[i*batch_size:i*batch_size+batch_size])
            sentence_masks = sentence_masks.to(device)

            nli_label = batch["nli"].to(device)
            
            evidence_out, evidence_preds, evidence_loss, nli_out, nli_loss = \
                model(input_ids, attention_mask, nli_label=nli_label,
                      evidence_label = padded_evidence_label.to(device),
                      transformation_indices=transformation_indices)

            batch_labels = batch["labels"]
            
            batch_selected = (torch.softmax(evidence_out, dim=2)[:,:,1] > 0.5).tolist()
            for idx in range(len(batch_selected)):
                selected = [1 if l else 0 for l in batch_selected[idx]]
                evidence_predictions.extend(selected)
                
                true = [1 if c=="1" else 0 for c in batch_labels[idx]]
                evidence_labels.extend(true)
                
                if len(evidence_labels) > len(evidence_predictions):
                    miss = len(evidence_labels) - len(evidence_predictions)
                    evidence_predictions.extend([0] * miss)
                elif len(evidence_labels) < len(evidence_predictions):
                    miss = len(evidence_predictions) - len(evidence_labels) 
                    evidence_labels.extend([0] * miss)
                    
               
            nli_labels.extend(nli_label.cpu().numpy().tolist())
            nli_preds.extend(nli_out)

    nli_f1 = f1_score(nli_labels,nli_preds, average="macro")
    nli_precision = precision_score(nli_labels,nli_preds,average="macro")
    nli_recall = recall_score(nli_labels,nli_preds,average="macro")
    
    #print(evidence_predictions)
    evidence_f1 = f1_score(evidence_labels,evidence_predictions,average="macro")
    evidence_precision = precision_score(evidence_labels,evidence_predictions,average="macro")
    evidence_recall = recall_score(evidence_labels,evidence_predictions,average="macro")
    return nli_f1, nli_precision, nli_recall, evidence_f1, evidence_precision, evidence_recall
    

#Main training loop.
def train():
    #Load the base model.
    deberta = AutoModel.from_pretrained(DEBERTA_PATH)
    deberta = deberta.to(device)

    #Instantiate the developed model.
    model = ModelForSequenceClassification(deberta)
    model.to(device)
    settings = [{'params': model.deberta.parameters(), 'lr': 1e-5}]
    for module in model.extra_modules:
        settings.append({'params': module.parameters(), 'lr': 5e-6})
        
    #Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_PATH, model_max_length=512)

    #Prepare and generate all data for the model.
    joint_train, nli_labels_train, evidence_labels_train = generate_joint_data(TRAIN_PATH)
    joint_dev, nli_labels_dev, evidence_labels_dev = generate_joint_data(TEST_PATH)

    encoded_train = tokenizer(joint_train, return_tensors='pt',
                        truncation_strategy='only_first', add_special_tokens=True, padding=True)
    encoded_dev = tokenizer(joint_dev, return_tensors='pt',
                        truncation_strategy='only_first', add_special_tokens=True, padding=True)
    train_masks = generate_masks(encoded_train)
    dev_masks = generate_masks(encoded_dev)

    train_dataset = CtDataset(encoded_train, evidence_labels_train, nli_labels_train)
    dev_dataset = CtDataset(encoded_dev, evidence_labels_dev, nli_labels_dev)

    optimizer = torch.optim.AdamW(settings)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, epochs)
    model.train()

    #Hyperparameters.
    epochs = 5
    batch_size = 1
    update_step = 10
    NUM_ACCUMULATION_STEPS = 4
    prev_performance = 0

    #Main training loop.
    for epoch in range(epochs):
        model.train()

        tq = tqdm(DataLoader(train_dataset, batch_size=batch_size, shuffle=False))
        for i, batch in enumerate(tq):
            optimizer.zero_grad()

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            transformation_indices = token_idx_by_sentence(input_ids, 2, "bert")
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            padded_evidence_label, evidence_label = batch_evidence_label(batch["labels"], padding_idx = 2)
            sentence_masks = batch_sentence_mask(train_masks[i*batch_size:i*batch_size+batch_size])
            sentence_masks = sentence_masks.to(device)

            nli_label = batch["nli"].to(device)
                
            evidence_out, evidence_preds, evidence_loss, nli_out, nli_loss = \
                model(input_ids, attention_mask, sentence_masks, nli_label=nli_label,
                    evidence_label = padded_evidence_label.to(device),
                    transformation_indices=transformation_indices)
                
            evidence_loss *= 6. #LOSS RATIO
            loss = evidence_loss + nli_loss
            
            loss = loss / NUM_ACCUMULATION_STEPS
            try:
                loss.backward()
            except:
                optimizer.zero_grad()
                continue
            
            if ((i + 1) % NUM_ACCUMULATION_STEPS == 0) or (i + 1 == len(train_dataset)):
                optimizer.step()
            
            if i % update_step == update_step - 1:
                print(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)}')
                
        scheduler.step()

        
        train_score = evaluation(model, train_dataset, train_masks)
        print(f'Epoch {epoch}, train nli f1 p r: %.4f, %.4f, %.4f, evidence f1 p r: %.4f, %.4f, %.4f' % train_score)

        dev_score = evaluation(model, dev_dataset, dev_masks)
        print(f'Epoch {epoch}, dev nli f1 p r: %.4f, %.4f, %.4f, evidence f1 p r: %.4f, %.4f, %.4f' % dev_score)

        dev_perf = dev_score[0] * dev_score[3]
        print(dev_perf)
        if dev_perf >= prev_performance:
            torch.save(model.state_dict(), "checkpoint.model")
            prev_performance = dev_perf
            print("New model saved.")
        else:
            print("Skip saving model.")




