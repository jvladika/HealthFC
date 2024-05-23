import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_dim, n_labels, hidden_dropout_prob = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_dim, n_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

#Applies a linear weighting / self-attention layer.
class WordAttention(nn.Module):
    """
    x: (BATCH_SIZE, N_sentence, N_token, INPUT_SIZE)
    token_mask: (batch_size, N_sep, N_token)
    out: (BATCH_SIZE, N_sentence, INPUT_SIZE)
    mask: (BATCH_SIZE, N_sentence)
    """
    def __init__(self, INPUT_SIZE, PROJ_SIZE, dropout = 0.0):
        super(WordAttention, self).__init__()

        self.activation = torch.tanh
        self.att_proj = nn.Linear(INPUT_SIZE, PROJ_SIZE)
        self.dropout = nn.Dropout(dropout)
        self.att_scorer = nn.Linear(PROJ_SIZE, 1)
        
    def forward(self, x, token_mask):
        proj_input = self.att_proj(self.dropout(x.view(-1, x.size(-1))))
        proj_input = self.dropout(self.activation(proj_input))
        raw_att_scores = self.att_scorer(proj_input).squeeze(-1).view(x.size(0),x.size(1),x.size(2)) # (Batch_size, N_sentence, N_token)
        att_scores = F.softmax(raw_att_scores.masked_fill((1 - token_mask).bool(), float('-inf')), dim=-1)
        att_scores = torch.where(torch.isnan(att_scores), torch.zeros_like(att_scores), att_scores) # Replace NaN with 0
        batch_att_scores = att_scores.view(-1, att_scores.size(-1)) # (Batch_size * N_sentence, N_token)
        out = torch.bmm(batch_att_scores.unsqueeze(1), x.view(-1, x.size(2), x.size(3))).squeeze(1) 
        # (Batch_size * N_sentence, INPUT_SIZE)
        out = out.view(x.size(0), x.size(1), x.size(-1))
        mask = token_mask[:,:,0]
        return out, mask

class DynamicSentenceAttention(nn.Module):
    """
    input: (BATCH_SIZE, N_sentence, INPUT_SIZE)
    output: (BATCH_SIZE, INPUT_SIZE)
    """
    def __init__(self, INPUT_SIZE, PROJ_SIZE, REC_HID_SIZE = None, dropout = 0.1):
        super(DynamicSentenceAttention, self).__init__()
        self.activation = torch.tanh
        self.att_proj = nn.Linear(INPUT_SIZE, PROJ_SIZE)
        self.dropout = nn.Dropout(dropout)
        
        if REC_HID_SIZE is not None:
            self.contextualized = True
            self.lstm = nn.LSTM(PROJ_SIZE, REC_HID_SIZE, bidirectional = False, batch_first = True)
            self.att_scorer = nn.Linear(REC_HID_SIZE, 2)
        else:
            self.contextualized = False
            self.att_scorer = nn.Linear(PROJ_SIZE, 2)
        
    def forward(self, sentence_reps, sentence_mask, att_scores, valid_scores):
        # sentence_reps: (BATCH_SIZE, N_sentence, INPUT_SIZE)
        # sentence_mask: (BATCH_SIZE, N_sentence)
        # att_scores: (BATCH_SIZE, N_sentence)
        # valid_scores: (BATCH_SIZE, N_sentence)
        # result: (BATCH_SIZE, INPUT_SIZE)
        #att_scores = evidence_out[:,:,1] # (BATCH_SIZE, N_sentence)
        #valid_scores = evidence_out[:,:,1] > evidence_out[:,:,0] # Only consider sentences predicted as evidences
        sentence_mask = torch.logical_and(sentence_mask, valid_scores)
        

        if sentence_reps.size(0) > 0:
            att_scores = F.softmax(att_scores.masked_fill((~sentence_mask).bool(), -1e4), dim=-1)
            result = torch.bmm(att_scores.unsqueeze(1), sentence_reps).squeeze(1)
            return result 
        else:
            return sentence_reps[:,0,:]


#The final joint model used for the tasks.
class ModelForSequenceClassification(nn.Module):
    def __init__(self, base_model, hidden_dim=1024, n_labels=2):
        super().__init__()
        
        #DeBERTa-v3-large hidden size iz 1024.
        #We use DeBERTa as the base model for encoding the data instances.
        self.deberta = base_model
        
        self.word_attention = WordAttention(hidden_dim, hidden_dim, dropout=0.0)
        self.evidence_linear = ClassificationHead(hidden_dim=hidden_dim, 
                                        n_labels=n_labels, hidden_dropout_prob=0.0)
        
        self.evidence_criterion = nn.CrossEntropyLoss(ignore_index=2)
        self.nli_criterion = nn.CrossEntropyLoss()
        
        self.sentence_attention = DynamicSentenceAttention(hidden_dim, hidden_dim, dropout=0.0)
        self.nli_linear = ClassificationHead(hidden_dim, 3, hidden_dropout_prob = 0.0)
        
        self.extra_modules = [
            self.sentence_attention,
            self.nli_linear,
            self.evidence_linear,
            self.nli_criterion,
            self.evidence_criterion,
            self.word_attention
        ]            
   
    def select_valid(self, token_reps, token_mask, valid_sentences):
        # token_reps: (BATCH_SIZE, N_sentence, N_token, INPUT_SIZE)
        # token_mask: (BATCH_SIZE, N_sentence, N_token)
        # valid_sentences: (BATCH_SIZE, N_sentence)
    
        #valid_sentences = evidence_out[:,:,1] > evidence_out[:,:,0] # Only consider sentences predicted as evidences
        if valid_sentences.size(1) > token_reps[:,1:,:,:].size(1):
            valid_sentences = valid_sentences[:, :token_reps[:,1:,:,:].size(1)]
               
        evidence_reps = token_reps[:,1:,:,:][valid_sentences]
        evidence_token_mask = token_mask[:,1:,:][valid_sentences]
        evidence_reps = evidence_reps.view(1, evidence_reps.size(0), evidence_reps.size(1), evidence_reps.size(2))
        evidence_token_mask = evidence_token_mask.view(1, evidence_token_mask.size(0), evidence_token_mask.size(1))
        if len(evidence_reps.shape) == 3 or evidence_reps.size(1) == 0:
            evidence_reps = token_reps[:,1,:,:].unsqueeze(1) # First sentence is claim; second is dummy
            evidence_token_mask = token_mask[:,1,:].unsqueeze(1)
        return evidence_reps, evidence_token_mask
        
        
    def forward(
        self,
        encoded,
        attention_mask,
        nli_label, 
        evidence_label,
        transformation_indices,
        sample_p = 1,
        #return_features=True,
        **kwargs
    ):
        batch_indices, indices_by_batch, mask = transformation_indices # (batch_size, N_sep, N_token)
        
       
        # (Batch_size, N_sep, BERT_DIM), (Batch_size, N_sep)

        deberta_out = self.deberta(encoded, attention_mask)[0] # (BATCH_SIZE, sequence_len, BERT_DIM)
        deberta_tokens = deberta_out[batch_indices, indices_by_batch, :]
    
        #represent sentences as weighted self-attention reps
        sentence_reps, sentence_mask = self.word_attention(deberta_tokens, mask)         
        
        #logits of linear predictor
        evidence_out = self.evidence_linear(sentence_reps)      
        
        ## New linear
        att_scores = evidence_out[:,:,1] # (BATCH_SIZE, N_sentence)
         
        if bool(torch.rand(1) < sample_p): # Choose sentence according to predicted evidence
            valid_scores = evidence_out[:,:,1] > evidence_out[:,:,0]
        else:
            valid_scores = evidence_label == 1 # Ground truth
            valid_scores = valid_scores[:,:mask.size(1)]
        
        paragraph_rep = self.sentence_attention(sentence_reps, sentence_mask, att_scores, valid_scores) 
        # (BATCH_SIZE, BERT_DIM) 
        
        nli_out = self.nli_linear(paragraph_rep) # (Batch_size, 3)
        
        #for loss calculation
        if evidence_label.size(1) > evidence_out.size(1):
            evidence_label = evidence_label[:,:evidence_out.size(1)]
        evidence_loss = self.evidence_criterion(evidence_out.view(-1, 2), 
                                                      evidence_label.reshape(-1)) # ignore index 2
        
        evidence_preds = (torch.softmax(evidence_out, dim=1)[:, 1] > 0.5).nonzero().flatten()
        
        nli_loss = self.nli_criterion(nli_out, nli_label)
        nli_out = torch.argmax(nli_out.cpu(), dim=-1).detach().numpy().tolist()
            
        return evidence_out, evidence_preds, evidence_loss, nli_out, nli_loss
    
    
    def evaluate(
        self,
        encoded,
        attention_mask,
        transformation_indices,
        **kwargs
    ):
        batch_indices, indices_by_batch, mask = transformation_indices # (batch_size, N_sep, N_token)
       
        deberta_out = self.deberta(encoded, attention_mask)[0] # (BATCH_SIZE, sequence_len, BERT_DIM)
        deberta_tokens = deberta_out[batch_indices, indices_by_batch, :]
    
        #represent sentences as weighted self-attention reps
        sentence_reps, sentence_mask = self.word_attention(deberta_tokens, mask)         
        
        #logits of linear predictor
        evidence_out = self.evidence_linear(sentence_reps)      
        
        att_scores = evidence_out[:,:,1] # (BATCH_SIZE, N_sentence)    
        valid_scores = evidence_out[:,:,1] > evidence_out[:,:,0]
        
        paragraph_rep = self.sentence_attention(sentence_reps, sentence_mask, att_scores, valid_scores) 
        # (BATCH_SIZE, BERT_DIM) 
        
        evidence_preds = (torch.softmax(evidence_out, dim=1)[:, 1] > 0.5).nonzero().flatten()

        nli_out = self.nli_linear(paragraph_rep) # (Batch_size, 3)        
        nli_out = torch.argmax(nli_out.cpu(), dim=-1).detach().numpy().tolist()
            
        self.deberta.train()
        return evidence_out, evidence_preds, nli_out
    


