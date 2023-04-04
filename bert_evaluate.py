import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.utils.data import DataLoader

df_test = pd.read_csv('bert_ner_data_test.csv')
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

tokenizer = BertTokenizerFast.from_pretrained('bert_ner_tokenizer')

unique_labels=['B-call', 'I-call','O']
labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
print(labels_to_ids)

label_all_tokens = False

def align_label(texts, labels):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df):

        lb = [i.split() for i in df['labels'].values.tolist()]
        txt = df['text'].values.tolist()
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for i in txt]
        self.labels = [align_label(i,j) for i,j in zip(txt, lb)]

    def __len__(self):

        return len(self.labels)

    def get_batch_data(self, idx):

        return self.texts[idx]

    def get_batch_labels(self, idx):

        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels



class BertModel(torch.nn.Module):

    def __init__(self):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output

model=torch.load('ner_trained',map_location=torch.device('cpu'))

def align_word_ids(texts):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

def evaluate_one_text(model, sentence):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    text = tokenizer(sentence, padding='max_length', max_length=512, truncation=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    return(prediction_label)

true_positives = 0
false_positives = 0
false_negatives = 0
count = 0

for i in range(len(df_test['text'])):
    prediction_label = evaluate_one_text(model, df_test['text'][i])

    for j in range(len(prediction_label)):
        if prediction_label[j] == 'I-call':
            if j == 0:
                prediction_label[j] = 'O'
            else:
                if prediction_label[j-1] == 'I-call' or prediction_label[j-1] == 'B-call':
                    pass
                else:
                    prediction_label[j] = 'O'

    callsign = []
    if 'B-call' in prediction_label:
        for j in range(len(prediction_label)):
            if prediction_label[j] == 'B-call' or prediction_label[j] == 'I-call':
                callsign.append(df_test['text'][i].split()[j])
        callsign = ' '.join(callsign)
        if callsign == df_test['callsign'][i]:
            true_positives += 1
        else:
            print('Callsign predicted when there are no callsigns in actual utterance')
            print('utterance: ',df_test['text'][i])
            print('prediction:', callsign)
            print('actual:',df_test['callsign'][i])
            print('-' * 100)
            false_positives += 1

    callsign = []
    if str(df_test['callsign'][i]) != 'nan':
        if 'B-call' not in prediction_label:
            false_negatives += 1
            print('Callsign not predicted when callsign is in actual utterance')
            print('utterance: ',df_test['text'][i])
            print('Prediction: nan')
            print('actual:', df_test['callsign'][i])
            print('-' * 100)
        elif 'B-call' in prediction_label:
            for j in range(len(prediction_label)):
                if prediction_label[j] == 'B-call' or prediction_label[j] == 'I-call':
                    callsign.append(df_test['text'][i].split()[j])
            callsign = ' '.join(callsign)
            if callsign != df_test['callsign'][i]:
                false_negatives += 1
                print('Callsign predicted is different from actual callsign')
                print('utterance: ',df_test['text'][i])
                print('prediction:', callsign)
                print('actual:', df_test['callsign'][i])
                print('-' * 100)


precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = (2 * precision * recall) / (precision + recall)
print('Precision:', precision)
print('Recall', recall)
print('F1-score:',f1_score)
