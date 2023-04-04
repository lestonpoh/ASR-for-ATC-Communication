import torch
import pandas as pd
from transformers import BertTokenizerFast, BertForTokenClassification

#load bert model
tokenizer = BertTokenizerFast.from_pretrained('bert_ner_tokenizer')

class BertModel(torch.nn.Module):

    def __init__(self):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=3)

    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output

bert_model = BertModel()
bert_model.load_state_dict(torch.load('ner_state_dict',map_location=torch.device('cpu')))

#miscellaneous

df_numbers = pd.read_csv('numbers.csv')
for i in range(len(df_numbers['words'])):
    if '-' in df_numbers['words'][i]:
        df_numbers['words'][i] = df_numbers['words'][i].replace('-', ' ')

numbers_list = []
for numbers in df_numbers['words']:
    numbers_list.append(numbers)

decimal_list = ['.', 'decimal', 'point']

# FL#
miscellaneous_list_fl = ['ehm', 'ah', 'of', 'feet', 'and', 'below', 'altitude', 'to', 'then']
fl_action_list = ['maintain', 'maintaining', 'climb', 'climbing', 'descend', 'descending', 'descent', 'passing',
                  'approaching', 'approach', 'below', 'above', 'request', 'requesting']
# FL#

# COMM#
miscellaneous_list_communication = ['ehm', 'ah', 'the', 'on', 'frequency', 'now', 'correction', 'is']
# COMM

# HEADING#
miscellaneous_list_heading = ['ehm', 'ah', 'of', 'maintain']
heading_action_list = ['right', 'left', 'turn', 'continue', 'report', 'maintain', 'maintaining', 'present', 'request',
                       'fly', 'set']
# HEADING#

# location list for communication
location_list = [
    'boston',
    'bratislava',
    'control',
    'geneva',
    'ground',
    'karlovy vary',
    'kbely',
    'krakow',
    'marseille',
    'frankfurt',
    'milan',
    'milano',
    'munich',
    'munchen',
    'ostrava',
    'paris',
    'praha',
    'radar',
    'rhein',
    'ruzyne',
    'tower',
    'us',
    'vienna',
    'reims',
    'warsaw',
    'wien',
    'zurich',
]

location_list_splitted = set()
for i in location_list:
    for j in i.split():
        location_list_splitted.add(j)


def extract_heading_info(utterance):
    heading_info = []
    utterance_splitted = utterance.split()
    i = 0

    while i < len(utterance_splitted):

        if utterance_splitted[i] == 'heading':

            heading_temp = []
            heading_temp.append('heading')

            for n in range(1, 5):
                if (i - n) >= 0:
                    if utterance_splitted[i - n] in heading_action_list:
                        heading_temp.insert(0, utterance_splitted[i - n].upper())

            if i < len(utterance_splitted) - 1:

                i += 1

                while utterance_splitted[i] in numbers_list \
                        or utterance_splitted[i] in miscellaneous_list_heading:

                    if utterance_splitted[i] in numbers_list:
                        heading_temp.append(utterance_splitted[i])

                    if i < len(utterance_splitted) - 1:
                        i += 1
                    else:
                        break

                heading_info.extend(heading_temp)

            else:
                heading_info.extend(heading_temp)
                break

        else:
            i += 1

    if ' '.join(heading_info) == '':
        return None
    else:
        return ' '.join(heading_info)


# function to extract radio frequency or transponder code
def extract_communication_info(utterance):
    if 'contact' in utterance:

        contact_freq_info = []
        utterance_splitted = utterance.split()
        i = 0

        while i < len(utterance_splitted) - 1:

            if utterance_splitted[i] == 'contact' \
                    and (utterance_splitted[i + 1] in numbers_list \
                         or utterance_splitted[i + 1] in location_list_splitted \
                         or utterance_splitted[i + 1] in miscellaneous_list_communication) \
                    and utterance_splitted[i - 1] != 'radar':

                contact_freq_temp = []
                contact_freq_temp.append('CONTACT')

                if i < len(utterance_splitted) - 1:
                    i += 1

                    while utterance_splitted[i] in numbers_list \
                            or utterance_splitted[i] in location_list_splitted \
                            or utterance_splitted[i] in decimal_list \
                            or utterance_splitted[i] in miscellaneous_list_communication:

                        contact_freq_temp.append(utterance_splitted[i])

                        if i < len(utterance_splitted) - 1:
                            i += 1
                        else:
                            break

                    contact_freq_info.extend(contact_freq_temp)

                else:
                    break

            else:
                i += 1

        if ' '.join(contact_freq_info) == '':
            return None
        else:
            return ' '.join(contact_freq_info)

    if 'squawk' in utterance or 'squak' in utterance:

        squawk_info = []
        utterance_splitted = utterance.split()
        i = 0

        while i < len(utterance_splitted):
            if (utterance_splitted[i] == 'squawk' or utterance_splitted[i] == 'squawking' or utterance_splitted[
                i] == 'squak'):

                squawk_temp = []
                squawk_temp.append('SQUAWK')

                if i < len(utterance_splitted) - 1:
                    i += 1

                    while utterance_splitted[i] in numbers_list \
                            or utterance_splitted[i] in miscellaneous_list_communication:

                        squawk_temp.append(utterance_splitted[i])

                        if i < len(utterance_splitted) - 1:
                            i += 1
                        else:
                            break

                    squawk_info.extend(squawk_temp)

                else:
                    squawk_info.extend(squawk_temp)
                    break


            else:
                i += 1

        if ' '.join(squawk_info) == '':
            return None
        else:
            return ' '.join(squawk_info)

        # function to extract flight level


def extract_fl_info(utterance):
    if 'level' in utterance:

        fl_info = []
        utterance_splitted = utterance.split()
        i = 0

        while i < len(utterance_splitted):

            if utterance_splitted[i] == 'level':

                fl_temp = []
                fl_temp.append('level')

                for n in range(1, 5):
                    if (i - n) >= 0:
                        if utterance_splitted[i - n] in fl_action_list:
                            fl_temp.insert(0, utterance_splitted[i - n].upper())
                        if utterance_splitted[i - n] == 'flight':
                            fl_temp.insert(0, 'flight')
                if i < len(utterance_splitted) - 1:

                    i += 1

                    while utterance_splitted[i] in numbers_list \
                            or utterance_splitted[i] in miscellaneous_list_fl:

                        fl_temp.append(utterance_splitted[i])
                        if i < len(utterance_splitted) - 1:
                            i += 1
                        else:
                            break

                    fl_info.extend(fl_temp)

                else:
                    fl_info.extend(fl_temp)
                    break

            else:
                i += 1

        if ' '.join(fl_info) == '':
            return None
        else:
            return ' '.join(fl_info)

    elif 'descend' in utterance or 'climb' in utterance:

        fl_info = []
        utterance_splitted = utterance.split()
        i = 0

        while i < len(utterance_splitted):

            if utterance_splitted[i] in ['descend', 'descending', 'climb', 'climbing']:

                fl_temp = []
                fl_temp.append(utterance_splitted[i].upper())

                for n in range(1, 5):
                    if (i - n) >= 0:
                        if utterance_splitted[i - n] in fl_action_list and (i - n) >= 0:
                            fl_temp.insert(0, utterance_splitted[i - n].upper())

                if i < len(utterance_splitted) - 1:

                    i += 1

                    while utterance_splitted[i] in numbers_list \
                            or utterance_splitted[i] in miscellaneous_list_fl \
                            or utterance_splitted[i] in fl_action_list:

                        if utterance_splitted[i] in fl_action_list:
                            fl_temp.append(utterance_splitted[i].upper())
                        else:
                            fl_temp.append(utterance_splitted[i])

                        if i < len(utterance_splitted) - 1:
                            i += 1
                        else:
                            break
                    fl_info.extend(fl_temp)

                else:
                    fl_info.extend(fl_temp)
                    break

            else:
                i += 1

        if ' '.join(fl_info) == '':
            return None
        else:
            return ' '.join(fl_info)


# callsign extraction
unique_labels = ['B-call', 'I-call', 'O']
labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}


def align_word_ids(texts,tokenizer):
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


def evaluate_one_text(bert_model, tokenizer, sentence):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        bert_model = bert_model.cuda()

    text = tokenizer(sentence, padding='max_length', max_length=512, truncation=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence,tokenizer)).unsqueeze(0).to(device)

    logits = bert_model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]

    for i in range(len(prediction_label)):
        if prediction_label[i] == 'I-call':
            if i == 0:
                prediction_label[i] = 'O'
            else:
                if prediction_label[i - 1] == 'I-call' or prediction_label[i - 1] == 'B-call':
                    pass
                else:
                    prediction_label[i] = 'O'

    return prediction_label


def extract_callsign(bert_model, tokenizer, utterance):
    utterance_splitted = utterance.split()
    callsign = []
    prediction_label = evaluate_one_text(bert_model,tokenizer, utterance)
    i = 0
    while i < len(prediction_label):
        if len(callsign) == 0:

            if prediction_label[i] == 'B-call':
                callsign1 = [utterance_splitted[i]]

                i += 1
                while i < len(prediction_label):

                    if prediction_label[i] == 'I-call':
                        callsign1.append(utterance_splitted[i])
                        i += 1
                    else:
                        break
                callsign.append((' '.join(callsign1)).upper())
            else:
                i += 1

        else:

            if prediction_label[i] == 'B-call':
                callsign2 = [utterance_splitted[i]]

                i += 1
                while i < len(prediction_label):

                    if prediction_label[i] == 'I-call':
                        callsign2.append(utterance_splitted[i])
                        i += 1
                    else:
                        break
                callsign.append((' '.join(callsign2)).upper())
                break

            else:
                i += 1

    return callsign

def print_info(prediction):
    print('{:13}{}{}'.format('UTTERANCE', ': ', prediction))
    
    #callsign info
    if len(extract_callsign(bert_model, tokenizer, prediction)) == 0:
        pass
    elif len(extract_callsign(bert_model, tokenizer, prediction)) == 1:
        print('{:13}{}{}'.format('CALLSIGN', ': ' ,extract_callsign(bert_model, tokenizer, prediction)[0]))
    else:
        print('{:13}{}{}'.format('CALLSIGN1', ': ', extract_callsign(bert_model, tokenizer, prediction)[0]))
        print('{:13}{}{}'.format('CALLSIGN2', ': ', extract_callsign(bert_model, tokenizer, prediction)[1]))
    #heading info
    if extract_heading_info(prediction) != None:
        print('{:13}{}{}'.format('HEADING', ': ', extract_heading_info(prediction)))    
    #communication info
    if extract_communication_info(prediction) != None:
        print('{:13}{}{}'.format('COMMUNICATION', ': ', extract_communication_info(prediction)))           
    #flight level info
    if extract_fl_info(prediction) != None:
        print('{:13}{}{}'.format('FLIGHT LEVEL', ': ', extract_fl_info(prediction)))