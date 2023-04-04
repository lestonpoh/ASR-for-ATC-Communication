import pickle
import random
import spacy
from tqdm import tqdm
from spacy.training import Example

with open('/home/lpoh003/spacy_ner/train_data', 'rb') as outputfile:
    train_data = pickle.load(outputfile)
print(train_data)

model = None
output_dir='/home/lpoh003/spacy_ner/spacy_ner_trained'
n_iter=100

#load the model

if model is not None:
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('en')
    print("Created blank 'en' model")

if 'ner' not in nlp.pipe_names:
    nlp.add_pipe('ner')
    ner = nlp.get_pipe('ner')
else:
    ner = nlp.get_pipe('ner')

for _, annotations in train_data:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        for text, annotations in tqdm(train_data):
            doc=nlp.make_doc(text)
            example=Example.from_dict(doc,annotations)
            nlp.update(
                [example],
                drop=0.5,
                sgd=optimizer,
                losses=losses)
        print(losses)


nlp.to_disk(output_dir)
print("Saved model to", output_dir)