#this may be all that is needed
from transformers import pipeline
import os
import pandas as pd
from transformers import *
import pandas as pd
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from sklearn.model_selection import train_test_split
NLP_SENTENCE_LABELS = [
    ('1','POSITIVE'),('0','NEGATIVE'),('2','NEUTRAL'),('-1','BAD SENTENCE'),]

def train(  trainset='bacteria_lamp.tsv',
            epochs = 100,
            outfolder = 'path/to/out/folder',
            add_drop = 'add_drop.tsv',
            train_size = 0.80,
            test_size = 0.10,
            valid_size = 0.10):

    # prep our sentences
    from lamp.sentence import clean
    print('\ntraining!\n')
    # read in our training csv
    df = pd.read_csv(trainset,sep = '\t')
    # follow the instructions from:
    # https://huggingface.co/transformers/custom_datasets.html
    train_labels = list(df.label.values)[:int(len(list(df.label.values))*train_size)]
    train_labels = [0 if x==1 else 1 for x in train_labels]
    test_labels  = list(df.label.values)[int(len(list(df.label.values))*test_size):int(len(list(df.label.values))*(test_size+train_size))]
    test_labels = [0 if x==1 else 1 for x in test_labels]
    val_labels   = list(df.label.values)[int(len(list(df.label.values))*(test_size+train_size)):]
    val_labels = [0 if x==1 else 1 for x in val_labels]
    #try swapping labels to train
    # 1 = Good; 0 = Bad

    texts = []
    for i in df.text:
        texts.append(clean(i, add_drop = add_drop))

    train_texts = texts[:int(len(texts)*(train_size))]
    test_texts  = texts[int(len(texts)*train_size):int(len(texts)*(test_size+train_size))]
    val_texts = texts[int(len(texts)*(test_size+train_size)):]

    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings   = tokenizer(val_texts,   truncation=True, padding=True)
    test_encodings  = tokenizer(test_texts,  truncation=True, padding=True)

    import torch
    class BioDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = BioDataset(train_encodings, train_labels)
    val_dataset = BioDataset(val_encodings, val_labels)
    test_dataset = BioDataset(test_encodings, test_labels)
    from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments,DistilBertForSequenceClassification
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=10,              # total number of training epochs
        per_device_train_batch_size=10,  # batch size per device during training
        per_device_eval_batch_size=20,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
    )
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    trainer = Trainer(
        model=  model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    trainer.train()
    trainer.save_model('./results')
    tokenizer.save_pretrained("./results")

def evaluate(eval_tsv, model, add_drop='add_drop.tsv'):
    from lamp.sentence import clean
    df = pd.read_csv(eval_tsv,sep = '\t')
    train_labels = list(df.label.values)
    train_conv = []
    for i in train_labels:
        if i ==0:
            train_conv.append('POSITIVE')
        elif i ==1:
            train_conv.append('NEGATIVE')
    classifier = pipeline('sentiment-analysis',model=model)
    # 1 = Good; 0 = Bad
    eval_labels = []
    for i in df.text:
        eval_labels.append(classifier((clean(i, add_drop = add_drop)))[0]['label'])
    correct = 0
    ml = [['text','ann','human']]
    for k,v,t in zip(eval_labels,train_conv, list(df.text.values)):
        if k ==v:
            correct+=1
        else:
            ml.append([t,k,v])
    d = {}
    d['accuracy']=(correct/len(eval_labels))*100
    d['misslabeled']=pd.DataFrame(ml[1:], columns = ml[0])
    return d
