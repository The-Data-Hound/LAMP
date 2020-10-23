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
def runall(sentences):
    classifier = pipeline('sentiment-analysis')
    label = []
    for sentence in sentences:
        label.append(classifier(sentence))
    return label
def getparent():
    import os
    this_file = os.path.abspath(__file__)
    this_dir = os.path.dirname(this_file)
    parent = os.path.abspath(os.path.join(this_dir, os.pardir))
    return parent
def lamp_pipeline_genome(genome, email, api_key, db = 'both',outfolder='Here', retmax=20, wordcloud = True, have_genome=True):
    from lamp.sentence import clean, tokenize_sentences, species_sentences, word_cloud, word_stats
    from lamp.datamine import datamine
    from lamp.genome import get_blast_species, prokka, card
    from shutil import copyfile
    import os
    from datetime import datetime

    date = datetime.now()
    parent = getparent()
    if outfolder =='Here':
        try:
            os.mkdir('lamp_output')
            outfolder = os.getcwd()+'/lamp_output'
            os.mkdir(outfolder+'/graphs')
            os.mkdir(outfolder+'/imgs')
            for img in os.listdir(parent+'/imgs'):
                copyfile(parent+'/imgs/'+img, outfolder+'/imgs/'+img)
        except:
            outfolder = os.getcwd()+'/lamp_output'
            print('Output Folder Already Exists\nWill Overwrite')
    else:
        try:
            os.mkdir(outfolder+'lamp_output')
            outfolder = outfolder+'lamp_output'
            os.mkdir(outfolder+'/graphs')
            os.mkdir(outfolder+'/imgs')
            os.mkdir(outfolder+'/tables')
            for img in os.listdir(parent+'/imgs'):
                copyfile(parent+'/imgs/'+img, outfolder+'/imgs/'+img)
        except:
            outfolder = os.getcwd()+'/lamp_output'
            print('Output Folder Already Exists\nWill Overwrite')
    if have_genome==True:
        species = get_blast_species(genome, outfolder)
        prokka(genome,outfolder+'/tables/')
        card(genome, outfolde+'/tables/')
    vals = ['a','b','c',1,2,3]
    r = {}
    r['{% query %}']=species
    r['{% date %}']=date
    r['{% email %}']=email
    r['{% positive counts %}'] = 1087
    r['{% negative counts %}'] = 1502
    html = (''.join(open(parent+'/templates/base.html').readlines()))
    for k,v in r.items():
        html = html.replace(str(k),str(v))



    newhtml = open(outfolder+'/click_me.html','w')
    newhtml.write(html)
    newhtml.close()
    abstracts = datamine(species, email, retmax = retmax, api_key=api_key)
    sentences = []
    for a in abstracts:
        sentences+=species_sentences(tokenize_sentences(a), species)
    clean_sentences = [clean(i) for i in sentences]
    label = runall(clean_sentences)
    score = []
    lab = []
    for i in range(0,len(label)):
        score.append(label[i][0]['score'])
        lab.append(label[i][0]['label'])
    if wordcloud == True:
        try:
            word_cloud(word_stats(sentences, mymin =0))
        except:
            print('word cloud broken')
    return pd.DataFrame([clean_sentences,lab, score]), abstracts, sentences
