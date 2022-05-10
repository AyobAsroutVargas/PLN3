import sys
import pandas as pd
from collections import Counter
from pathlib import Path
import numpy as np
import re
import pandas as pd
from math import log
import spacy
import openpyxl

nlp = spacy.load('en_core_web_sm')

spaces_regex = '\s+'
hashtags_regex = '#\S+'
mentions_regex = '@\S+'
unicode_regex = '[^a-z^\ ]'
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


positiveFileName = sys.argv[1]
negativeFileName = sys.argv[2]
testFileName = sys.argv[3]

def preprocess_tweet(text):
  tokens = []
  doc = nlp(text)
  for token in doc:
      word = token.text
      mentions = re.findall(mentions_regex, word)
      hashtags = re.findall(hashtags_regex, word)
      url = re.findall(url_regex, word)
      unicode = re.findall(unicode_regex, word)
      spaces = re.findall(spaces_regex, word)
      if not token.is_stop and not token.is_punct and not word.isdigit() and not len(word) < 1 and not word == '' and not spaces and not mentions and not hashtags and not unicode and not url:
        if(word.islower()):
            tokens.append(word)
            #tokens = np.append(tokens, word)
        else:
            tokens.append(word.lower())
            #tokens = np.append(tokens, word.lower())
  #np_tokens = np.array(tokens)
  return tokens

def readFromFile(name):
  file = open(name, "r")
  words_array = file.read().splitlines()
  file.close()
  return words_array

data = pd.read_excel(testFileName, header=None, engine="openpyxl")
data.columns = ["Tweet"]

negativeModel = np.array(readFromFile(negativeFileName))
positiveModel = np.array(readFromFile(positiveFileName))

negativeModel = np.delete(negativeModel, 0)
negativeModel = np.delete(negativeModel, 0)
negativeModel = np.delete(negativeModel, 0)

positiveModel = np.delete(positiveModel, 0)
positiveModel = np.delete(positiveModel, 0)
positiveModel = np.delete(positiveModel, 0)

neg_model_info = negativeModel[:1]
pos_model_info = positiveModel[:1]
data["processed tweets"] = data["Tweet"].apply(lambda x: preprocess_tweet(x))

tweets = data["processed tweets"].tolist()

negative_frequences = []
negative_logProbs = []
words = []

for row in negativeModel:
  temp = row.split(' ')
  words.append(temp[1])

for row in negativeModel:
  temp = row.split(' ')
  negative_frequences.append(temp[3])
  negative_logProbs.append(temp[5])

positive_frequences = []
positive_logProbs = []

for row in positiveModel:
  temp = row.split(' ')
  positive_frequences.append(temp[3])
  positive_logProbs.append(temp[5])

clasificationFile = open("clasificacion_alu0101350158.txt", "w+")
resumenClasificationFile = open("resumen_alu0101350158.txt", "w+")

tweetIndex = 0
for tweet in tweets:
  probPos = 0
  probNeg = 0
  firstCharacters = data["Tweet"][tweetIndex][0 : 10]
  for word in tweet:
    try:
      wordIndex = words.index(word)
    except:
      wordIndex = words.index("<UNK>")

    if positive_logProbs[wordIndex] == 0:
      wordIndex = words.index("<UNK>")
    if negative_logProbs[wordIndex] == 0:
      wordIndex = words.index("<UNK>")
      
    probPos = probPos + float(positive_logProbs[wordIndex])
    probNeg = probNeg + float(negative_logProbs[wordIndex])

  if probPos > probNeg:
    clasificationFile.write(firstCharacters + ", " + str(probPos) + ", " + str(probNeg) + ", " + "P\n")
    resumenClasificationFile.write("P\n")
  else:
    clasificationFile.write(firstCharacters + ", " + str(probPos) + ", " + str(probNeg) + ", " + "N\n")
    resumenClasificationFile.write("N\n")
  tweetIndex = tweetIndex + 1
