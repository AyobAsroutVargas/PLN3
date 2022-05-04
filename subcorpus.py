import spacy
import openpyxl
from pathlib import Path
import numpy as np
import re
import pandas as pd

nlp = spacy.load('en_core_web_sm')

spaces_regex = '\s+'
hashtags_regex = '#\S+'
mentions_regex = '@\S+'
unicode_regex = '[^a-z^\ ]'
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def preprocess(text):
  tokens = []
  for row in text:
    doc = nlp(row)
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
  np_tokens = np.array(tokens)
  np_tokens.sort()
  return np_tokens

data = pd.read_excel('COV_train.xlsx', header=None, engine="openpyxl")
data.columns = ["Tweet", "Target"]

is_negative = data["Target"]=="Negative"
is_positive = data["Target"]=="Positive"

tweets = data.iloc[:,0]

negative_tweets = data[is_negative]
positive_tweets = data[is_positive]

negative_list = np.array(negative_tweets.iloc[:,0])
positive_list = np.array(positive_tweets.iloc[:,0])


# print(tweets.shape)
# print(negative_list)
# print(len(negative_list))
print('preprocesando negativos')
negative_tokens = preprocess(negative_list)
print('preprocesando positivos')
positive_tokens = preprocess(positive_list)

print('escribiendo positivos')
file = open("corpusP.txt", "w+")
for word in positive_tokens:
    file.write(word + "\n")
file.close()

print('escribiendo negativos')
file = open("corpusN.txt", "w+")
for word in negative_tokens:
    file.write(word + "\n")
file.close()

file = open("modelo_lenguaje_P.txt", "w+")
file.write("Numero de documentos (tweets) del corpus: " + str(positive_list.size) + "\n")
file.write("Número de palabras del corpus: " + str(positive_tokens.size) + "\n")
file.close()

file = open("modelo_lenguaje_N.txt", "w+")
file.write("Numero de documentos (tweets) del corpus: " + str(negative_list.size) + "\n")
file.write("Número de palabras del corpus: " + str(negative_tokens.size) + "\n")
file.close()