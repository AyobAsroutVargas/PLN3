import pandas as pd
from collections import Counter
from pathlib import Path
import numpy as np
import re
import pandas as pd
import math
from math import log

def readFromFile(name):
  file = open(name, "r")
  words_array = file.read().splitlines()
  file.close()
  return words_array

data = pd.read_excel('COV_train.xlsx', header=None, engine="openpyxl")
data.columns = ["Tweet", "Target"]

is_negative = data["Target"]=="Negative"
is_positive = data["Target"]=="Positive"

negative_tweets = np.array(data[is_negative])
positive_tweets = np.array(data[is_positive])

vocabulary = np.array(readFromFile("vocabulario.txt"))
# La primera linea de vocabulario.txt es "Number of tokens: <number>"
vocabulary = np.delete(vocabulary, 0)
negativeCorpus = np.array(readFromFile("corpusN.txt"))
positiveCorpus = np.array(readFromFile("corpusP.txt"))
tempPositiveModel = np.array(readFromFile("modelo_lenguaje_P.txt"))
tempNegativeModel = np.array(readFromFile("modelo_lenguaje_N.txt"))

pos_basic_info = tempPositiveModel[0 : 2]

neg_basic_info = tempNegativeModel[0 : 2]

negativeCounter = Counter(negativeCorpus.tolist())
positiveCounter = Counter(positiveCorpus.tolist())

negativeModel = open("modelo_lenguaje_N.txt", "w+")
positiveModel = open("modelo_lenguaje_P.txt", "w+")

for word in pos_basic_info:
  positiveModel.write(word + "\n")

for word in neg_basic_info:
  negativeModel.write(word + "\n")

# positive_unknowns = 0
# negative_unknowns = 0
unknowns = 0

for word in vocabulary:
  wordNegativeFrequency = negativeCounter[word]
  wordPositiveFrequency = positiveCounter[word]

  if wordNegativeFrequency + wordNegativeFrequency > 3:
    logProbability = log(wordNegativeFrequency + 1) - log(negativeCorpus.size + vocabulary.size)
    negativeModel.write("\nPalabra: " + word + " Frec: " + str(wordNegativeFrequency) + " LogProb: " + str(logProbability))

    logProbabilityN = log(wordPositiveFrequency + 1) - log(positiveCorpus.size + vocabulary.size)
    positiveModel.write("\nPalabra: " + word + " Frec: " + str(wordPositiveFrequency) + " LogProb: " + str(logProbabilityN))
  else:
    negativeModel.write("\nPalabra: " + word + " Frec: 0 " + "LogProb: 0")
    positiveModel.write("\nPalabra: " + word + " Frec: 0 " + "LogProb: 0")
    if wordNegativeFrequency + wordNegativeFrequency > 0:
      unknowns += wordNegativeFrequency + wordNegativeFrequency #1
    else:
      unknowns += 1

  # if wordNegativeFrequency + wordNegativeFrequency > 3:
  #   logProbabilityN = log(wordPositiveFrequency + 1) - log(positiveCorpus.size + vocabulary.size)
  #   positiveModel.write("\nPalabra: " + word + " Frec: " + str(wordPositiveFrequency) + " LogProb: " + str(logProbabilityN))
  # else:
  #   positiveModel.write("\nPalabra: " + word + " Frec: 0 " + " LogProb: 0")
  #   unknowns += 1


negative_unknown_logProb = log(unknowns + 1) - log(negativeCorpus.size + vocabulary.size)
negativeModel.write("\nPalabra: <UNK>" + " Frec: " + str(unknowns) + " LogProb: " + str(negative_unknown_logProb))


positive_unknown_logProb = log(unknowns + 1) - log(positiveCorpus.size + vocabulary.size)
positiveModel.write("\nPalabra: <UNK>" + " Frec: " + str(unknowns) + " LogProb: " + str(positive_unknown_logProb))

positiveModel.close()
negativeModel.close()