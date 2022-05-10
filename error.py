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

def readFromFile(name):
  file = open(name, "r")
  words_array = file.read().splitlines()
  file.close()
  return words_array

predictionFile = readFromFile("resumen_alu0101350158.txt")

actualData = pd.read_excel("COV_train.xlsx", header=None, engine="openpyxl")
actualData.columns = ["Tweet", "Target"]

aciertos = 0
totalTweets = len(predictionFile)
for i in range(0, len(predictionFile)):
  if predictionFile[i] == actualData["Target"][i][0]:
    aciertos = aciertos + 1

result = (aciertos / totalTweets) * 100

print("Acierto: " + str(result) + "%")