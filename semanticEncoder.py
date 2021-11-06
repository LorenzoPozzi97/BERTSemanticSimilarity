import json
from google.colab import files
import pandas as pd
import itertools
import torch
import numpy as np
import math
import pickle

from transformers import AutoTokenizer, AutoModel

model_name = "GroNLP/bert-base-dutch-cased"
tz = AutoTokenizer.from_pretrained(model_name)  # v1 is the old vocabulary
model = AutoModel.from_pretrained(model_name)

def longest_statement(dataset): 
  list_of_seq = []
  for row in range(len(dataset)):
    list_of_seq.append([list(dataset.iloc[row][0].values())[0]])

    for dictionary in dataset.iloc[row][1]:
      list_of_seq.append(list(dictionary.values())[0])  
      
  #list_of_seq1 = list(dict.fromkeys(list_of_seq))
  #print(list_of_seq1)
  print("Max length:", len(tz.convert_tokens_to_ids(tz.tokenize(max(list_of_seq, key=len)))))

# function to create list containing text input (in the first position) + scripted statements for row in dataset

def list_of_encodings(dataset):
  list_of_enc = []
  for row in range(len(dataset)):
    tokens = {'input_ids': [], 'attention_mask': []}
    sentences = [list(dataset.iloc[row][0].values())[0]]

    for option in dataset.iloc[row][1]:
      sentences.append(list(option.values())[0])

    for sentence in sentences: 
      new_tokens = tz.encode_plus(sentence, 
                                  max_length=41, 
                                  truncation=True, 
                                  padding='max_length', 
                                  return_tensors='pt')
      tokens['input_ids'].append(new_tokens['input_ids'][0])
      tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    list_of_enc.append(tokens)

  return list_of_enc

longest_statement(trainSet)

# convert the list of tensor in a single tensor
encodings = list_of_encodings(trainSet)
for encoding in encodings:
  encoding['input_ids'] = torch.stack(encoding['input_ids'])
  encoding['attention_mask'] = torch.stack(encoding['attention_mask'])

outputs = []
for encoding in encodings: 
  outputs.append(model(**encoding))

path = "/content/drive/MyDrive/Colab_Notebooks/trainEmbeddings[0-79]"

open_file = open(path, "wb")
pickle.dump(outputs, open_file)

open_file.close()

masks = []
masked_embeddings = []
summed = []
counts = []
mean_pooled = []


for i in range(0, 533, 80):
  filename = "/content/drive/MyDrive/Colab_Notebooks/trainEmbeddings["+ str(i) + "-" + str(i+80-1) + "]"
  embeddings = []
  masks = []
  masked_embeddings = []
  
  if i==480:
    filename = "/content/drive/MyDrive/Colab_Notebooks/trainEmbeddings["+ str(i) + "-" + str(533) + "]"

  infile = open(filename,'rb')
  outputs = pickle.load(infile)
  infile.close()
  
  for output in outputs:
    embeddings.append(output.last_hidden_state)

  e = encodings[i:i+80]

  #add dimension to attentio_mask
  for j in range(len(e)):
    masks.append(e[j]['attention_mask'])
    masks[-1] = masks[-1].unsqueeze(-1).expand(embeddings[j].shape)
    masks[-1] = masks[-1].float()
  #print(masks[0].shape)
  
  #apply the mask to embeddings
  for x in range(len(embeddings)):
    masked_embeddings.append(embeddings[x] * masks[x])


  #create a pickle file for each batch of masked embeddings
  new_path = "/content/drive/MyDrive/Colab_Notebooks/maskedTrainEmbeddings["+ str(i) + "-" + str(i+80-1) + "]"
  if i==480:
    new_path = "/content/drive/MyDrive/Colab_Notebooks/maskedTrainEmbeddings["+ str(i) + "-" + str(533) + "]"
  print(new_path)
  open_file = open(new_path, "wb")
  pickle.dump(masked_embeddings, open_file)
  open_file.close()
  
  mean_pooled = []
summed = []
counts = []

for i in range(0, 533, 80):
  filename = "/content/drive/MyDrive/Colab_Notebooks/maskedTrainEmbeddings["+ str(i) + "-" + str(i+80-1) + "]"
  if i==480:
    filename = "/content/drive/MyDrive/Colab_Notebooks/maskedTrainEmbeddings["+ str(i) + "-" + str(533) + "]"

  print(filename)
  infile = open(filename,'rb')
  me = pickle.load(infile)
  infile.close()

  for masked_emb in me:
    #print(masked_emb.shape)
    sum = torch.sum(masked_emb, 1)
    #print(summed_masked_emb.shape)
    count = torch.count_nonzero(masked_emb, 1)
    #print(count.shape)
    #print("-------")

    mean_pooled.append(sum / count)
  #print((mean_pooled))
    

  

print(len(mean_pooled))
"""for m in masks:
  counts.append(torch.clamp(m.sum(1), min=1e-9))"""


evaluation_vect = []
for i in range(len(mean_pooled)):
  mean_pooled[i] = mean_pooled[i].detach().numpy()
  sim = cosine_similarity([mean_pooled[i][0]], mean_pooled[i][1:])
  evaluation_vect.append([sim, trainSet['goldenMatch'][i]])
  #print(cosine_similarity([mean_pooled[i][0]], mean_pooled[i][1:]), trainSet['goldenMatch'][i])
print((trainSet['goldenMatch'][0]))
for i in range(len(evaluation_vect)):
  evaluation_vect[i][0] = np.argmax(evaluation_vect[i][0][0])+1
  if math.isnan(trainSet['goldenMatch'][i]) == False:
    evaluation_vect[i][1] = int(trainSet['goldenMatch'][i])
  else: 
    evaluation_vect[i][1] = 0

correct = 0
for v in evaluation_vect:
  if v[0] == v[1]:
    correct += 1
print("accuracy: ", correct/len(evaluation_vect)) 
