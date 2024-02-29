import os, glob, sys
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

with open("similarity.txt", "r") as f1:
    lines = f1.readlines()

sentences = []
for line in lines:
    sentences.append(line.strip().split("\n")[0])

model = SentenceTransformer('sentence-transformers/LaBSE')
embeddings = model.encode(sentences, normalize_embeddings=True)

for i in range(0, len(sentences)):
    score = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[i].reshape(1, -1))
    print(score[0][0])
