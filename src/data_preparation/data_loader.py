import pandas as pd
from typing import List, Any
import numpy as np


class DataLoader:
    def __init__(self, df=None):
        self.df = df

    def load_dataset(self, dataset_filename):
        self.df = pd.read_csv(dataset_filename, encoding="latin1", names=["Sentence", "Intent"])
        intent = self.df["Intent"]
        unique_intent = list(set(intent))
        sentences: List[Any] = list(self.df["Sentence"])
        intent_dataset = dict()
        intent_dataset["intent"] = intent
        intent_dataset["unique_intent"] = unique_intent
        intent_dataset["sentences"] = sentences
        return intent_dataset

    def load_glove_embeddings(self,vocab_size, word_tokenizer):
        embeddings_index = dict()
        f = open("../../data/glove.6B.50d.txt")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            ##print(embeddings_index[word])
        f.close()

        embedding_matrix = np.zeros((vocab_size, 50))
        for word, index in word_tokenizer.word_index.items():
            print(word, " ", index)
            if index > vocab_size - 1:
                break
            else:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector

        return embedding_matrix
