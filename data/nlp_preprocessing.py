import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        vocab = set()
        all_sentences = positive + negative
        for sentence in all_sentences:
            for word in sentence.split():
                vocab.add(word)
        sorted_vocab = sorted(list(vocab))
        token_map = {} # dictionary mapping word_token to integer
        for i, word in enumerate(sorted_vocab):
            token_map[word] = i+1

        # 2. Encode each sentence by replacing words with their IDs
        encoded_all_sentences = []
        for sentence in all_sentences:
            integers = []
            for word in sentence.split():
                integers.append(token_map[word])
            encoded_all_sentences.append(torch.tensor(integers))

        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)

        return nn.utils.rnn.pad_sequence(encoded_all_sentences, batch_first=True)
        
        

