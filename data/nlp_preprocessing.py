import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        all_sentences = positive + negative
        vocab = set()
        for sentence in all_sentences:
            for word in sentence.split():
                vocab.add(word)
        sorted_vocab = sorted(list(vocab))
        token_map = {word: i+1 for i, word in enumerate(sorted_vocab)}
        
        # 2. Encode each sentence by replacing words with their IDs
        output = []
        for sentence in all_sentences:
            tokenized_sentence = []
            for word in sentence.split():
                tokenized_sentence.append(token_map[word])
            output.append(torch.tensor(tokenized_sentence))
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        return nn.utils.rnn.pad_sequence(output, batch_first = True)
