import nltk
import numpy as np
from compute_metric import get_bleu

def selectBest(sentences):
    selfBleu = [[] for _ in range(len(sentences))]
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            score = get_bleu(s1, s2)
            selfBleu[i].append(score)
    idx = np.argmax(np.sum(selfBleu, -1))
    return sentences[idx]
