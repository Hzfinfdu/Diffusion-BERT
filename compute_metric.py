import nltk
import datasets
from tqdm import tqdm
from dataloader import QQPLoader, QTLoader
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction as SF
from transformers import RobertaTokenizer
from torchmetrics.text.rouge import ROUGEScore

def dist1(hypn, n=1):
    dist_list = []
    for hyp in hypn:
        hyp_ngrams = []
        hyp_ngrams += nltk.ngrams(hyp.split(), n)
        total_ngrams = len(hyp_ngrams)
        unique_ngrams = len(list(set(hyp_ngrams)))
        if total_ngrams == 0:
            return 0
        dist_list.append(unique_ngrams/total_ngrams)
    return np.mean(dist_list)

def div4(hypn, n=4):
    hyp_ngrams = []
    for hyp in hypn:
        hyp_ngrams += nltk.ngrams(hyp.split(), n)
    total_ngrams = len(hyp_ngrams)
    unique_ngrams = len(list(set(hyp_ngrams)))
    if total_ngrams == 0:
        return 0
    dist_n = unique_ngrams/total_ngrams
    return dist_n

smooth_method = SF().method4

def get_bleu(recover, reference):
    return sentence_bleu([recover.split()], reference.split(), smoothing_function=smooth_method)

def self_bleu(sentences):
    """
    This function is only used to evaluate the self-BLEU score in conditional generation tasks to align with the implementation in
    https://github.com/Shark-NLP/DiffuSeq/blob/main/scripts/eval_seq2seq.py
    """
    selfBleu = []
    # print(sentences)
    for i, sentence in enumerate(sentences):
        for j in range(i+1, len(sentences)):
            # print(sentence, sentences[j])
            score = get_bleu(sentence, sentences[j])
            selfBleu.append(score)
    if len(selfBleu) == 0:
        selfBleu.append(0)
    return np.mean(selfBleu)


def compute_quality_in_conditional_gen(task_name, file_name):
    rougeScore = ROUGEScore()
    Dataloaders = {
        'qqp': QQPLoader,
        'QT': QTLoader,
    }
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    test_data = Dataloaders[task_name](tokenizer=tokenizer).my_load(splits=['test'])[0]
    bleu = 0.
    rouge = 0.

    with open(file_name, 'r') as f:
        preds = f.readlines()
        for pred, ref in tqdm(zip(preds, test_data['trg'])):
            pred, ref = pred.strip(), ref.strip()
            # print(pred, ref)
            bleu += nltk.translate.bleu_score.sentence_bleu([pred.split()], ref.split(),
                                                            smoothing_function=smooth_method)
            rouge += rougeScore(pred, ref)['rougeL_fmeasure'].item()

        print(bleu / len(preds))
        print(rouge / len(preds))

def compute_quality_in_unconditional_gen(file_name):
    test_data = datasets.load_dataset('lm1b', split='test')
    corpus = [[d['text'].split(' ') for d in test_data]]

    with open(file_name, 'r') as f:
        data = [line.strip().split(' ') for line in f.readlines()]

    res = 0.
    for i, d in enumerate(tqdm(data)):
        res += nltk.translate.bleu_score.corpus_bleu(corpus, [d])

    print(res / len(data))
    # return res / len(data)

def self_bleu_for_unconditional_generation(file_name):
    """
    This function is a canonical implementation of self-BLEU.
    The deviation from the above one is that the references are ALL THE REST sentences and this one uses CORPUS bleu.
    """
    with open(file_name, 'r') as f:
        data = [line.strip().split(' ') for line in f.readlines()]

    res = 0.
    for i in range(len(data)):
        res += nltk.translate.bleu_score.corpus_bleu([data[:i] + data[i + 1:]], [data[i]])
    print(res / len(data))