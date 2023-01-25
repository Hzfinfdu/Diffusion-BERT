import torch
import os
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig
from transformers import BertTokenizer as ElasticBertTokenizer
from models.modeling_roberta import RobertaForMaskedLM
from sample import Categorical, WholeWordMasking
import time
from tqdm import tqdm
from compute_metric import self_bleu, dist1, div4
import nltk
import argparse
from models.modeling_bert import BertForMaskedLM
from dataloader import QQPLoader, QTLoader
import diffusion_condition as diffusion
import functools
from MBR_decoding import selectBest
from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--topk", default=15, type=int, required=False)
parser.add_argument("--step_size", default=10, type=int, required=False, help='Time step size during inference')
parser.add_argument("--task_name", default='qqp', type=str, required=False)
parser.add_argument("--ckpt_path", default='', type=str, required=False)
parser.add_argument("--MBR_size", default=3, type=int, required=False, help=r'The MBR size \mathcal{S}. Generates that many sentences for 1 source sentence.')
parser.add_argument("--seq_len", default=128, type=int, required=False, help='Max seq length in generation')
args = parser.parse_args()

step_size = args.step_size
device = 'cuda:0'
model_name = 'roberta-base'
predict_x0 = True
sample_strategy = 'Categorical'
num_steps = 2000
schedule = 'mutual'
topk = args.topk
task_name = args.task_name
model_ckpt_path = args.ckpt_path
temperature = 1.0
batch_size = 32
MBR_size = args.MBR_size

if not os.path.exists('./generation_results'):
    os.mkdir('generation_results')

Dataloaders = {
    'qqp': QQPLoader,
    'QT': QTLoader,
}

if model_name in ['roberta-base']:
    model_cls = RobertaForMaskedLM
    cfg_cls = RobertaConfig
    tok_cls = RobertaTokenizer
elif model_name in ['bert-base-uncased', 'bert-large-uncased']:
    model_cls = BertForMaskedLM
    cfg_cls = BertConfig
    tok_cls = BertTokenizer
else:
    raise NotImplementedError


tokenizer = tok_cls.from_pretrained(model_name)


if sample_strategy == 'Categorical':
    sample_cls = Categorical()
elif sample_strategy == 'wwm':
    sample_cls = WholeWordMasking(tokenizer)
else:
    raise ValueError


word_freq = torch.ones(tokenizer.vocab_size)
def word_freq_preprocess_fn(wf):
    wf = wf + 1
    wf = wf.log()
    wf = wf / wf.max()

    # range: 0 - 1
    return wf

word_freq = word_freq_preprocess_fn(word_freq)
diffusion_schedule = diffusion.create_discrete_diffusion_schedule(schedule, num_steps=num_steps)
diffusion_instance = diffusion.MaskDiffusion(
    dim=tokenizer.vocab_size,
    schedule=diffusion_schedule,
    tokenizer=tokenizer,
    sample_cls=sample_cls,
    word_freq=word_freq,
    word_freq_lambda=0.
    )

cfg = cfg_cls.from_pretrained(model_name)
cfg.overall_timestep = diffusion_instance.num_steps

model = model_cls(cfg).to(device)
ckpt = torch.load(model_ckpt_path)

# print(ckpt['model'])

model.load_state_dict(ckpt['model'])


def denoise_fn(input_ids, corrupted_input_ids, timestep, attention_mask, target_mask):
    new_input_ids = torch.where(target_mask.bool(), corrupted_input_ids, input_ids)
    return model(
        input_ids=new_input_ids,
        attention_mask=attention_mask,
        # token_type_ids=token_type_ids if model_name in ['bert-base-uncased', 'bert-large-uncased'] else None
    )['logits']

model.eval()


def process_fn_in_collate(wf):
    return wf - wf.mean()

def collate_fn(batch_input):
    input_ids = pad_sequence([torch.tensor(
        [tokenizer.cls_token_id] + d['source'] + [tokenizer.mask_token_id] * (args.seq_len - len(d['source']) - 1)
    ) for d in batch_input], batch_first=True)

    attention_mask = torch.ones_like(input_ids)

    target_mask = torch.stack([torch.cat([
        torch.zeros(len(d['source']) + 1), torch.ones(input_ids.size(1) - len(d['source']) - 1)
    ]) for d in batch_input])
    target_start = torch.tensor([len(d['source']) + 1 for d in batch_input]).long()

    assert input_ids.size() == attention_mask.size() == target_mask.size()
    return {
        'input_ids': input_ids.repeat(1, MBR_size).view(-1, input_ids.size(-1)),
        'attention_mask': attention_mask.repeat(1, MBR_size).view(-1, input_ids.size(-1)),
        'target_mask': target_mask.repeat(1, MBR_size).view(-1, input_ids.size(-1)),
        'target_start': target_start
    }

test_data = Dataloaders[task_name](tokenizer=tokenizer).my_load(splits=['test'])[0]
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True)

with open(f'./generation_results/{task_name}_{args.topk}_MBR.txt', 'w+') as f_mbr:
    with open(f'./generation_results/{task_name}_{args.topk}_raw.txt', 'w+') as f_raw:
        s_bleu = 0.
        dist_1 = 0.
        div_4 = 0.
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                state = diffusion.discrete_diffusion_predict_fn(
                    input_ids=batch['input_ids'],
                    target_mask=batch['target_mask'],
                    denoise_fn=functools.partial(
                                denoise_fn,
                                input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                # token_type_ids=batch['token_type_ids'],
                                target_mask=batch['target_mask']
                            ),
                    diffusion=diffusion_instance,
                    predict_x0=predict_x0,
                    sample_cls=sample_cls,
                    step_size=step_size,
                    topk=topk,
                    show_process=False,
                    temperature=temperature,
                    MBR_size=MBR_size
                )['final_state']
                state = state.view(batch_size, MBR_size, -1)
                for start, pred in zip(batch['target_start'], state):
                    pred_candidate_ids = pred[:, start:]
                    sentences = tokenizer.batch_decode(pred_candidate_ids)
                    try:
                        sentences = [pred[:pred.index('</s>')] for pred in sentences]
                    except ValueError:
                        pass
                    print(selectBest(sentences), file=f_mbr, flush=True)
                    print(' '.join(sentences), file=f_raw, flush=True)
                    s_bleu += self_bleu(sentences)
                    dist_1 += dist1(sentences)
                    div_4 += div4(sentences)
                print(dist_1 / (batch_size * (i + 1)))
                print(div_4 / (batch_size * (i + 1)))
        print(f'dist1 {dist_1 / len(test_data)}, div4 {div_4 / len(test_data)}, self_bleu {s_bleu / len(test_data)}')
