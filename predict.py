import torch
import os
from transformers import BertTokenizer, BertConfig
from sample import Categorical, WholeWordMasking
import time
from tqdm import tqdm
from compute_metric import bleu, self_bleu
import nltk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--topk", default=30, type=int, required=False)
parser.add_argument("--step_size", default=2, type=int, required=False)
parser.add_argument("--name", default='D3PM', type=str, required=False)
args = parser.parse_args()

step_size = args.step_size
device = 'cuda:0'
model_name = 'bert-base-uncased'
predict_x0 = True
sample_strategy = 'Categorical'
num_steps = 512
schedule = 'mutual'
topk = args.topk
iteration = 2
shape = torch.Size([32, 32])
name = args.name
temperature = 0.1

model_path_dict = {
    'D3PM': ('/remote-home/zfhe/projects/diffusion_torch/D3PM_new_timestep_ckpts/best(1799999).th', 'layerwise'),
    'dbnotimestep': ('/remote-home/zfhe/projects/diffusion_torch/diffusion_bert_base_no_timestep_ckpts/best(1749999).th', 'none'),
    'dbnewtimestep': ('/remote-home/zfhe/projects/diffusion_torch/diffusion_bert_base_new_timestep_ckpts/best(1849999).th', 'layerwise'),
    'dbtokentimestep': ('/remote-home/zfhe/projects/diffusion_torch/model_name_bert-base-uncased_lr_3e-06_seed_42_numsteps_512_sample_Categorical_hybridlambda_0.01_schedule_mutual_new_attmask_ckpts/best(1549999).th', 'token'),
    'word_freq5': ('/remote-home/zfhe/projects/diffusion_torch/model_name_bert-base-uncased_lr_3e-06_seed_42_numsteps_512_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.5_ckpts/best(1749999).th', 'embedding'),
    'word_freq3': ('/remote-home/zfhe/projects/diffusion_torch/model_name_bert-base-uncased_lr_3e-06_seed_42_numsteps_512_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_False_ckpts/best(1849999).th', 'none'),
    'word_freq3_newtimestep': ('/remote-home/zfhe/projects/diffusion_torch/model_name_bert-base-uncased_lr_3e-06_seed_42_numsteps_512_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_False_new_timestep_ckpts/best(1499999).th', 'layerwise'),
    'word_freq_D3PM': ('/remote-home/zfhe/projects/diffusion_torch/model_name_bert-base-uncased_lr_8e-06_seed_42_numsteps_512_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.5_frpmscratch_True_ckpts/best(1849999).th', 'layerwise')
}
model_ckpt_path, timestep = model_path_dict[name]
if name.startswith('word_freq'):
    kind = 'word_freq'
else:
    kind = 'base'

if timestep in ['none', 'token']:
    from models.modeling_bert import BertForMaskedLM
elif timestep == 'embedding':
    from models.modeling_bert_timestep import BertForMaskedLM
elif timestep == 'layerwise':
    from models.modeling_bert_new_timestep import BertForMaskedLM
else:
    raise NotImplementedError


if model_name in ['fnlp/elasticbert-base', 'fnlp/elasticbert-large']:
    model_cls = ElasticBertForPreTraining
    cfg_cls = ElasticBertConfig
    tok_cls = ElasticBertTokenizer
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


if kind == 'word_freq':
    import diffusion_word_freq as diffusion
    word_freq = torch.load(f'./word_freq/{model_name}.pt').to(device)
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
        word_freq_lambda=0.3
    )
elif kind == 'base':
    import diffusion

    diffusion_schedule = diffusion.create_discrete_diffusion_schedule(schedule, num_steps=num_steps)
    diffusion_instance = diffusion.MaskDiffusion(
        dim=tokenizer.vocab_size,
        schedule=diffusion_schedule,
        tokenizer=tokenizer,
        sample_cls=sample_cls,
    )
else:
    raise ValueError





cfg = cfg_cls.from_pretrained(model_name)
cfg.overall_timestep = diffusion_instance.num_steps

if model_name in ['fnlp/elasticbert-base', 'fnlp/elasticbert-large']:
    cfg.num_output_layers = cfg.num_hidden_layers
    cfg.num_base_layers = 0
model = model_cls(cfg).to(device)

ckpt = torch.load(model_ckpt_path)
model.load_state_dict(ckpt['model'])

cls = torch.full((1, 1), fill_value=tokenizer.cls_token_id, device=device)
sep = torch.full((1, 1), fill_value=tokenizer.sep_token_id, device=device)


if timestep == 'none':
    def denoise_fn(targets, timestep, attention_mask):
        assert len(targets.size()) == 2  # bsz * seqlen
        bsz = targets.size(0)
        targets = torch.cat((cls.repeat(bsz, 1), targets, sep.repeat(bsz, 1)), dim=1)
        # attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
        return model(
            input_ids=targets,
            # timestep=timestep - 1,
            # attention_mask=attention_mask
        )['logits'][:, 1:-1, :]
elif timestep == 'token':
    def denoise_fn(targets, timestep, attention_mask):
        assert len(targets.size()) == 2  # bsz * seqlen
        bsz = targets.size(0)
        targets = torch.cat((
            cls.repeat(bsz, 1),
            torch.full((bsz, 1), fill_value=timestep.item() + 110, device=device),
            targets,
            sep.repeat(bsz, 1)
        ), dim=1)
        # attention_mask = torch.cat((torch.ones((bsz, 2), device=device), attention_mask, torch.zeros((bsz, 1), device=device)), dim=1)
        return model(
            input_ids=targets,
            timestep=timestep - 1,
            # attention_mask=attention_mask
        )['logits'][:, 2:-1, :]
elif timestep in ['layerwise', 'embedding']:
    def denoise_fn(targets, timestep, attention_mask):
        assert len(targets.size()) == 2  # bsz * seqlen
        bsz = targets.size(0)
        targets = torch.cat((cls.repeat(bsz, 1), targets, sep.repeat(bsz, 1)), dim=1)
        # attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
        return model(
            input_ids=targets,
            timestep=timestep - 1,
            # attention_mask=attention_mask
        )['logits'][:, 1:-1, :]
else:
    raise NotImplementedError
# att_ones = torch.ones((1, 1), device=device)
# att_zeros = torch.zeros((1, 1), device=device)


model.eval()

with open(f'./temp.txt', 'a+') as fdata:
    with open(f'./generation_results/{name}_step_curve.txt', 'a+') as fcurve:
        sentences = []
        wfs = []
        with torch.no_grad():
            for i in tqdm(range(iteration)):
                start = time.time()
                state = diffusion.discrete_diffusion_predict_fn(
                    shape=shape,
                    denoise_fn=denoise_fn,
                    diffusion=diffusion_instance,
                    predict_x0=predict_x0,
                    sample_cls=sample_cls,
                    step_size=step_size,
                    topk=topk,
                    target_mask=torch.ones(shape, device=device),
                    show_process=False,
                    temperature=temperature
                    # word_freq=True
                    # context_fn=context_fn
                )['final_state']
                t = time.time() - start
                print(t, file=fcurve, end=' ')
                sentence = tokenizer.batch_decode(state)
                sentences.extend(sentence)
                # print(sentence)
                for s in sentence:
                    print(s, file=fdata, flush=True)
