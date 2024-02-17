import functools
import os
import sys
import random
import numpy as np
import argparse
import torch
import fitlog
from dataloader import QQPLoader, QTLoader
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig
from models.modeling_bert import BertForMaskedLM
from models.modeling_roberta import RobertaForMaskedLM
import diffusion_condition
from torch.optim import AdamW
import fastNLP
from tqdm import tqdm
from sample import Categorical, WholeWordMasking
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
import wandb

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=200, type=int, required=False)
    parser.add_argument("--model_name_or_path", default='roberta-base', type=str, required=False)
    parser.add_argument("--task_name", default='qqp', type=str, required=False)
    parser.add_argument("--lr", default=5e-5, type=float, required=False)
    parser.add_argument("--batch_size", default=64, type=int, required=False)
    parser.add_argument("--word_freq_lambda", default=0.0, type=float, required=False)
    parser.add_argument("--num_steps", default=2000, type=int, required=False)
    parser.add_argument("--eval_step_size", default=80, type=int, required=False)
    parser.add_argument("--accumulation_steps", default=4, type=int, required=False)
    parser.add_argument("--hybrid_lambda", default=3e-4, type=float, required=False)
    parser.add_argument("--eval_steps", default=2000, type=int, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--device", default='cuda:0', type=str, required=False)
    parser.add_argument("--logging_steps", default=200, type=int, required=False)
    parser.add_argument("--save_steps", default=2000, type=int, required=False)
    parser.add_argument('--predict_x0', default=True, type=bool, required=False)
    parser.add_argument("--load_step", default=-1, type=int, required=False)
    parser.add_argument("--sample_strategy", default='Categorical', type=str, required=False)
    parser.add_argument("--schedule", default='mutual', type=str, required=False)
    parser.add_argument("--from_scratch", default=False, type=bool, required=False)
    parser.add_argument("--timestep", default='none', type=str, required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device("cuda", local_rank)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=9600))
    set_seed(args)

    if dist.get_rank() == 0:
        log_dir = './logs'
        fitlog.set_log_dir(log_dir)
        fitlog.commit(__file__)
        fitlog.add_hyper(args)
        fitlog.add_hyper_in_file(__file__)

        wandb_name = f"{args.model_name_or_path}-lr_{args.lr}-seed_{args.seed}-numsteps_{args.num_steps}-sample_{args.sample_strategy}-schedule_{args.schedule}-hybridlambda_{args.hybrid_lambda}-wordfreqlambda_{args.word_freq_lambda}-fromscratch_{args.from_scratch}-timestep_{args.timestep}"
        wandb.init(
                    project="diffusion_bert", 
                    name=wandb_name, 
                    config={
                    "learning_rate": args.lr,
                    "architecture": args.model_name_or_path,
                    "from_scratch": args.from_scratch,
                    "epochs": args.epochs,
                    "num_steps": args.num_steps,
                    "hybrid_lambda": args.hybrid_lambda,
                    "sample_strategy": args.sample_strategy,
                    "schedule": args.schedule,
                    "word_freq_lambda": args.wordfreqlambda,
                    "seed": args.seed,
                    "timestep": args.timestep,
                    })

    Dataloaders = {
        'qqp': QQPLoader,
        'QT': QTLoader,
    }

    Loader = Dataloaders[args.task_name]

    save_path = f'./model_name_{args.model_name_or_path}_taskname_{args.task_name}_lr_{args.lr}_seed_{args.seed}_numsteps_{args.num_steps}_sample_{args.sample_strategy}_schedule_{args.schedule}_hybridlambda_{args.hybrid_lambda}_wordfreqlambda_{args.word_freq_lambda}_fromscratch_{args.from_scratch}_timestep_{args.timestep}_ckpts'
    

    
    if args.model_name_or_path in ['bert-base-uncased', 'bert-large-uncased']:
        model_cls = BertForMaskedLM
        cfg_cls = BertConfig
        tok_cls = BertTokenizer
    elif args.model_name_or_path in ['roberta-base']:
        model_cls = RobertaForMaskedLM
        cfg_cls = RobertaConfig
        tok_cls = RobertaTokenizer
    else:
        raise NotImplementedError

    tokenizer = tok_cls.from_pretrained(args.model_name_or_path)
    word_freq = torch.zeros(tokenizer.vocab_size)
    assert word_freq.size(0) == tokenizer.vocab_size


    def word_freq_preprocess_fn(wf):
        wf = wf + 1
        wf = wf.log()
        wf = wf / wf.max()

        # range: 0 - 1
        return wf


    word_freq = word_freq_preprocess_fn(word_freq)

    # word_freq[tokenizer.pad_token_id] = 0.  # stable training

    if args.sample_strategy == 'Categorical':
        sample_cls = Categorical()
    elif args.sample_strategy == 'wwm':
        sample_cls = WholeWordMasking(tokenizer)
    else:
        raise ValueError

    diffusion_schedule = diffusion_condition.create_discrete_diffusion_schedule(args.schedule, num_steps=args.num_steps)
    diffusion_instance = diffusion_condition.MaskDiffusion(
        dim=tokenizer.vocab_size,
        schedule=diffusion_schedule,
        tokenizer=tokenizer,
        sample_cls=sample_cls,
        word_freq=word_freq,
        word_freq_lambda=args.word_freq_lambda,
        device=device
    )

    if args.load_step > 0:
        ckpt = torch.load(os.path.join(save_path, f'{args.load_step}.th'))
    cfg = cfg_cls.from_pretrained(args.model_name_or_path)
    cfg.overall_timestep = diffusion_instance.num_steps

    if args.from_scratch:
        model = model_cls(cfg).to(device)
    elif args.load_step <= 0:
        model = model_cls.from_pretrained(args.model_name_or_path, config=cfg).to(device)
    else:
        model = model_cls(cfg).to(device)
        model.load_state_dict(ckpt['model'])

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=10000)

    train_data, dev_data = Loader(tokenizer=tokenizer).my_load(splits=['train', 'validation'])

    if dist.get_rank() == 0:
        logger = fastNLP.logger
        print('# of train data: {}'.format(len(train_data)))
        print('Example:')
        print(train_data[0])
        print('\n# of dev data: {}'.format(len(dev_data)))
        print('Example:')
        print(dev_data[0])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, collate_fn=functools.partial(Loader.collate_fn, tokenizer=tokenizer),
                                               num_workers=4, pin_memory=True, sampler=train_sampler)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size * 2, collate_fn=functools.partial(Loader.collate_fn, tokenizer=tokenizer),
                                             num_workers=4, pin_memory=True, sampler=dev_sampler)


    if args.load_step > 0:
        optimizer.load_state_dict(ckpt['optimizer'])
        warmup_scheduler.load_state_dict(ckpt['warmup_scheduler'])
    model.train()

    def denoise_fn(input_ids, corrupted_input_ids, timestep, attention_mask, target_mask):
        # input_ids 'I am from China. I am Chinese. [PAD][PAD]'
        # corrupted_input_ids  '[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]'
        # target mask  '0 0 0 0 0 1 1 1 1 1 0 0'
        # new_input_ids  'I am from China. [MASK] [MASK] [MASK] [MASK] [PAD] [PAD]'

        # input_ids 'I am from China. I am Chinese. [PAD][PAD]'
        # corrupted_input_ids  '[MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]'
        # target mask  '0 0 0 0 0 1 1 1 1 1 1 1'
        # new_input_ids  'I am from China. [MASK] [MASK] [MASK] [MASK] [MASK] [MASK]'
        new_input_ids = torch.where(target_mask.bool(), corrupted_input_ids, input_ids)
        return model(
            input_ids=new_input_ids,
            attention_mask=attention_mask,
        )['logits']


    if dist.get_rank() == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        best_dev_elbo = float('inf')

    train_loss = .0
    nan_count = 0
    loss_list = [torch.tensor(0., device=device) for _ in range(dist.get_world_size())]
    i = -1
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        dev_loader.sampler.set_epoch(epoch)
        for batch in tqdm(train_loader):
            i += 1
            for k, v in batch.items():
                batch[k] = v.to(device)
            t = diffusion_instance.sample_t()
            metrics = diffusion_condition.compute_kl_reverse_process(
                batch['input_ids'],
                t.to(device),
                denoise_fn=functools.partial(
                    denoise_fn,
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    target_mask=batch['target_mask']
                ),
                diffusion=diffusion_instance,
                target_mask=batch['target_mask'],
                hybrid_lambda=args.hybrid_lambda,
                predict_x0=args.predict_x0,
                word_freq_logits=torch.zeros_like(batch['input_ids'])
            )

            loss = metrics['loss']
            dist.all_gather(loss_list, loss)
            if torch.stack(loss_list).isnan().any():
                nan_count += 1
                if dist.get_rank() == 0:
                    logger.warning(f'NaN encountered {nan_count} times')
                continue
            train_loss += loss.item()
            loss = loss / args.accumulation_steps
            loss.backward()
            # diffusion_instance.update_loss(t.numpy(), loss.item())
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            if i % args.accumulation_steps == args.accumulation_steps - 1:
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()
                warmup_scheduler.step()

            if dist.get_rank() == 0:
                if i % args.logging_steps == args.logging_steps - 1:
                    logger.info(f'Loss at step {i} is {train_loss / args.logging_steps}')
                    fitlog.add_loss(train_loss / args.logging_steps, name='train_loss', step=i)
                    wandb.log({"train loss": train_loss / args.logging_steps, 
                               "train step": i})
                    
                train_loss = .0
            if i % args.eval_steps == args.eval_steps - 1:
                nan_count_in_dev = 0
                model.eval()
                dev_metrics = {
                    'elbo': .0,
                    'elbo_in_bits_per_dim': .0,
                    # 'likelihood': .0,
                    # 'prior': .0,
                }
                with torch.no_grad():
                    for dev_batch in dev_loader:
                        for k, v in dev_batch.items():
                            dev_batch[k] = v.to(device)
                        batch_dev_metrics = diffusion_condition.discrete_diffusion_elbo(
                            dev_batch['input_ids'],
                            denoise_fn=functools.partial(
                                denoise_fn,
                                input_ids=dev_batch['input_ids'],
                                attention_mask=dev_batch['attention_mask'],
                                target_mask=dev_batch['target_mask']
                            ),
                            diffusion=diffusion_instance,
                            target_mask=dev_batch['target_mask'],
                            normalize_without_padding=True,
                            eval_step_size=args.eval_step_size,
                            word_freq_logits=torch.zeros_like(dev_batch['input_ids'])
                        )
                        if dist.get_rank() == 0:
                            m = [torch.tensor(0., device=device) for _ in range(dist.get_world_size())]
                            for name in dev_metrics.keys():
                                dist.gather(batch_dev_metrics[name].squeeze(), m)
                                temp = sum(m)
                                if not torch.isnan(temp):
                                    dev_metrics[name] += temp
                                else:
                                    nan_count_in_dev += 1
                                    logger.warning(f'NaN encountered {nan_count_in_dev} times in dev')
                        else:
                            for name in dev_metrics.keys():
                                dist.gather(batch_dev_metrics[name].squeeze())
                    if dist.get_rank() == 0:
                        for name in dev_metrics.keys():
                            dev_metrics[name] /= (len(dev_data) - nan_count_in_dev * 2 * args.batch_size)
                            fitlog.add_metric(dev_metrics[name], name=name, step=i)
                        
                        wandb.log({"val elbo": dev_metrics["elbo"], 
                                    "val elbo_in_bits_per_dim": dev_metrics["elbo_in_bits_per_dim"],
                                    "train step": i})
                        if dev_metrics['elbo_in_bits_per_dim'] <= best_dev_elbo:
                            best_dev_elbo = dev_metrics['elbo_in_bits_per_dim']
                            fitlog.add_best_metric(dev_metrics['elbo_in_bits_per_dim'], name='dev_elbo_in_bits_per_dim')
                            torch.save({
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'warmup_scheduler': warmup_scheduler.state_dict(),
                            }, f'./{save_path}/best({i}).th')
                model.train()

    wandb.finish()
