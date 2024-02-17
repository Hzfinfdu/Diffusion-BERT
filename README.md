# Diffusion Models beat Auto-regressive Models on Text Generations

Adapted from [DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models](https://arxiv.org/abs/2211.15029).


### Preparing the Environment & Data

We have prepared the required environment in `requirements.txt`. We use `fitlog` to monitor our training process.

```bash
conda create --name DB python=3.8
conda activate DB
pip install -r requirements.txt
```

The LM1B dataset is available at 🤗 Datasets. We use the same data for conditional generation in [DiffuSeq](https://github.com/Shark-NLP/DiffuSeq). We have prepared the loading script in `conditional_data/`. One only needs to download the `*.jsonl` files to the corresponding directories.

### Training

To add spindle schedule into our training process, we first need to run `python word_freq.py` to get the frequency in the text corpus.

We have prepared the default training parameters in `run.sh` for unconditional generation and `run_condition.sh` for *Seq2seq* tasks.

In general, training with only 1 NVIDIA RTX 3090 GPU acheives comparable performance with the results reported in the paper. But this requires gradient accumulation to slightly enlarge the batch size, say 64.

### Sampling

We need to pass the path to the checkpoint obtained during training to `predict.py` (resp. `predict_downstream_condition.py`) to unconditionally (resp. conditionally) sample from DiffusionBERT.

The arguments of `predict_downstream_condition.py` include MBR size and time step size etc. The generation results and the MBR selected text are saved to `./generation_results`

### Evaluation

To evaluate the performance of DiffusionBERT, we can usethe functions provided in `compute_elbo.py` and `compute_metric.py`. ELBO is used to derive perplexity. Other metrics reported in the paper are included in `compute_metric.py`.



Welcome to post an issue or send me an email if there are any questions.

### Citation

```
@article{he2022diffusionbert,
  title={DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models},
  author={He, Zhengfu and Sun, Tianxiang and Wang, Kuanning and Huang, Xuanjing and Qiu, Xipeng},
  journal={arXiv preprint arXiv:2211.15029},
  year={2022}
}
```
