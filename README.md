# DiffusionBERT

![](src/DiffusionBERT.gif)

Official implementation of [DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models](https://arxiv.org/abs/2211.15029).

### Update

**2023.1.25** Release the code. Please check out our paper for more details.

**2022.11.30** Initial commit.

### Abstract
We present DiffusionBERT, a new generative masked language model based on discrete diffusion models.
Diffusion models and many pre-trained language models have a shared training objective, i.e., denoising, making it possible to combine the two powerful models and enjoy the best of both worlds. 
On the one hand, diffusion models offer a promising training strategy that helps improve the generation quality.
On the other hand, pre-trained denoising language models (e.g., BERT) can be used as a good initialization that accelerates convergence.
We explore training BERT to learn the reverse process of a discrete diffusion process with an absorbing state and elucidate several designs to improve it.
First, we propose a new noise schedule for the forward diffusion process that controls the degree of noise added at each step based on the information of each token.
Second, we investigate several designs of incorporating the time step into BERT.
Experiments on unconditional text generation demonstrate that DiffusionBERT achieves significant improvement over existing diffusion models for text (e.g., D3PM and Diffusion-LM) and previous generative masked language models in terms of perplexity and BLEU score.

### Preparing the Environment & Data

We have prepared the required environment in `requirements.txt`. We use `fitlog` to monitor our training process.

```bash
conda create --name DB python=3.8
conda activate DB
pip install -r requirements.txt
```

The LM1B dataset is available at 🤗 Datasets. We use the same data for conditional generation in [DiffuSeq](https://github.com/Shark-NLP/DiffuSeq). We have prepared the loading script in `conditional_data/`. One only needs to download the `*.jsonl` files to the corresponding directories.

### Training

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