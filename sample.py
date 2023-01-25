import torch
import abc


class SampleClassBase(abc.ABC):
    def sample(self, logits, x_0):
        raise NotImplementedError

    def post_process_sample_in_prediction(self, sample, x_0):
        return sample


class Categorical(SampleClassBase):
    def sample(self, logits, x_0):
        return torch.distributions.categorical.Categorical(logits=logits).sample()


class WholeWordMasking(SampleClassBase):
    def __init__(self, tokenizer):
        self.dim = tokenizer.vocab_size
        self.mask_id = tokenizer.mask_token_id
        self.post_tokens = torch.zeros(size=(tokenizer.vocab_size,), device='cuda:0', dtype=torch.long)
        for token, id in tokenizer.vocab.items():
            if token.startswith('##'):
                self.post_tokens[id] = 1

    def sample(self, logits, x_0):
        is_post = (self.post_tokens * x_0).sum(-1).nonzero()
        samp = torch.distributions.categorical.Categorical(logits=logits).sample()
        for index in is_post:
            samp[index[0], index[1]] = self.mask_id if samp[index[0], index[1] - 1] == self.mask_id else x_0[index[0], index[1]].argmax()
        return samp

    def post_process_sample_in_prediction(self, sample, x_0):
        x_0 = torch.nn.functional.one_hot(x_0, num_classes=self.dim)
        is_post = (self.post_tokens * x_0).sum(-1).nonzero()
        for index in is_post:
            sample[index[0], index[1]] = self.mask_id if sample[index[0], index[1] - 1] == self.mask_id else x_0[index[0], index[1]].argmax()
        return sample



