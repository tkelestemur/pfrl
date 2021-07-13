import torch
from torch import nn
from torch.distributions import Categorical


class SoftmaxCategoricalHead(nn.Module):
    def forward(self, logits):
        return torch.distributions.Categorical(logits=logits)


# class MultiSoftmaxCategoricalHead(nn.Module):
#     def forward(self, logits):
#         return Independent(Categorical(logits=logits), reinterpreted_batch_ndims=1)


class MultiCategorical():
    def __init__(self, dims=None, logits=None):
        self.dims = dims
        logits = torch.split(logits, tuple(dims), dim=1)
        self.dists = [Categorical(logits=logits_dim) for logits_dim in logits]

    def log_prob(self, actions):
        actions = torch.unbind(actions, dim=1)
        logprobs = torch.stack([
            dist.log_prob(action) for dist, action in zip(self.dists, actions)
        ], dim=1)
        return logprobs.sum(dim=1)

    def entropy(self):
        return torch.stack([dist.entropy() for dist in self.dists], dim=1).sum(dim=1)

    def sample(self):
        return torch.stack([dist.sample() for dist in self.dists], dim=1)

    def mode(self):
        return torch.stack([
            torch.argmax(dist.probs, dim=1) for dist in self.dists
        ], dim=1)


class MultiSoftmaxCategoricalHead(nn.Module):
    def __init__(self, dims=None):
        self.dims = dims
        super().__init__()

    def forward(self, logits):
        return MultiCategorical(dims=self.dims, logits=logits)
