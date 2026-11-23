"""The network's front end: turns the observation dict into one feature vector.

Ids (a card, a monster, a relic...) mean nothing as numbers: card 212 is not
"more" than card 211. So every id column goes through an embedding, a learned
vector per name. One table per vocabulary, shared everywhere that vocabulary
appears, so "Bash" means the same thing in the hand, the deck and a shop.

Each row (one card, one enemy, one option) is then its embeddings plus its
numeric features, through a small layer to a fixed width. Slot-ordered groups
keep their rows in place, concatenated, because actions point at slots: the
logit for "play hand card 3" has to be able to see card 3. Pooled groups (the
deck, relics, powers) are sets, so they're summarised by the mean and max of
their present rows, which doesn't care about order or count.
"""
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sts2.env import GROUPS
from sts2.vocab import VOCABS

EMBED_DIM = 16


class Sts2Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, embed_dim=EMBED_DIM):
        labels = sorted({label for group in GROUPS.values() for label in group.ids})
        features = 0
        for group in GROUPS.values():
            features += group.width * (2 if group.pooled else group.rows)
        super().__init__(observation_space, features_dim=features)

        self.embeddings = nn.ModuleDict({label: nn.Embedding(VOCABS[label].size, embed_dim) for label in labels})
        self.rows = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(len(group.ids) * embed_dim + group.n_num, group.width), nn.ReLU())
            for name, group in GROUPS.items()
        })

    def forward(self, observations):
        parts = []
        for name, group in GROUPS.items():
            ids = observations[f"{name}_ids"].long()
            num = observations[f"{name}_num"]
            embedded = [self.embeddings[label](ids[..., col].clamp(0, VOCABS[label].size - 1))
                        for col, label in enumerate(group.ids)]
            rows = self.rows[name](th.cat([*embedded, num], dim=-1))  # (batch, rows, width)
            present = num[..., :1]  # column 0 is "this row is present"
            rows = rows * present  # absent rows contribute nothing
            if group.pooled:
                count = present.sum(dim=1).clamp(min=1.0)
                mean = rows.sum(dim=1) / count
                # Absent rows are 0 after the mask and ReLU outputs are >= 0, so max over all rows is safe.
                largest = rows.max(dim=1).values
                parts += [mean, largest]
            else:
                parts.append(rows.flatten(start_dim=1))
        return th.cat(parts, dim=1)
