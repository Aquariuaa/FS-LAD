import pytorch_lightning as L
import torch
import numpy as np
import torch.nn as nn
adaptive_filter=True



def generate_binomial_mask(B, T, p=0.98):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class PatchEmbed(L.LightningModule):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


class ICB(L.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        embedding_dim = 300
        self.patch_embed = PatchEmbed(
            seq_len=300, patch_size=10,
            in_chans=300, embed_dim=embedding_dim
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embedding_dim), requires_grad=True)

        self.input_fc = nn.Linear(in_features, in_features)
        padding1 = (1 * (3 - 1)) // 2
        padding2 = (2 * (3 - 1)) // 2
        in_features_patch = 59
        self.dconv1 = nn.Conv1d(in_features_patch, hidden_features, 3,1, padding=padding1, dilation=1)
        self.dconv2 = nn.Conv1d(hidden_features, hidden_features, 3,1, padding=padding2, dilation=2)

        self.conv1 = nn.Conv1d(in_features_patch, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.patch_embed(x)

        # Position embedding
        x = x + self.pos_embed

        # Event Masking
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        mask &= nan_mask
        x[~mask] = 0

        # Dilation Convolution
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)
        x2 = self.dconv1(x)
        x2 = self.dconv2(x2)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        # Interactive Module
        out1 = x1 * x2_2
        out2 = x2 * x1_2
        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        out_f = x.flatten(start_dim=1)
        return out_f


def MITNet(flatten):
    in_features = 300
    hidden_features = 32
    drop = 0.1
    model = ICB(in_features, hidden_features, drop)
    return model


if __name__ == '__main__':
    in_features = 300
    hidden_features = 32
    drop = 0.1
    model = ICB(in_features, hidden_features, drop)
    input= torch.randn(64, 300, in_features)
    output = model(input)
    print("input.shape:", input.shape)
    print("output.shape:", output.shape)


