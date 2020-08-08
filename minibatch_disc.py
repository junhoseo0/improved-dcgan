import torch
import torch.nn as nn

# In the original paper, there's a very little information about
# how to choose the dimension of tensor, or initialize value of it.
# Also, if you look at the official code, there's some crazy theta
# -weight scale thing going on, which are never mentioned in the paper.
# Therefore, I'll just ignore the official code and implement mini
# batch discrimination as-is described in the paer.
class MinibatchDisc(nn.Module):
    def __init__(self, in_features, out_features, inter_features):
        super(MinibatchDisc, self).__init__()

        if out_features <= in_features:
            raise Exception("out_features must be greater than in_features")

        self.T = nn.Parameter(torch.randn(
            in_features,
            out_features-in_features,
            inter_features,
        ))

    # As the output dimension is heavily modified,
    # I do not expect that minibatch output would used
    # in between convolutional layers. Instead, I'll use
    # it before last linear (FC) layer.
    def forward(self, x):
        # (N, InCh, Dim, Dim)->(N, InCh*Dim*Dim = InF)
        flat = torch.flatten(x, start_dim=1)
        # (N, InF)*(N, InF, Out, Inter)->(N, Out, Inter)
        # There's a performance concern about torch.einsum
        mat = torch.einsum("ni,ijk->njk", flat, self.T)
        # Tricks using broadcasting:
        # (N, Out, Inter)->(N, N, Out, Inter)
        mat_diff = torch.abs(mat.unsqueeze(dim=1) - mat)
        # (N, N, Out, Inter)->(N, N, Out), L1-norm
        row_diff = torch.sum(mat_diff, dim=3)
        diff_exp = torch.exp(-row_diff) # c(x_i)_b
        # (N, N, Out)->(N, Out), Sum of exp's
        o = torch.sum(diff_exp, dim=1) # o(x_i), o(x_i)_b
        # (N, InF),(N, Out)->(N, InF+Out)
        return torch.cat((flat, o), dim=1)

if __name__ == "__main__":
    x = torch.randn(64, 2*2*1024)
    y = MinibatchDisc(2*2*1024, 5*1024, 10)(x)
    assert y.shape == torch.Size([64, 5*1024])
