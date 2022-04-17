import torch
from torch import nn

# We want to be able to cmopute the steady state concentration of each PRT and each RNA
# steady state of RNA: Qr = SUM(N * Tc) / Rdeg
# steady state of PRT: Qp = SUM(Qr * Tl) / Rdeg
# we need to produce a computational graph, using pytorc

class ERNSequestron(nn.Module):
  def __init__(self, in_neg, in_pos, n_out=1):
    super().__init__()
    self.negweight = nn.Parameter(torch.randn(in_neg, n_out))
    self.posweight = nn.Parameter(torch.randn(in_neg, n_out))

  def forward(self, neg_input, pos_input):
    return max(0,(pos_input @ self.weight) - (neg_input @ self.weight))


# n_in inputs [dna|rna] -> n_out outputs [rna|prt] ; fixed degradation rate
class SimpleForward(nn.Module):
  def __init__(self, n_in, n_out, deg_rate = 1.0):
    super().__init__()
    self.deg_rate = deg_rate
    self.weight = nn.Parameter(torch.randn(n_in,n_out))

  def forward(self, x):
    return max(0,(x @ self.weight) / self.deg_ate)

