import torch

a = torch.tensor([1.0, float("nan"), 2.0, float("inf"), -float("inf")])

if torch.isnan(a).any():
    print("a is nan")
if torch.isinf(a).any():
    print("a is inf")
else:
    print("a is not nan")
