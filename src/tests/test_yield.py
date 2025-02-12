"""
yield 太伟大了, 这样就不用 再重新写一个组织 forward 的结构了
"""

import torch
from torch import Tensor


def forward(x: Tensor):
    x = x * 2
    y = x * 3
    print(f"{y=}")
    x = yield x
    z = x + y
    return z


if __name__ == "__main__":
    gen = forward(torch.tensor([1.0]))
    print(next(gen))
    try:
        gen.send(torch.tensor([5.0]))
    except StopIteration as e:
        print(e.value)
