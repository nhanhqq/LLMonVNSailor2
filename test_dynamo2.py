import torch._dynamo
torch._dynamo.config.disable = True
import torch

@torch.compile(fullgraph=True)
def f(x):
    return x + 1

try:
    print(f(torch.tensor([1, 2])))
    print("Dynamo disable worked!")
except Exception as e:
    print("Dynamo disable failed:", e)

