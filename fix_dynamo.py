import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 1000
