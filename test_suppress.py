from unsloth import FastLanguageModel
import torch._dynamo
# Set after unsloth import!
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.recompile_limit = 10000

print("suppress_errors:", torch._dynamo.config.suppress_errors)
print("recompile_limit:", torch._dynamo.config.recompile_limit)
