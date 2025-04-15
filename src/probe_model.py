import torch.nn as nn

hs_dict = {
        "DeepSeek-R1-Distill-Qwen-32B": 5120,
        "DeepSeek-R1-Distill-Qwen-1.5B": 1536,
        "DeepSeek-R1-Distill-Qwen-7B": 3584,
        "DeepSeek-R1-Distill-Llama-8B": 4096,
        "DeepSeek-R1-Distill-Llama-70B": 8192,
        "QwQ-32B": 5120
    }

class MLPProbe(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(MLPProbe, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)  # No Sigmoid here
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)  # Raw logits
        return x

class LinearProbe(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearProbe, self).__init__()
        self.output = nn.Linear(input_size, output_size)  # No Sigmoid here

    def forward(self, x):
        x = self.output(x)  # Raw logits
        return x

def load_model(input_size, hidden_size, output_size, ckpt_weight=None):
    if hidden_size==0:
        model = LinearProbe(input_size, output_size)
    else:
        model = MLPProbe(input_size, hidden_size, output_size)
    if ckpt_weight is not None:
        model.load_state_dict(ckpt_weight)
    return model

