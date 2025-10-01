import torch as t
import numpy as np
class Config:
    def __init__(self):
        self.prefill_size = 480
        self.max_seq_length = 512
        self.hidden_size = 16


class NaiveModel(t.nn.Module):
    def __init__(self, cfg:Config):
        super(NaiveModel, self).__init__()
        self.cfg = cfg
        self.linear = t.nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x):
        return self.linear(x)

def run_navie():
    cfg=Config()  # Initialize configuration
    data=np.random.random(size=(cfg.prefill_size, cfg.hidden_size))  # Create random input data
    t.variable(t.tensor(data, dtype=t.float32))  # Convert to PyTorch tensor
    mask=t.ones(cfg.prefill_size)  # Create attention mask
    print(data.shape, mask.shape)
    y=t.zeros(cfg.max_seq_length, cfg.hidden_size)  # Placeholder for output
    # y.copy_(data[:cfg.max_seq_length])  # Copy input data to output
    
    pass

if __name__ == "__main__":
    run_navie()