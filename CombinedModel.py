import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self, class_model, seq_model):
        super(CombinedModel, self).__init__()
        self.class_model = class_model
        self.seq_model = seq_model
    
    def forward(self, x):
        class_output = self.class_model(x)
        seq_output = self.seq_model(x)
        return class_output, seq_output