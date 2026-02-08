import torch
import torch.nn as nn
import torch.nn.functional as F

class OCNN(nn.Module):

    #One-Class Neural Network head.

    #Maps an input representation z ∈ R^{indim} to a scalar anomaly score
    #via a linear projection, an optional activation function, and a final
    #linear scoring layer.
    
    def __init__(self, indim, outdim, activation="linear"):
        super().__init__()
        # Linear projection to hidden space
        self.V = nn.Linear(indim, outdim, bias=False)
        # Final linear scoring layer
        self.w = nn.Linear(outdim, 1, bias=False)

        if activation == "linear":
            self.activation = nn.Identity()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation '{activation}'")

    def forward(self, z):
         # z: [B, indim]
        h = self.activation(self.V(z)) # [B, outdim]
        return self.w(h).squeeze(1)     # [B]

    def get_trainable(self):
         # Return trainable parameters grouped by role
        return {"w_params": [self.w.weight], "V_params": [self.V.weight]}
