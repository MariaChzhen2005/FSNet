from torch import nn

# Neural Network Models
# Multi-Layer Perceptron (MLP) with adjustable dropout and layer count  

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]

        for i in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout/(i+1))) # Adjust dropout rate for each layer
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid()) # Sigmoid activation for output layer
        self.mlp = nn.Sequential(*layers)
        # initialize weights if necessary               
    def forward(self, x):
        return self.mlp(x)