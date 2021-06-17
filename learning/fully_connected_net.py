from learning.utils import *
import math

class fully_connected_net(Module):
    def __init__(self):
        super(fully_connected_net, self).__init__()
        # D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        self.D_in, self.H, self.D_out = CHUNK_SIZE, NUM_OF_HIDDEN_LAYERS, NUM_OF_CONGESTION_CONTROL_LABELING
        self.conv2d_layer = Conv2d(1, NUM_OF_CONV_FILTERS, kernel_size=(3, NUM_OF_CLASSIFICATION_PARAMETERS), stride=1, padding=(1, 0))
        self.linear_layers = Sequential(
            Linear(self.D_in, self.H),
            ReLU(),
            Linear(self.H, self.D_out),
        )
        self.conv1d_layer = torch.nn.Conv1d(NUM_OF_CONV_FILTERS, 1, kernel_size=3, padding=1)


    # Defining the forward pass
    def forward(self, x):
        x = self.conv2d_layer(x)
        x = x.mean(axis=3)
        x = self.linear_layers(x)
        x = self.conv1d_layer(x)
        x = x.squeeze(1)
        return x
