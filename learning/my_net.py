from learning.utils import *

class my_net(Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv2d: output size = image_size - filter_size / stride + 1. usually stride = 1, filter [x,x], and padding = (f-1)/2
        # maxpool2d: output size = image_size - filter_size
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, NUM_OF_CONV_FILTERS, kernel_size=(3, NUM_OF_CLASSIFICATION_PARAMETERS), stride=1, padding=(1, 0)), # NUM_OF_CLASSIFICATION_PARAMETERS
            # channels: 1, filters: 10.
            BatchNorm2d(NUM_OF_CONV_FILTERS),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=(2, 1), stride=1),
            # Defining another 2D convolution layer
            Conv2d(NUM_OF_CONV_FILTERS, NUM_OF_CONV_FILTERS, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BatchNorm2d(NUM_OF_CONV_FILTERS),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=(2, 1), stride=1),
        )

        self.linear_layers = Sequential(
            Linear(NUM_OF_CONV_FILTERS * 1 * (NUM_OF_TIME_SAMPLES - 2) * 1, NUM_OF_CONGESTION_CONTROL_LABELING)  # 1 instead of NUM_OF_CLASSIFICATION_PARAMETER
            # an error because the labels must be 0 indexed. So, for example, if you have 20 classes, and the labels are 1th indexed, the 20th label would be 20, so cur_target < n_classes assert would fail. If it’s 0th indexed, the 20th label is 19, so cur_target < n_classes assert passes.
            # input features: 10 channels * number of rows * number of columns, output features: number of labels = 2.
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)  # flattening?
        x = self.linear_layers(x)
        return x
