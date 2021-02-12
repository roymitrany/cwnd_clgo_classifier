from learning.utils import *

class my_net(Module):
    def __init__(self):
        super(my_net, self).__init__()
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

        self.conv2d_layer1 = Conv2d(1, NUM_OF_CONV_FILTERS, kernel_size=(3, NUM_OF_CLASSIFICATION_PARAMETERS), stride=1, padding=(1, 0)) # NUM_OF_CLASSIFICATION_PARAMETERS
        self.BN_layer1 = BatchNorm2d(NUM_OF_CONV_FILTERS)
        self.Relu_layer1 = ReLU(inplace=True)
        self.maxpool_layer1 = MaxPool2d(kernel_size=(6, 1), stride=6)
        self.conv2d_layer2 = Conv2d(NUM_OF_CONV_FILTERS, NUM_OF_CONV_FILTERS, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.BN_layer2 = BatchNorm2d(NUM_OF_CONV_FILTERS)
        self.Relu_layer2 = ReLU(inplace=True)
        self.maxpool_layer2 = MaxPool2d(kernel_size=(10, 1), stride=10)
        self.conv2d_layer3 = Conv2d(NUM_OF_CONV_FILTERS, NUM_OF_CONV_FILTERS, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.BN_layer3 = BatchNorm2d(NUM_OF_CONV_FILTERS)
        self.Relu_layer3 = ReLU(inplace=True)
        self.maxpool_layer3 = MaxPool2d(kernel_size=(10, 1), stride=10)
        self.conv2d_layer4 = Conv2d(NUM_OF_CONV_FILTERS, NUM_OF_CONV_FILTERS, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.BN_layer4 = BatchNorm2d(NUM_OF_CONV_FILTERS)
        self.Relu_layer4 = ReLU(inplace=True)
        self.maxpool_layer4 = MaxPool2d(kernel_size=(10, 1), stride=10)
        self.conv2d_layer5 = Conv2d(NUM_OF_CONV_FILTERS, NUM_OF_CONV_FILTERS, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.BN_layer5 = BatchNorm2d(NUM_OF_CONV_FILTERS)
        self.Relu_layer5 = ReLU(inplace=True)
        self.maxpool_layer5 = MaxPool2d(kernel_size=(10, 1), stride=10)
        self.conv1d_layer5 = torch.nn.Conv1d(NUM_OF_CONV_FILTERS, NUM_OF_HIDDEN_LAYERS, kernel_size=3, padding=1)
        self.gru = torch.nn.GRU(input_size=NUM_OF_HIDDEN_LAYERS, hidden_size=NUM_OF_HIDDEN_LAYERS, num_layers=2)
        self.lstm = torch.nn.LSTM(input_size=NUM_OF_HIDDEN_LAYERS, num_layers=1, hidden_size=NUM_OF_HIDDEN_LAYERS)
        self.BN_final = torch.nn.BatchNorm1d(NUM_OF_HIDDEN_LAYERS)
        self.conv1d_final = torch.nn.Conv1d(NUM_OF_HIDDEN_LAYERS, NUM_OF_CONGESTION_CONTROL_LABELING, kernel_size=3, padding=1)
        self.linear = Linear(NUM_OF_CONV_FILTERS * 1 * (NUM_OF_TIME_SAMPLES - 2) * 1, NUM_OF_CONGESTION_CONTROL_LABELING)

    # Defining the forward pass
    def forward(self, x):
        """
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)  # flattening?
        x = self.linear_layers(x)
        return x
        """
        x = self.conv2d_layer1(x)
        x = self.BN_layer1(x)
        x = self.Relu_layer1(x)
        x = self.maxpool_layer1(x)
        x = self.conv2d_layer2(x)
        x = self.BN_layer2(x)
        x = self.Relu_layer2(x)
        x = self.maxpool_layer2(x)
        x = self.conv2d_layer3(x)
        x = self.BN_layer3(x)
        x = self.Relu_layer3(x)
        x = self.maxpool_layer3(x)
        x = self.conv2d_layer4(x)
        x = self.BN_layer4(x)
        x = self.Relu_layer4(x)
        x = self.maxpool_layer4(x)
        x = self.conv2d_layer5(x)
        x = self.BN_layer5(x)
        x = self.Relu_layer5(x)
        x = self.maxpool_layer5(x)

        x = x.squeeze(2)
        x = self.conv1d_layer5(x)

        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        x, _ = self.gru(x)
        #x, _ = self.lstm(x)
        x = x.transpose(0, 1)
        x = x.transpose(1, 2)
        x = self.BN_final(x)

        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        x, _ = self.gru(x)
        #x, _ = self.lstm(x)
        x = x.transpose(0, 1)
        x = x.transpose(1, 2)
        x = self.BN_final(x)

        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        x, _ = self.gru(x)
        #x, _ = self.lstm(x)
        x = x.transpose(0, 1)
        x = x.transpose(1, 2)
        x = self.BN_final(x)

        x = self.conv1d_final(x)
        x = x.squeeze(2)
        return x
