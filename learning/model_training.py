# importing the libraries
import pickle
from datetime import datetime
# for evaluating the model
from sklearn.metrics import accuracy_score
# for creating validation set
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d
# importing project functions
from classification.utils import *

# consts definitions
NUM_OF_CLASSIFICATION_PARAMETERS = 2 # timestemp & CBIQ
NUM_OF_TIME_SAMPLES = 60000
NUM_OF_CONGESTION_CONTROL_LABELING = 3 # Reno, Cubic, & BBR
NUM_OF_CONV_FILTERS = 50
NUM_OF_EPOCHS = 100

def init_weights(model):
    if type(model) == Linear:
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)
    if type(model) == Conv2d:
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)

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

class deepcci_net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.channels = 64
        self.num_layers = 100
        self.hidden_size = 64
        # 1
        self.conv1d_layer1_1 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer1_1 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_1 = nn.ReLU(inplace=False)
        self.conv1d_layer2_1 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer2_1 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_1 = nn.ReLU(inplace=False)
        self.maxpool_layer_1 = nn.MaxPool1d(kernel_size=4, stride=4, ceil_mode=False)
        # 2
        self.conv1d_layer1_2 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer1_2 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_2 = nn.ReLU(inplace=False)
        self.conv1d_layer2_2 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer2_2 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_2 = nn.ReLU(inplace=False)
        self.maxpool_layer_2 = nn.MaxPool1d(kernel_size=4, stride=4, ceil_mode=False)
        # 3
        self.conv1d_layer1_3 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer1_3 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_3 = nn.ReLU(inplace=False)
        self.conv1d_layer2_3 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer2_3 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_3 = nn.ReLU(inplace=False)
        self.maxpool_layer_3 = nn.MaxPool1d(kernel_size=4, stride=4, ceil_mode=False)
        # 4
        self.conv1d_layer1_4 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer1_4 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_4 = nn.ReLU(inplace=False)
        self.conv1d_layer2_4 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer2_4 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_4 = nn.ReLU(inplace=False)
        self.maxpool_layer_4 = nn.MaxPool1d(kernel_size=4, stride=4, ceil_mode=False)
        # 5
        self.conv1d_layer1_5 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer1_5 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_5 = nn.ReLU(inplace=False)
        self.conv1d_layer2_5 = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1)
        self.BN_layer2_5 = nn.BatchNorm1d(self.channels)
        self.Relu_layer_5 = nn.ReLU(inplace=False)
        self.maxpool_layer_5 = nn.MaxPool1d(kernel_size=4, stride=4, ceil_mode=False)

        # 1
        self.BN_layer_1 = nn.BatchNorm1d(100)
        self.lstm_layer_1 = nn.LSTM(input_size=self.channels, num_layers=1, hidden_size=100)
        # 2
        self.BN_layer_2 = nn.BatchNorm1d(100)
        self.lstm_layer_2 = nn.LSTM(input_size=100, num_layers=1, hidden_size=100)
        # 3
        self.BN_layer_3 = nn.BatchNorm1d(100)
        self.lstm_layer_3 = nn.LSTM(input_size=100, num_layers=1, hidden_size=100)

        self.init_fg_bias()

        self.conv1 = nn.Conv1d(1, self.channels, kernel_size=1)

        self.first_BN_layer = (nn.BatchNorm1d(self.channels))
        self.last_conv1d_layer = nn.Conv1d(100, 5, kernel_size=1, padding=1)
        self.last_BN_layer = nn.BatchNorm1d(5)

    def init_fg_bias(self):
        for names in self.lstm_layer_1._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm_layer_1, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)
        for names in self.lstm_layer_2._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm_layer_2, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)
        for names in self.lstm_layer_3._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm_layer_3, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    # Defining the forward pass
    def forward(self, x):
        # x = torch.zeros([180, 1, 60000], dtype=torch.int32)
        # x=x.type('torch.FloatTensor')

        x = self.conv1(x)
        # 1
        residual = x
        x = self.conv1d_layer1_1(x)
        x = self.BN_layer1_1(x)
        x = self.Relu_layer_1(x)
        x = self.conv1d_layer2_1(x)
        x = self.BN_layer2_1(x)
        x = x + residual
        x = self.Relu_layer_1(x)
        x = self.maxpool_layer_1(x)
        # 2
        residual = x
        x = self.conv1d_layer1_2(x)
        x = self.BN_layer1_2(x)
        x = self.Relu_layer_2(x)
        x = self.conv1d_layer2_2(x)
        x = self.BN_layer2_2(x)
        x = x + residual
        x = self.Relu_layer_2(x)
        x = self.maxpool_layer_2(x)
        # 3
        residual = x
        x = self.conv1d_layer1_3(x)
        x = self.BN_layer1_3(x)
        x = self.Relu_layer_3(x)
        x = self.conv1d_layer2_3(x)
        x = self.BN_layer2_3(x)
        x = x + residual
        x = self.Relu_layer_3(x)
        x = self.maxpool_layer_3(x)
        # 4
        residual = x
        x = self.conv1d_layer1_4(x)
        x = self.BN_layer1_4(x)
        x = self.Relu_layer_4(x)
        x = self.conv1d_layer2_4(x)
        x = self.BN_layer2_4(x)
        x = x + residual
        x = self.Relu_layer_4(x)
        x = self.maxpool_layer_4(x)
        # 5
        residual = x
        x = self.conv1d_layer1_5(x)
        x = self.BN_layer1_5(x)
        x = self.Relu_layer_5(x)
        x = self.conv1d_layer2_5(x)
        x = self.BN_layer2_5(x)
        x = x + residual
        x = self.Relu_layer_5(x)
        x = self.maxpool_layer_5(x)

        x = self.first_BN_layer(x)
        # 1
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        x, _ = self.lstm_layer_1(x)
        x = x.transpose(0, 1)
        x = x.transpose(1, 2)
        x = self.BN_layer_1(x)
        # 2
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        x, _ = self.lstm_layer_2(x)
        x = x.transpose(0, 1)
        x = x.transpose(1, 2)
        x = self.BN_layer_2(x)
        # 3
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        x, _ = self.lstm_layer_3(x)
        x = x.transpose(0, 1)
        x = x.transpose(1, 2)
        x = self.BN_layer_3(x)

        x = self.last_conv1d_layer(x)
        x = self.last_BN_layer(x)

        x_variant = x[:, 0:3, :]
        x_pacing = x[:, 3:5, :]

        # pacing_output = self.pacing_softmax_layer(x_pacing)
        # variant_output = self.variant_softmax_layer(x_variant)

        pacing_output = x_pacing
        variant_output = x_variant

        return variant_output, pacing_output
        return x


def train(epoch):
    model.train()
    tr_loss = 0
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    # getting the validation set
    x_val, y_val = Variable(val_x), Variable(val_y)
    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    # prediction for training and validation set
    output_train = model(x_train.type('torch.FloatTensor'))
    output_val = model(x_val.type('torch.FloatTensor'))
    # computing the training and validation loss
    loss_train = criterion(output_train, y_train.type('torch.LongTensor'))  # Long instead of float (was float and changed to long- now works).
    loss_val = criterion(output_val, y_val.type('torch.LongTensor'))
    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch % 2 == 0:
        # printing the validation loss
        print('Epoch : ', epoch + 1, '\t', 'loss :', loss_val)
    training_accuracy = accuracy_single_sample(output_train, train_y, topk=(1, ))
    return loss_val, training_accuracy


def train(train_loader, model, variant_criterion, pacing_criterion, optimizer, epoch):
    # switch to train mode
    model.train()
    print('start training')
    for i, (data_input, expected_variant) in enumerate(train_loader):
        data_input = data_input.to(device)
        expected_variant = expected_variant.to(device)
        acc_variant_expected = expected_variant.clone()
        variant_output = model(data_input)
        expected_variant = expected_variant.long()
        variant_loss = variant_criterion(variant_output, expected_variant)
        acc_variant_output = variant_output.clone()

        # measure accuracy and record loss
        variant_acc1, variant_acc5 = accuracy(acc_variant_output, acc_variant_expected, topk=(1, 3))

        accuracy_per_epoch(acc_variant_output, acc_variant_expected, acc_pacing_output, acc_pacing_expected,
                           f1_calculator, epoch)
        accuracy_per_epoch(acc_variant_output, acc_variant_expected, acc_pacing_output, acc_pacing_expected,
                           f1_train_calculator, epoch)

        variant_losses.update(variant_loss.item(), data_input.size(0))

        variant_top1.update(variant_acc1[0], data_input.size(0))
        variant_top5.update(variant_acc5[0], data_input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % PRINT_EVERY == 0 and i > 0:
            progress.display(i)

    progress.display(len(train_loader))
    print('training is done')
    return variant_top1.avg, pacing_top1.avg, variant_losses.avg, pacing_losses.avg


def run(net, criterion, optimizer, scheduler):
    normalization_type = AbsoluteNormalization1()
    train_loader, val_loader = create_data(training_files_path, normalization_type, is_deepcci=false)
    for epoch in range(0, NUM_OF_EPOCHS):
        print('start epoch {}'.format(epoch))
        train_accuracy, train_loss = train(train_loader, net, criterion, optimizer)
        val_accuracy, val_loss = validate(val_loader, net, criterion)
        scheduler.step()
        print("The total acc for epoch {} is {}".format(epoch, total_acc))

if __name__ == '__main__':
    global model, val_x, val_y, optimizer, criterion, n_epochs, train_losses, val_losses
    # defining the model
    model = my_net()
    # initializing weights:
    model.apply(init_weights)
    # defining the loss function:
    criterion = CrossEntropyLoss()
    # criterion = L1Loss()
    # criterion = NLLLoss()
    # criterion = MSELoss()
    # criterion = SmoothL1Loss()
    # criterion = KLDivLoss()
    # empty list to store training losses
    train_losses = []
    # empty list to store validation losses
    val_losses = []
    # training the model:
    m = 25
    learning_rate_init = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, threshold=0.01, threshold_mode='abs')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)

    train_loader = create_dataloader(input_data, input_labeling)
    training_accuracy = []
    for i, (data_input, expected_variant) in enumerate(train_loader):
        for epoch in range(NUM_OF_EPOCHS):
            # learning_rate = learning_rate_init / (1 + epoch/m)
            # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            input_data = data_input
            input_labeling = expected_variant
            loss_val, accuracy_training = train(epoch)
            training_accuracy.append(accuracy_training)
            # scheduler.step(loss_val)
            scheduler.step()

    for epoch in range(NUM_OF_EPOCHS):
        #learning_rate = learning_rate_init / (1 + epoch/m)
        #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        loss_val, training_accuracy = train(epoch)
        #scheduler.step(loss_val)
        scheduler.step()
        # if epoch > min_n_epochs and val_losses[epoch] == 0:
        #     break

    # saving the trained model
    torch.save(model, training_parameters_path + normalization_types[normalization_counter] + '_mytraining.pt')
    torch.save(model.state_dict(), training_parameters_path + normalization_types[normalization_counter] + '_mytraining_state_dict.pt')

    tn = datetime.now()
    time_str = str(tn.month) + "." + str(tn.day) + "." + str(tn.year) + "@" + str(tn.hour) + "-" + str(tn.minute) + "-" + str(tn.second)
    #plt.savefig(training_parameters_path + normalization_types[normalization_counter] + '_graph.jpeg')

    # prediction for training set
    with torch.no_grad():
        output = model(train_x.type('torch.FloatTensor'))

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)

    # accuracy on training set
    print(accuracy_score(train_y, predictions))

    # prediction for validation set
    with torch.no_grad():
        output = model(val_x.type('torch.FloatTensor'))

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)

    # accuracy on validation set
    print(accuracy_score(val_y, predictions))
    print(training_accuracy)
    normalization_counter +=1