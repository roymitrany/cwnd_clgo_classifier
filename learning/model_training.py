# importing the libraries
import pickle
from datetime import datetime
# for evaluating the model
from sklearn.metrics import accuracy_score
# for creating validation set
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d
# importing project functions
from learning.utils import *
from learning.my_net import *
from learning.deepcci_net import *

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

def train(train_loader, model, criterion, optimizer):
    # enter train mode
    model.train()
    training_loss = []
    training_accuracy = []
    print('start training')
    for epoch, (data, labeling) in enumerate(train_loader):
        # use GPU
        data = data.to(device)
        #data = Variable(data)
        labeling = labeling.to(device)
        #labeling = Variable(labeling)
        # prediction for training set
        classification_labeling = model(data)
        #classification_labeling = model(data.type('torch.FloatTensor'))
        # measure accuracy and record loss
        loss = criterion(classification_labeling, labeling)
        #loss = criterion(classification_labeling, training_labeling.type('torch.LongTensor'))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss.append(loss)
        training_accuracy.append(accuracy_single_sample(classification_labeling, labeling, topk=(1,)))
        if epoch % 2 == 0:
            # printing the validation loss
            print('Epoch : ', epoch + 1, '\t', 'loss :', loss, '\t', 'accuracy:', training_accuracy[epoch])
    print('training is done')
    return training_loss, training_accuracy

def run(model, criterion, optimizer, scheduler):
    normalization_type = AbsoluteNormalization1()
    train_loader, val_loader = create_data(training_files_path, normalization_type, is_deepcci=False)
    for epoch in range(0, NUM_OF_EPOCHS):
        print('start epoch {}'.format(epoch))
        train_loss, train_accuracy = train(train_loader, model, criterion, optimizer)
        #val_accuracy, val_loss = validate(val_loader, net, criterion)
        scheduler.step()

if __name__ == '__main__':
    model = my_net().to(device)
    criterion = CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)
    run(model, criterion, optimizer, scheduler)
    print('done')
    """
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
    """