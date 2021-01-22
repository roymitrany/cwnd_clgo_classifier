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
IS_DEEPCCI = False

def init_weights(model):
    if type(model) == Linear:
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)
    if type(model) == Conv2d:
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)

def train(training_loader, model, criterion, optimizer, is_deepcci):
    # enter training mode
    model.train()
    training_loss = []
    training_accuracy = []
    print('start training')
    for epoch, (data, labeling) in enumerate(training_loader):
        # use GPU
        data = data.to(device)
        labeling = labeling.to(device)
        # prediction for training set
        classification_labeling = model(data.type('torch.FloatTensor'))  # data must be a double
        # measure accuracy and record loss
        loss = criterion(classification_labeling, labeling.type('torch.LongTensor'))  # labeling must be an integer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss.append(loss)
        training_accuracy.append(accuracy(classification_labeling, labeling, topk=(1,), is_deepcci=is_deepcci))
        if epoch % 2 == 0:
            # printing the validation loss
            print('Epoch : ', epoch + 1, '\t', 'loss :', loss, '\t', 'accuracy:', training_accuracy[epoch])
    print('training is done')
    return training_loss, training_accuracy

def validate(validation_loader, model, criterion, is_deepcci):
    # enter validation mode
    print('start validating')
    model.eval()
    validation_loss = []
    validation_accuracy = []
    with torch.no_grad():
        for epoch, (data, labeling) in enumerate(validation_loader):
            # use GPU
            data = data.to(device)
            labeling = labeling.to(device)
            # prediction for validation set
            classification_labeling = model(data.type('torch.FloatTensor'))  # data must be a double
            # measure accuracy and record loss
            loss = criterion(classification_labeling, labeling.type('torch.LongTensor'))  # labeling must be an integer
            validation_loss.append(loss)
            validation_accuracy.append(accuracy(classification_labeling, labeling, topk=(1,), is_deepcci=is_deepcci))
            if epoch % 2 == 0:
                # printing the validation loss
                print('Epoch : ', epoch + 1, '\t', 'loss :', loss, '\t', 'accuracy:', validation_accuracy[epoch])
    print('validation is done')
    return validation_loss, validation_accuracy

def run(model, criterion, optimizer, scheduler, unused_parameters, is_deepcci):
    normalization_type = AbsoluteNormalization1()
    training_loader, validation_loader = create_data(training_files_path=training_files_path, normalization_type=normalization_type, unused_parameters=unused_parameters, is_deepcci=is_deepcci)
    for epoch in range(0, NUM_OF_EPOCHS):
        print('start epoch {}'.format(epoch))
        training_loss, training_accuracy = train(training_loader, model, criterion, optimizer, is_deepcci)
        validation_loss, validation_accuracy = validate(validation_loader, model, criterion, is_deepcci)
        scheduler.step()

if __name__ == '__main__':
    if IS_DEEPCCI:
        model = deepcci_net().to(device)
    else:
        model = my_net().to(device)
    model.apply(init_weights)
    criterion = CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)
    unused_parameters = ['In Throughput', 'Out Throughput', 'Connection Num of Drops', 'Num of Drops', 'Num of Packets', 'Total Bytes in Queue']
    # unused_parameters = ['timestamp', 'In Throughput', 'Out Throughput', 'Connection Num of Drops', 'Num of Drops', 'Num of Packets', 'Total Bytes in Queue']
    # unused_parameters = ['timestamp', 'In Throughput', 'Out Throughput', 'Connection Num of Drops', 'CBIQ', 'Num of Drops', 'Num of Packets', 'Total Bytes in Queue']
    run(model, criterion, optimizer, scheduler, unused_parameters, IS_DEEPCCI)
    print('done')
    # saving the trained model
    torch.save(model, training_parameters_path  + '_mytraining.pt')
    torch.save(model.state_dict(), training_parameters_path + '_mytraining_state_dict.pt')

    tn = datetime.now()
    time_str = str(tn.month) + "." + str(tn.day) + "." + str(tn.year) + "@" + str(tn.hour) + "-" + str(tn.minute) + "-" + str(tn.second)
    #plt.savefig(training_parameters_path + normalization_types[normalization_counter] + '_graph.jpeg')