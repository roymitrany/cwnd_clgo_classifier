#!/usr/bin/env python
# importing the libraries
import pickle
import threading
from time import sleep
from datetime import datetime
# for evaluating the model
# from sklearn.metrics import accuracy_score
# for creating validation set
# from torch.autograd import Variable
# from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d
# importing project functions
# from learning.utils import *

from learning.my_net import *
from learning.deepcci_net import *
from learning.fully_connected_net import *
# from learning.env import *
# import learning.env

NUM_OF_EPOCHS = 100
NUM_OF_BATCHES = 10
BATCH_SIZE = 32
IS_BATCH = True

def init_weights(model):
    if type(model) == Linear:
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)
    if type(model) == Conv2d:
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)

def train(training_loader, model, criterion, optimizer, is_deepcci, device):
    # enter training mode
    model.train()
    training_loss = []
    training_accuracy = []
    training_accuracy_per_type = []
    print('start training')
    for epoch, (data, labeling) in enumerate(training_loader):
        # use GPU
        data = data.to(device)
        labeling = labeling.to(device)
        # prediction for training set
        if device == torch.device("cuda"):
            classification_labeling = model(data.type('torch.cuda.FloatTensor'))  # data must be a double
            # measure accuracy and record loss
            loss = criterion(classification_labeling,
                             labeling.type('torch.cuda.LongTensor'))  # labeling must be an integer
        else:
            classification_labeling = model(data.type('torch.FloatTensor'))  # data must be a double
            # measure accuracy and record loss
            loss = criterion(classification_labeling, labeling.type('torch.LongTensor'))  # labeling must be an integer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss.append(loss.item())
        training_accuracy.append(accuracy(classification_labeling, labeling, topk=(1,), is_deepcci=is_deepcci).item())
        if not is_deepcci:
            training_accuracy_per_type.append(accuracy_per_type(classification_labeling, labeling))
        #if epoch % 2 == 0:
            # printing the validation loss
            #print('Iteration : ', epoch + 1, '\t', 'loss :', training_loss[epoch], '\t', 'accuracy:', training_accuracy[epoch])
    print('training is done')
    return training_loss, training_accuracy, training_accuracy_per_type

def validate(validation_loader, model, criterion, is_deepcci, device):
    # enter validation mode
    print('start validating')
    model.eval()
    validation_loss = []
    validation_accuracy = []
    validation_accuracy_per_type = []
    with torch.no_grad():
        for epoch, (data, labeling) in enumerate(validation_loader):
            # use GPU
            data = data.to(device)
            labeling = labeling.to(device)
            # prediction for validation set
            if device == torch.device("cuda"):
                classification_labeling = model(data.type('torch.cuda.FloatTensor'))  # data must be a double
                # measure accuracy and record loss
                loss = criterion(classification_labeling, labeling.type('torch.cuda.LongTensor'))  # labeling must be an integer
            else:
                classification_labeling = model(data.type('torch.FloatTensor'))  # data must be a double
                # measure accuracy and record loss
                loss = criterion(classification_labeling, labeling.type('torch.LongTensor'))  # labeling must be an integer
            validation_loss.append(loss.item())
            validation_accuracy.append(accuracy(classification_labeling, labeling, topk=(1,), is_deepcci=is_deepcci).item())
            if not is_deepcci:
                validation_accuracy_per_type.append(accuracy_per_type(classification_labeling, labeling))
            #if epoch % 2 == 0:
                # printing the validation loss
                #print('Iteration : ', epoch + 1, '\t', 'loss :', validation_loss[epoch], '\t', 'accuracy:', validation_accuracy[epoch])
    print('validation is done')
    return validation_loss, validation_accuracy, validation_accuracy_per_type

def run(model, criterion, optimizer, scheduler, unused_parameters, is_deepcci, results_path, is_batch, plot_file_name, is_sample_rate, training_files_path, bg_flows, is_sample, diverse_training_folder, num_of_time_samples, chunk_size, is_diverse, num_of_classification_parameters, device, deepcci_num_of_time_samples):
    normalization_type = AbsoluteNormalization1()
    training_loader, validation_loader = create_data(training_files_path=training_files_path, results_path=results_path, normalization_type=normalization_type, unused_parameters=unused_parameters, is_deepcci=is_deepcci, is_batch=is_batch,
                                                     diverse_training_folder=diverse_training_folder, is_sample_rate=is_sample_rate, bg_flows=bg_flows, is_sample=is_sample, num_of_time_samples=num_of_time_samples, chunk_size=chunk_size, is_diverse=is_diverse, num_of_classification_parameters=num_of_classification_parameters, deepcci_num_of_time_samples=deepcci_num_of_time_samples)
    training_loss, training_accuracy, validation_loss, validation_accuracy = ([None] * NUM_OF_EPOCHS for i in range(4))
    training_accuracy_per_type, validation_accuracy_per_type = ([None] * NUM_OF_EPOCHS for i in range(2))
    f_graph = open(plot_file_name, "w+")
    f_graph.write('epoch, training_loss, training_accuracy, validation_loss, validation_accuracy\n')
    for epoch in range(0, NUM_OF_EPOCHS):
        print('start epoch {}'.format(epoch))
        training_loss[epoch], training_accuracy[epoch], training_accuracy_per_type[epoch] = train(training_loader, model, criterion, optimizer, is_deepcci, device)
        validation_loss[epoch], validation_accuracy[epoch], validation_accuracy_per_type[epoch] = validate(validation_loader, model, criterion, is_deepcci, device)
        scheduler.step()
        f_graph.write("{},{},{},{},{}\n".format(epoch, training_loss[epoch][-1], training_accuracy[epoch][-1], validation_loss[epoch][-1], validation_accuracy[epoch][-1]))
    f_graph.close()
    return training_loss, training_accuracy, training_accuracy_per_type, validation_loss, validation_accuracy, validation_accuracy_per_type

def test_model(model, criterion, is_deepcci, is_batch):
    normalization_type = AbsoluteNormalization1()
    _, validation_loader = create_data(training_files_path=training_files_path, normalization_type=normalization_type, unused_parameters=unused_parameters, is_deepcci=is_deepcci, is_batch=is_batch, diverse_training_folder=diverse_training_folder)
    validation_loss, validation_accuracy = ([None] * NUM_OF_EPOCHS for i in range(2))
    validation_accuracy_per_type = [None] * NUM_OF_EPOCHS
    validation_loss, validation_accuracy, validation_accuracy_per_type = validate(validation_loader, model, criterion, is_deepcci)
    return numpy.mean(validation_loss), numpy.mean(validation_accuracy), numpy.mean(validation_accuracy_per_type, axis=0)

#if __name__ == '__main__':
def main(training_file_path, unused_parameters, bg_flows, is_sample_rate, is_sample, is_deepcci, is_fully_connected_net, num_of_classification_parameters,
         chunk_size, num_of_congestion_controls, num_of_time_samples, DEVICE, results_path, is_test_only, model_path, diverse_training_folder, is_diverse, deepcci_num_of_time_samples):
    #sleep(60*60*60*48 + 60*60*1.5)
    #sleep(60*60*18 + 60*60*1.5)
    #sleep(60*60*0.1)
    if is_deepcci:
        model = deepcci_net(chunk_size, deepcci_num_of_time_samples).to(DEVICE)
        net = "deepcci_net"
    else:
        if is_fully_connected_net:
            model = fully_connected_net(num_of_classification_parameters, chunk_size, num_of_congestion_controls).to(DEVICE)
            net = "fully_connected_net"
        else:
            model = my_net(num_of_classification_parameters, chunk_size, num_of_congestion_controls, num_of_time_samples).to(DEVICE)
            net = "my_net"
    """
    if is_deepcci:
        model = deepcci_net(chunk_size, deepcci_num_of_time_samples).to(DEVICE)
        net = "deepcci_net"
        #unused_parameters = ['timestamp', 'In Throughput', 'Out Throughput', 'Connection Num of Drops', 'Connection Num of Retransmits', 'CBIQ', 'Num of Drops', 'Num of Packets', 'Total Bytes in Queue']
        # unused_parameters = ['timestamp', 'In Throughput', 'Out Throughput', 'Connection Num of Drops', 'Send Time Gap', 'Num of Drops', 'Num of Packets', 'Total Bytes in Queue']
        # online:
        #unused_parameters = ['timestamp', 'In Throughput', 'Out Throughput', 'CBIQ', 'Send Time Gap', 'In Goodput',
        #                     'Total Bytes in Queue', 'Num of Drops',
        #                     'Num of Packets']
        # online with sampling:
        unused_parameters = ['timestamp', 'CBIQ', 'In Throughput', 'Out Throughput', 'Capture Time Gap']
    else:
        if is_fully_connected_net:
            model = fully_connected_net(num_of_classification_parameters, chunk_size, num_of_congestion_controls).to(DEVICE)
            net = "fully_connected_net"
            unused_parameters = ['Connection Num of Drops', 'Connection Num of Retransmits', 'Num of Drops', 'Num of Packets', 'Total Bytes in Queue']
        else:
            model = my_net(num_of_classification_parameters, chunk_size, num_of_congestion_controls, num_of_time_samples).to(DEVICE)
            net = "my_net"
            # unused_parameters = ['In Throughput', 'Out Throughput', 'Send Time Gap', 'Num of Drops', 'Num of Packets', 'Total Bytes in Queue']
            # unused_parameters = ['In Throughput', 'Out Throughput', 'Send Time Gap', 'Connection Num of Drops', 'Num of Drops', 'Num of Packets', 'Total Bytes in Queue']

            #cbiq:
            #unused_parameters = ['In Throughput', 'Out Throughput', 'Connection Num of Drops', 'Connection Num of Retransmits', 'Send Time Gap', 'Num of Drops', 'Num of Packets', 'Total Bytes in Queue']
            # online:
            #unused_parameters = ['Capture Time Gap', 'In Throughput', 'Out Throughput', 'Send Time Gap', 'In Goodput', 'Total Bytes in Queue', 'Num of Drops', 'Num of Packets']
            #unused_parameters = ['In Throughput', 'Out Throughput', 'Send Time Gap', 'Total Bytes in Queue', 'Num of Drops', 'Num of Packets']
            # online with sampling:
            unused_parameters = ['Capture Time Gap', 'In Throughput', 'Out Throughput', 'deepcci']

            #my_deepcci:
            #unused_parameters = ['timestamp', 'Capture Time Gap', 'In Throughput', 'Out Throughput', 'CBIQ']


            #capture arrival time:
            #unused_parameters = ['timestamp', 'deepcci', 'In Throughput', 'Out Throughput', 'CBIQ']


            #throughput:
            #unused_parameters = ['CBIQ', 'Connection Num of Drops', 'Connection Num of Retransmits', 'Send Time Gap', 'Num of Drops', 'Num of Packets', 'Total Bytes in Queue']
            #unused_parameters = ['Out Throughput', 'CBIQ', 'Connection Num of Drops', 'Connection Num of Retransmits', 'Send Time Gap', 'Num of Drops', 'Num of Packets', 'Total Bytes in Queue']
            #unused_parameters = ['In Throughput', 'Out Throughput', 'Connection Num of Drops', 'Send Time Gap', 'Num of Drops', 'Num of Packets', 'Total Bytes in Queue']
            # online:
            #unused_parameters = ['Capture Time Gap', 'CBIQ', 'Send Time Gap', 'In Goodput', 'Total Bytes in Queue', 'Num of Drops', 'Num of Packets']
            # online with sampling:
            #unused_parameters = ['CBIQ', 'Capture Time Gap', 'deepcci']

            #5parameters:
            #unused_parameters = ['Connection Num of Drops', 'Connection Num of Retransmits', 'Num of Drops', 'Num of Packets', 'Total Bytes in Queue']
            #online:
            #unused_parameters = ['Send Time Gap', 'In Goodput', 'Num of Drops', 'Total Bytes in Queue', 'Num of Drops', 'Num of Packets']
            #unused_parameters = None
            # online with sampling:
            #unused_parameters = ['Capture Time Gap', 'deepcci']
            #unused_parameters = ['deepcci']

            #1parameter at a time:
            #unused_parameters = ['CBIQ', 'In Throughput', 'Out Throughput', 'Connection Num of Drops', 'Connection Num of Retransmits', 'Send Time Gap', 'Num of Drops', 'Num of Packets', 'Total Bytes in Queue']
            #unused_parameters=[]
    """
    tn = datetime.now()
    time_str = "_" + str(tn.month) + "." + str(tn.day) + "." + str(tn.year) + "@" + str(tn.hour) + "-" + str(
        tn.minute) + "-" + str(tn.second)
    # directory = graphs_path + "10bbr_cubic_reno_tcp_background_noise, "+ is_deepcci + ", " + "chunk_" + str(CHUNK_SIZE) +", shuffle_" + str(IS_SHUFFLE) + ", batch_" + str(BATCH_SIZE)
    #directory = graphs_path + "_" + str(NUM_OF_CLASSIFICATION_PARAMETERS) + "_parameters_my_deepcci_"+is_deepcci
    #directory = results_path + "_" + str(num_of_classification_parameters) + "_parameters_"+net
    directory = results_path
    # directory = graphs_path + is_deepcci + "All" + str(CHUNK_SIZE) + "_shuffle_" + str(
    #     IS_SHUFFLE) + "_batch_" + str(BATCH_SIZE)
    if not os.path.exists(results_path):
        os.makedirs(directory)
    plot_file_name = directory + "/statistics.csv"
    criterion = CrossEntropyLoss().to(DEVICE)
    if is_test_only:
        plot_file_name = directory + "/validation.png"
        model.load_state_dict(torch.load(model_path), strict=False)
        validation_loss, validation_accuracy, validation_accuracy_per_type = test_model(model, criterion, is_deepcci, IS_BATCH)
        with open(plot_file_name.replace('.png', ('_' + "f1")), 'w') as f:
            for item in [validation_loss, validation_accuracy, validation_accuracy_per_type]:
                f.write("%s\n" % item)
    else:
        model.apply(init_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)
        training_loss, training_accuracy, training_accuracy_per_type, validation_loss, validation_accuracy, validation_accuracy_per_type = run(model, criterion, optimizer, scheduler, unused_parameters, is_deepcci, results_path, IS_BATCH, plot_file_name, is_sample_rate, training_file_path, bg_flows, is_sample,diverse_training_folder, num_of_time_samples, chunk_size, is_diverse, num_of_classification_parameters, DEVICE, deepcci_num_of_time_samples)
        print('done')
        # saving the trained model
        torch.save(model, directory + '/model.pt')
        torch.save(model.state_dict(), directory + '/state_dict.pt')
        plot_file_name = directory + "/training.png"
        training_graph = Graph_Creator(training_loss, training_accuracy, training_accuracy_per_type, NUM_OF_EPOCHS, IS_BATCH, plot_file_name=plot_file_name, plot_fig_name="training statistics")
        training_graph.create_graphs()
        plot_file_name = directory + "/validation.png"
        validation_graph = Graph_Creator(validation_loss, validation_accuracy, validation_accuracy_per_type, NUM_OF_EPOCHS, IS_BATCH, plot_file_name=plot_file_name, plot_fig_name="validation statistics")
        validation_graph.create_graphs()