from learning.my_net import *
from learning.deepcci_net import *
from learning.fully_connected_net import *
from torch.nn import Linear, CrossEntropyLoss, Conv2d

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
    print('validation is done')
    return validation_loss, validation_accuracy, validation_accuracy_per_type

def run(model, criterion, optimizer, scheduler, unused_parameters, is_deepcci, results_path, is_batch, plot_file_name,
        is_sample_rate, training_files_path, bg_flows, is_sample, diverse_training_folder, num_of_time_samples, chunk_size,
        is_diverse, num_of_classification_parameters, device, deepcci_num_of_time_samples):
    normalization_type = AbsoluteNormalization1()
    training_loader, validation_loader = create_data(training_files_path=training_files_path, results_path=results_path, normalization_type=normalization_type,
                                                     unused_parameters=unused_parameters, is_deepcci=is_deepcci, is_batch=is_batch,
                                                     diverse_training_folder=diverse_training_folder, is_sample_rate=is_sample_rate, bg_flows=bg_flows, is_sample=is_sample,
                                                     num_of_time_samples=num_of_time_samples, chunk_size=chunk_size, is_diverse=is_diverse,
                                                     num_of_classification_parameters=num_of_classification_parameters, deepcci_num_of_time_samples=deepcci_num_of_time_samples)
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

def test_model(model, criterion, is_deepcci, training_files_path, unused_parameters, is_batch, diverse_training_folder):
    normalization_type = AbsoluteNormalization1()
    _, validation_loader = create_data(training_files_path=training_files_path, normalization_type=normalization_type, unused_parameters=unused_parameters, is_deepcci=is_deepcci, is_batch=is_batch, diverse_training_folder=diverse_training_folder)
    validation_loss, validation_accuracy = ([None] * NUM_OF_EPOCHS for i in range(2))
    validation_accuracy_per_type = [None] * NUM_OF_EPOCHS
    validation_loss, validation_accuracy, validation_accuracy_per_type = validate(validation_loader, model, criterion, is_deepcci)
    return numpy.mean(validation_loss), numpy.mean(validation_accuracy), numpy.mean(validation_accuracy_per_type, axis=0)

#def _main(training_file_path, unused_parameters, bg_flows, is_sample_rate, is_sample, is_deepcci, is_fully_connected_net, num_of_classification_parameters,
def _main(sim_params, model_params, net_type, DEVICE):
    net = net_type.get_net()
    is_deepcci = False
    if net == "deepcci_net":
        model = deepcci_net(model_params.chunk_size, model_params.deepcci_num_of_time_samples).to(DEVICE)
        is_deepcci = True
    else:
        if net == "my_net":
            model = my_net(net_type.get_num_of_classification_parameters(), model_params.chunk_size, model_params.num_of_congestion_controls, model_params.num_of_time_samples).to(DEVICE)
        else:
            model = fully_connected_net(net_type.get_num_of_classification_parameters(), model_params.chunk_size, model_params.num_of_congestion_controls).to(DEVICE)
    if not os.path.exists(sim_params.results_path):
        os.makedirs(sim_params.results_path)
    plot_file_name = sim_params.results_path + "/statistics.csv"
    criterion = CrossEntropyLoss().to(DEVICE)
    return model, net, plot_file_name, criterion, is_deepcci

def main_train_and_validate(sim_params, model_params, DEVICE):
    net_type = model_params.net_type
    model, net, plot_file_name, criterion, is_deepcci = _main(sim_params, model_params, net_type, DEVICE)
    # sleep(sleep_duration)
    if sim_params.csv_filename == "random":
        is_sample_rate = True
    else:
        is_sample_rate = False
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)
    training_loss, training_accuracy, training_accuracy_per_type, validation_loss, validation_accuracy, validation_accuracy_per_type = run(
        model, criterion, optimizer, scheduler, net_type.get_unused_parameters(), is_deepcci, sim_params.results_path, IS_BATCH, plot_file_name,
        is_sample_rate, sim_params.data_path, model_params.bg_flow, sim_params.is_data_sample, sim_params.diverse_data_path, model_params.num_of_time_samples,
        model_params.chunk_size, sim_params.is_diverse_data, net_type.get_num_of_classification_parameters(), DEVICE, net_type.get_deepcci_num_of_time_samples())
    print('done')
    # saving the trained model
    if sim_params.save_model_pt:
        torch.save(model, sim_params.results_path + '/model.pt')
        torch.save(model.state_dict(), sim_params.results_path + '/state_dict.pt')
    plot_file_name = sim_params.results_path + "/training.png"
    training_graph = Graph_Creator(training_loss, training_accuracy, training_accuracy_per_type, NUM_OF_EPOCHS,
                                   model_params.is_batch, plot_file_name=plot_file_name, plot_fig_name="training statistics")
    training_graph.create_graphs()
    plot_file_name = sim_params.results_path + "/validation.png"
    validation_graph = Graph_Creator(validation_loss, validation_accuracy, validation_accuracy_per_type, NUM_OF_EPOCHS,
                                     model_params.is_batch, plot_file_name=plot_file_name, plot_fig_name="validation statistics")
    validation_graph.create_graphs()

def main_validate(sim_params, model_params, DEVICE):
    net_type = model_params.net_type
    model, net, plot_file_name, criterion, is_deepcci = _main(sim_params, model_params, net_type, DEVICE)
    # sleep(sleep_duration)
    plot_file_name = sim_params.results_path + "/validation.png"
    model.load_state_dict(torch.load(sim_params.model_path), strict=False)
    validation_loss, validation_accuracy, validation_accuracy_per_type = test_model(model, criterion, is_deepcci, model_params.training_files_path, model_params.unused_parameters, model_params.is_batch, model_params.diverse_training_folder)
    with open(plot_file_name.replace('.png', ('_' + "f1")), 'w') as f:
        for item in [validation_loss, validation_accuracy, validation_accuracy_per_type]:
            f.write("%s\n" % item)