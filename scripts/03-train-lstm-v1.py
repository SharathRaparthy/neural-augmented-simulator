import os

from comet_ml import Experiment

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from nas.data import MODELS_PATH
from nas.data.datasets import RealRecordingsV1
from nas.models.networks import LstmNetRealv1
from nas.utils import log_parameters
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
experiment = Experiment(
    api_key="ZfKpzyaedH6ajYSiKmvaSwyCs",
    project_name="nas-v2",
    workspace="fgolemo")

HIDDEN_NODES = 128
LSTM_LAYERS = 3
EPOCHS = 5
VARIANT = "10"
EXPERIMENT_ID = 1  # increment and then commit to github when you change training/network code
start = 100
samples = 500
end = start + samples
x = np.arange(start, end)
MODEL_PATH = os.path.join(
    MODELS_PATH, f"model-"
    f"exp{EXPERIMENT_ID}-"
    f"h{HIDDEN_NODES}-"
    f"l{LSTM_LAYERS}-"
    f"v{VARIANT}-"
    f"e{EPOCHS}.pth")

log_parameters(
    experiment,
    hidden_nodes=HIDDEN_NODES,
    lstm_layers=LSTM_LAYERS,
    epochs=EPOCHS,
    variant=VARIANT,
    experiment_id=EXPERIMENT_ID)

dataset = RealRecordingsV1(dtype=VARIANT)
dataset_size = len(dataset)
indices = list(range(dataset_size))
# SPLITING THE DATASET INTO TRAIN AND TEST
test_split = 0.2
split = int(np.floor(test_split * dataset_size))
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)


# out_x = np.zeros((1120,998,30),dtype=np.float)
# out_y = np.zeros((1120,998,12), dtype=np.float)
#
# for i in range(len(ds)):
#     out_x[i] = ds[i]["x"]
#     out_y[i] = ds[i]["y"]
# xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

# batch size has to be 1, otherwise the LSTM doesn't know what to do
dataloader_train = DataLoader(
    dataset, batch_size=1, num_workers=1, sampler=train_sampler)
dataloader_test = DataLoader(
    dataset, batch_size=1, num_workers=1, sampler=test_sampler)
net = LstmNetRealv1(
    n_input_state_sim=12,
    n_input_state_real=12,
    n_input_actions=6,
    nodes=HIDDEN_NODES,
    layers=LSTM_LAYERS)
if torch.cuda.is_available():
    net = net.cuda()
net = net.float()


def extract(dataslice):
    x, y = (dataslice["x"].transpose(0, 1).float(),
            dataslice["y"].transpose(0, 1).float())

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    return x, y


def save(model):
    torch.save(model.state_dict(), MODEL_PATH)

pred_sim_real_trajectories = {'real_trajectories': np.zeros((998, 12)),
                              'pred_sim_trajectories': np.zeros((998, 12)),
                              'actions': np.zeros((998, 6))}
loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters())
test = next(iter(tqdm(dataloader_train, desc="EPISD: ")))
test_traj = test['x'].view(998, 1, -1).cuda()
for epoch in trange(EPOCHS, desc="EPOCH: "):

    loss_epoch = 0
    diff_epoch = 0
    test_loss_epoch = 0
    test_diff_epoch = 0

    for epi_idx, epi_data in enumerate(tqdm(dataloader_train, desc="EPISD: ")):
        x, y = extract(epi_data)
        net.zero_grad()
        net.zero_hidden()
        optimizer.zero_grad()

        delta = net.forward(x)

        # for idx in range(len(x)):
        #     print(idx, "=")
        #     print("real t1_x:", np.around(x[idx, 0, 12:24].cpu().data.numpy(), 2))
        #     print("sim_ t2_x:", np.around(x[idx, 0, :12].cpu().data.numpy(), 2))
        #     print("action__x:", np.around(x[idx, 0, 24:].cpu().data.numpy(), 2))
        #     print("real t2_x:",
        #           np.around(x[idx, 0, :12].cpu().data.numpy() + y[idx, 0].cpu().data.numpy(), 2))
        #     print("real t2_y:",
        #           np.around(x[idx, 0, :12].cpu().data.numpy() + delta[idx, 0].cpu().data.numpy(), 2))
        #     print("delta___x:",
        #           np.around(y[idx, 0].cpu().data.numpy(), 3))
        #     print("delta___y:",
        #           np.around(delta[idx, 0].cpu().data.numpy(), 3))
        #     print("===")

        loss = loss_function(delta, y)
        loss.backward()
        optimizer.step()

        loss_episode = loss.clone().cpu().data.numpy()
        diff_episode = (y.cpu().data.numpy()**2).mean(axis=None)

        experiment.log_metric("train loss episode", loss_episode)
        experiment.log_metric("train diff episode", diff_episode)

        loss.detach_()
        net.hidden[0].detach_()
        net.hidden[1].detach_()

        loss_epoch += loss_episode
        diff_epoch += diff_episode



        if (epi_idx + 1) % 10 == 0:
            # Testing
            with torch.no_grad():
                net.eval()
                for test_epi_idx, test_epi_data in enumerate(tqdm(dataloader_test, desc="EPISD: ")):
                    test_x, test_y = extract(test_epi_data)
                    test_delta = net.forward(test_x)
                    test_loss_episode = loss_function(test_delta, test_y)
                    test_diff_episode = (test_y.cpu().data.numpy() ** 2).mean(axis=None)
                    experiment.log_metric("test loss episode", test_loss_episode)
                    experiment.log_metric("test diff episode", test_diff_episode)
                    test_loss_epoch += test_loss_episode
                    test_diff_epoch += test_diff_episode

            with torch.no_grad():
                net.zero_hidden()
                input_tensor = test_traj[0].unsqueeze(0)
                print(input_tensor.shape)
                diff = net.forward(input_tensor.float())
                count = 0
                for traj in test_traj[1:]:
                    net.zero_hidden()
                    input_real = diff.cpu().numpy() + input_tensor.cpu().numpy()[:, :, 18:]
                    net.hidden[0].detach_()
                    net.hidden[1].detach_()
                    pred_sim_real_trajectories["pred_sim_trajectories"][count, :] = input_real
                    pred_sim_real_trajectories["actions"][count, :] = input_tensor.cpu().numpy()[:, :, 12:18]
                    input_array = traj.unsqueeze(0).cpu().numpy()
                    pred_sim_real_trajectories["real_trajectories"][count, :] = input_array[:, :, :12]
                    input_array[:, :, :12] = input_real
                    input_tensor = torch.FloatTensor(input_array).cuda()
                    diff = net.forward(input_tensor)

                    count += 1
            plt.figure(1)
            for i in range(4):
                plt.subplot(int("22" + str(i+1)))
                plt.plot(pred_sim_real_trajectories["real_trajectories"][start:end, i + 1], label=f"motor {i + 1} real-pos")
                plt.plot(pred_sim_real_trajectories["pred_sim_trajectories"][start:end, i + 1], label=f"motor {i + 1} lstm-pos", linestyle="dashed")
                plt.plot(pred_sim_real_trajectories["actions"][start:end, i + 1], label=f"motor {i + 1} action", linestyle="dotted")
                plt.ylim(-1,1)
                plt.title(f"Epoch : {epoch} - Episode : {epi_idx} - Start : {start} End: {end}")
                plt.legend()
                
            plt.savefig("/home/sharath/neural-augmented-simulator/image.png")
            experiment.log_image('/home/sharath/neural-augmented-simulator/image.png')
            plt.clf()
            net.train()



    # print(loss_epoch, diff_epoch)
    experiment.log_metric("loss epoch", loss_epoch / len(dataloader_train))
    experiment.log_metric("diff epoch", diff_epoch / len(dataloader_train))
    experiment.log_metric("loss epoch", test_loss_epoch / len(dataloader_test))
    experiment.log_metric("diff epoch", test_diff_epoch / len(dataloader_test))

    save(net)

    #TODO take a single trajectory and roll it out based on the real data and based on the LSTM corrections, then print it to matplotlib and upload to comet via
    # experiment.log_figure()

    #TODO split into train and test set
