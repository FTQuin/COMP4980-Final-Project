from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from SeasonDataset import SeasonDataset


# CONSTANTS
DATA_FILE = './data/f_and_a_no0GP_-1.csv'
SAVE_NET_FILE = './networks/test_1.pt'

BATCH_SIZE = 500
EPOCHS = 1
LEARNING_RATE = 0.001
TEST_PERCENT = 0.1
TEST = True

OPTIMIZER = torch.optim.SGD
CRITERION = nn.MSELoss()


# MODEL
class MLPModel(nn.Module):

    def __init__(self, num_features):
        super(MLPModel, self).__init__()

        self.dense1 = nn.Linear(num_features, 1500)
        self.dense2 = nn.Linear(1500, 700)
        self.dense3 = nn.Linear(700, 500)
        self.dense4 = nn.Linear(500, 300)
        self.dense5 = nn.Linear(300, 100)
        self.dense6 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.tanh(self.dense1(x))
        x = torch.tanh(self.dense2(x))
        x = torch.tanh(self.dense3(x))
        x = torch.tanh(self.dense4(x))
        x = torch.tanh(self.dense5(x))
        x = torch.sigmoid(self.dense6(x))
        return x


# MAIN
if __name__ == '__main__':

    # get dataset
    sd = SeasonDataset(DATA_FILE)
    # train test split
    sd_train, sd_test = random_split(sd, [len(sd) - int(len(sd) * TEST_PERCENT), int(len(sd) * TEST_PERCENT)])
    dl_train = DataLoader(sd_train, batch_size=BATCH_SIZE, shuffle=True)

    # init network, loss, and optimizer
    model = MLPModel(sd.num_features)
    criterion = CRITERION
    optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE)

    # variables for graphing
    total_batches = 0
    loss_values = []
    total_tests = 0
    test_losses = []

    # -------- train loop --------
    for epoch in range(EPOCHS):
        for X, y in dl_train:
            # get prediction
            pred_y = model(X)
            # calc loss
            loss = criterion(pred_y, y.view(-1, 1))
            # back propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # vars for graph
            total_batches += 1
            loss_values.append(loss.item())
            # show progress
            if total_batches * BATCH_SIZE % 1000 == 0:
                print((total_batches * BATCH_SIZE) / (len(sd_train) * EPOCHS))

        print('epoch {}\n loss {}\n'.format(epoch, loss.item(),))

        if TEST:
            # test at each epoch
            with torch.no_grad():
                # vars for testing
                dl_test = DataLoader(sd_test, batch_size=BATCH_SIZE, shuffle=True)
                for X, y in dl_test:
                    pred_y = model(X)

                    num_correct = torch.sum(torch.round(pred_y) == y).float()

                    total_tests += 1
                    test_losses.append(num_correct/len(y))

                print('Test average: ' + str(np.average(test_losses)))

    # save model
    torch.save(model, SAVE_NET_FILE)

    # show graphs
    # loss over time
    plt.figure(figsize=(15, 5))
    plt.plot(range(total_batches), loss_values)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Train'+str(datetime.now()))
    plt.show()
    # tests correct
    plt.figure(figsize=(15, 5))
    plt.plot(range(total_tests), test_losses)
    plt.xlabel('Test')
    plt.ylabel('Tests Correct')
    plt.title('Test'+str(datetime.now()))
    plt.show()
