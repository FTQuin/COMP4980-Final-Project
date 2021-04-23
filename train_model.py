from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from SeasonDataset import SeasonDataset


# CONSTANTS
DATA_FILE = './data/f_and_a_no0GP_-1.csv'
SAVE_NET_FILE = 'models/test_4.pt'

BATCH_SIZE = 500
EPOCHS = 2
LEARNING_RATE = 0.01
TEST_PERCENT = 0.1
TEST = True

OPTIMIZER = torch.optim.Adam
CRITERION = nn.MSELoss()


# MODEL
class MLPModel(nn.Module):

    def __init__(self, num_features):
        super(MLPModel, self).__init__()

        self.dense1 = nn.Linear(num_features, 50)
        self.dense2 = nn.Linear(50, 25)
        self.dense3 = nn.Linear(25, 10)
        self.dense4 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        x = torch.sigmoid(self.dense4(x))
        return x


# MAIN
if __name__ == '__main__':

    # get dataset
    sd = SeasonDataset(DATA_FILE)
    sd_train = SeasonDataset(DATA_FILE)
    sd_test = SeasonDataset(DATA_FILE)
    indicies = sd_train.df_features.index.values.tolist()
    # train test split
    train_index, test_index = train_test_split(indicies, test_size=TEST_PERCENT)
    sd_train.df_features = sd_train.df_features.iloc[train_index]
    sd_test.df_features = sd_test.df_features.iloc[test_index]
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
                print(loss.item())
                print()

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
