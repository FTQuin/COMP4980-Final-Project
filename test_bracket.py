import pickle
import pandas as pd
import torch
from train_model import MLPModel
from SeasonDataset import SeasonDataset

# constants
DATA_FILE = './data/f_and_a_no0GP_-1.csv'
DT_SAVE_FILE = 'models/dt.pickle'
MLP_SAVE_FILE = 'models/test_2.pt'
TEAMS_FILE = 'ncaam-march-mania-2021/MDataFiles_Stage2/MTeams.csv'

sd = SeasonDataset(DATA_FILE)

teams = pd.read_csv(TEAMS_FILE)

games = [
            [   # round of 64
                (1276, 1411, 1276),
                (1104, 1233, 1104),
                (1400, 1101, 1101),
                (1199, 1422, 1199),
                (1160, 1207, 1160),
                (1140, 1417, 1417),
                (1163, 1268, 1268),
                (1261, 1382, 1261),
                (1211, 1313, 1211),
                (1234, 1213, 1234),
                (1242, 1186, 1242),
                (1438, 1325, 1325),
                (1166, 1364, 1166),
                (1425, 1179, 1425),
                (1332, 1433, 1332),
                (1328, 1281, 1328),
                (1228, 1180, 1228),
                (1222, 1156, 1222),
                (1452, 1287, 1452),
                (1329, 1251, 1329),
                (1397, 1333, 1333),
                (1361, 1393, 1393),
                (1155, 1353, 1353),
                (1260, 1210, 1260),
                (1124, 1216, 1124),
                (1326, 1331, 1331),
                (1116, 1159, 1116),
                (1345, 1317, 1317),
                (1437, 1457, 1437),
                (1403, 1429, 1403),
                (1196, 1439, 1196),
                (1314, 1458, 1458)
            ],
            # round 32
            [
                (1276, 1261, 1276), #michigan
                (1160, 1199, 1199), #florida st
                (1417, 1101, 1417), #UCLA
                (1268, 1104, 1104), #alabama
                (1211, 1328, 1211), #Gonzaga
                (1166, 1325, 1166), #Creighton
                (1425, 1242, 1425), #USC
                (1332, 1234, 1332), #Oregon
                (1228, 1260, 1260), #Loyola-Chicago
                (1329, 1333, 1333), #Oregon St
                (1393, 1452, 1393), #Syracuse
                (1353, 1222, 1222), #Houston
                (1124, 1458, 1124), #Baylor
                (1437, 1317, 1437), #Villanova
                (1403, 1116, 1116), #Arkansas
                (1196, 1331, 1331) #Oral Roberts
            ],
            #sweet 16
            [
                (1276, 1199, 1276), #michigan
                (1417, 1104, 1417), #UCLA
                (1211, 1166, 1211), #Gonzaga
                (1425, 1332, 1425), #USC
                (1260, 1333, 1333), #Oregon St
                (1393, 1222, 1222), #Houston
                (1124, 1437, 1124), #Baylor
                (1116, 1331, 1116) #Arkansas
            ],
            # elite 8
            [
                (1276, 1417, 1417), # UCLA
                (1211, 1425, 1211), # Gonzaga
                (1333, 1222, 1222), # Houston
                (1124, 1116, 1124) # Baylor
            ],
            # final four
            [
                (1211, 1417, 1211), # Gonzaga
                (1124, 1222, 1124) # Baylor
            ],
            # finals
            [
                (1211, 1124, 1124) # Baylor
            ]
     ]


def get_score(model):
    score = 0
    remaining_teams = set()

    # get list of unique teams
    for g in games[0]:
        for t in g:
            remaining_teams.add(t)

    for r in range(len(games)):
        for game in games[r]:
            pred_winner = game[0 if model.guess(sd.get_custom_game(game[0], game[1])) else 1]
            pred_loser = game[1 if model.guess(sd.get_custom_game(game[0], game[1])) else 0]
            print(teams[teams['TeamID'] == pred_winner]['TeamName'].values[0])

            if pred_winner in remaining_teams:
                if pred_winner == game[2]:
                    score += 2 ** r * 10
                    print('Correct, +', 2 ** r * 10)
                else:
                    remaining_teams.remove(pred_winner)

            if pred_loser in remaining_teams:
                remaining_teams.remove(pred_loser)

        print()

    return score


if __name__ == '__main__':
    # DECISION TREE
    # with open(DT_SAVE_FILE, 'rb') as file:
    #     dt = pickle.load(file)
    #
    # def guess(game):
    #     drop_features = sd.one_hot_features.copy()
    #     drop_features.remove('season')
    #     game = game.drop(labels=drop_features)
    #
    #     return dt.predict([game])
    #
    # setattr(dt, 'guess', guess)
    # print('Final score: ', get_score(dt))

    # MLP
    with open(MLP_SAVE_FILE, 'rb') as file:
        mlp = torch.load(MLP_SAVE_FILE)

    def guess(game):
        with torch.no_grad():
            X, y = sd.get_as_tensor(game.append(pd.Series([2021, True], index=['season', 'label_t1_won'])))
            pred = mlp(X)
            print(pred)

        return 0 if pred < 0.5 else 1

    setattr(mlp, 'guess', guess)
    print('Final score: ', get_score(mlp))
