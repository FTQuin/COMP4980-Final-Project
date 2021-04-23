import torch
import pandas as pd
from torch.utils.data import Dataset
from calculate_features import one_hot_features, data_headers, label


class SeasonDataset(Dataset):

    def __init__(self, features_file):
        # column names
        self.one_hot_features = one_hot_features
        self.data_headers = data_headers
        self.label = label

        # features df
        self.df_features = pd.read_csv(features_file)

        # teams series for one hot encoding
        self.df_teams = pd.Series(pd.unique(pd.concat(
            [self.df_features['t1_team_id'], self.df_features['t2_team_id']])))
        # season series for one hot encoding
        self.df_seasons = pd.Series(pd.unique(self.df_features['season']))
        # location series for one hot encoding
        self.df_locations = pd.Series(pd.unique(self.df_features['t1_location']))

        # number of features
        self.num_features = len(self.data_headers)-6 + len(self.df_teams) * 2 +\
                                len(self.df_seasons) + len(self.df_locations) * 2

    def __len__(self):
        return len(self.df_features)

    def __getitem__(self, item):
        # get row
        raw_row = self.df_features.iloc[item]
        return self.get_as_tensor(raw_row)

    def get_as_tensor(self, raw_row):
        # convert to one hots
        oht_t1 = self.to_one_hot(raw_row['t1_team_id'], self.df_teams)
        oht_t2 = self.to_one_hot(raw_row['t2_team_id'], self.df_teams)
        oht_season = self.to_one_hot(raw_row['season'], self.df_seasons)
        oht_t1_location = self.to_one_hot(raw_row['t1_location'], self.df_locations)
        oht_t2_location = self.to_one_hot(raw_row['t2_location'], self.df_locations)
        X_one_hots = torch.cat((oht_t1, oht_t2,
                                oht_season,
                                oht_t1_location, oht_t2_location))

        # get other features
        X_others = torch.tensor(raw_row.drop(labels=self.one_hot_features+self.label).to_numpy().astype(float)).float()

        # combine them all
        X = torch.cat((X_one_hots, X_others))

        # add label to the end
        y = torch.tensor(raw_row[self.label]).float()

        # send it out
        return X, y

    # turns a team id into a one hot vector
    @staticmethod
    def to_one_hot(value, series):
        v = torch.eye(len(series))[series[series == value].index[0]].view(-1)
        return v

    def get_custom_game(self, team1ID, team2ID):
        t1 = self.get_team_stat(team1ID, 1)
        t2 = self.get_team_stat(team2ID, 2)

        game = t1.append(t2)
        return game

    def get_team_stat(self, teamID, new_team_position):
        t_1 = self.df_features.loc[self.df_features['t1_team_id'] == teamID]
        t_2 = self.df_features.loc[self.df_features['t2_team_id'] == teamID]

        if t_1.index.max() > t_2.index.max():
            cols = [c for c in self.df_features.columns if c[:2] == 't1']
            t = self.df_features.iloc[t_1.index.max()][cols]
        else:
            cols = [c for c in self.df_features.columns if c[:2] == 't2']
            t = self.df_features.iloc[t_2.index.max()][cols]

        t = t.rename(lambda s: 't'+str(new_team_position)+s[2:], axis='columns')

        return t
