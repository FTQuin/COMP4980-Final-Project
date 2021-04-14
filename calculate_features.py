import pandas as pd
from sklearn.preprocessing import StandardScaler


# CONSTANTS
RAW_DATA_FILE_NAME = 'ncaam-march-mania-2021\MDataFiles_Stage2/MRegularSeasonDetailedResults.csv'
TEAM_DATA_FILE_NAME = 'ncaam-march-mania-2021\MDataFiles_Stage2/MTeams.csv'
DATA_START_POINT = 0
NUM_OF_OUTPUTS = -1
OUTPUT_FILE_NAME = './data/f_and_a_no0GP_' + str(NUM_OF_OUTPUTS) + '.csv'

# VARIABLES
# create features
one_hot_features = ['season', 't1_team_id', 't1_location', 't2_team_id', 't2_location']
team_features = ['games_played', 'wins', 'wl_ratio',
                 'points',
                 'FG_attempted', 'FG_made', 'FG_missed',
                 '3s_attempted', '3s_made', '3s_missed',
                 'FT_attempted', 'FT_made', 'FT_missed',
                 'O_rebounds', 'D_rebounds', 'total_rebounds',
                 'assists', 'TOs', 'a_TO_ratio',
                 'steals',
                 'blocks', 'fouls', 'b_f_ratio']
team_features_for = ['t1_' + f for f in team_features]
team_features_for += ['t2_' + f for f in team_features]
team_features_against = [f + '_agst' for f in team_features_for]

# create label
label = ['label_t1_won']

# create order of features
data_headers = one_hot_features + team_features_for + team_features_against + label
df_features = pd.DataFrame(None, columns=data_headers)

# create dataframes for different team stats
stats = ['team_id'] + team_features
# stats that a team and
# stats that a team had against them
df_season_stats_for = pd.DataFrame(None, columns=stats)
df_season_stats_agst = pd.DataFrame(None, columns=stats)

# create dataframe for raw data
df_raw = pd.DataFrame()
# create map from raw name to parsed name
name_map = {
    'points': 'Score',
    'FG_made': 'FGM',
    'FG_attempted': 'FGA',
    '3s_made': 'FGM3',
    '3s_attempted': 'FGA3',
    'FT_made': 'FTM',
    'FT_attempted': 'FTA',
    'O_rebounds': 'OR',
    'D_rebounds': 'DR',
    'assists': 'Ast',
    'TOs': 'TO',
    'steals': 'Stl',
    'blocks': 'Blk',
    'fouls': 'PF'
}

# other variables
curr_season = 0


# FUNCTIONS
def check_new_season(row_):
    global curr_season

    if row_['Season'] != curr_season:
        # set curr_season to right value
        curr_season = row_['Season']

        # set all stats to zero
        for col in df_season_stats_for.columns:
            df_season_stats_for[col].values[:] = 0
        for col in df_season_stats_agst.columns:
            df_season_stats_agst[col].values[:] = 0


def get_stats(row_, T):
    # get id
    team_id = row_[T+'TeamID']

    # get location
    location = row_['WLoc']
    # flip location for loser
    if T == 'L' and location == 'H':
        location = 'A'
    elif T == 'L' and location == 'A':
        location = 'H'

    # get stats for and against
    team_stats_for = df_season_stats_for.loc[team_id]
    team_stats_agst = df_season_stats_agst.loc[team_id]
    team_stats_agst = team_stats_agst.rename(lambda a: a+'_agst', axis='columns')
    # combine
    team_stats = pd.concat([team_stats_for, team_stats_agst], axis=0)
    # add team id
    team_stats['team_id'] = team_id
    # add location
    team_stats['location'] = location

    return team_stats


def stats_to_feature(t1_stats_, t2_stats_, t1_won_):
    global curr_season

    t1_stats_ = t1_stats_.rename(lambda a: 't1_'+a, axis='columns')
    t2_stats_ = t2_stats_.rename(lambda a: 't2_'+a, axis='columns')

    df_features_ = pd.concat([t1_stats_, t2_stats_], axis=0)

    df_features_['season'] = curr_season

    df_features_['label_t1_won'] = t1_won_

    return df_features_


def write_feature(f, out_file_):
    f = f.to_frame().transpose()
    f = f[data_headers]
    f.to_csv(out_file_, header=False, index=False, mode='a')


def update_stats(row_):
    update_single_stat_df(row_, df_season_stats_for, 'W', True)
    update_single_stat_df(row_, df_season_stats_agst, 'W', False)
    update_single_stat_df(row_, df_season_stats_for, 'L', True)
    update_single_stat_df(row_, df_season_stats_agst, 'L', False)


def update_single_stat_df(row_, stat_df_, team_T, is_stats_for):
    if is_stats_for:
        T = team_T
    else:
        T = 'L' if team_T == 'W' else 'W'
    OT = 'L' if T == 'W' else 'W'

    # get team id
    team_id = row_[team_T+'TeamID']

    # update stats
    # update games played
    stat_df_.loc[team_id, 'games_played'] += 1
    # set games played to a variable to help calc averages
    games_played = stat_df_.loc[team_id, 'games_played']
    # update wins
    if row_[T+name_map['points']] > row_[OT+name_map['points']]:
        stat_df_.loc[team_id, 'wins'] += 1

    # update stats in name_map
    for stat_name, raw_stat_name in name_map.items():
        stat_df_.loc[team_id, stat_name] = update_average(games_played, stat_df_.loc[team_id, stat_name],
                                                          row_[T+raw_stat_name])

    # update other stats (totals, ratios, differences)
    # update win loss ratio
    stat_df_.loc[team_id, 'wl_ratio'] = stat_df_.loc[team_id, 'wins'] / games_played
    # update missed shots
    # missed field goals
    stat_df_.loc[team_id, 'FG_missed'] = stat_df_.loc[team_id, 'FG_attempted'] - stat_df_.loc[team_id, 'FG_made']
    # missed 3s
    stat_df_.loc[team_id, '3s_missed'] = stat_df_.loc[team_id, '3s_attempted'] - stat_df_.loc[team_id, '3s_made']
    # missed free throws
    stat_df_.loc[team_id, 'FT_missed'] = stat_df_.loc[team_id, 'FT_attempted'] - stat_df_.loc[team_id, 'FT_made']
    # update non scoring stats
    # total rebounds
    stat_df_.loc[team_id, 'total_rebounds'] = stat_df_.loc[team_id, 'O_rebounds'] + stat_df_.loc[team_id, 'D_rebounds']
    # assist to turnover ratio
    stat_df_.loc[team_id, 'a_TO_ratio'] = stat_df_.loc[team_id, 'assists'] / stat_df_.loc[team_id, 'TOs']
    # block to foul ratio
    stat_df_.loc[team_id, 'b_f_ratio'] = stat_df_.loc[team_id, 'blocks'] / stat_df_.loc[team_id, 'fouls']


def update_average(games_played_, curr_avg, new_value):
    return (((games_played_ - 1) * curr_avg) + new_value) / games_played_


def normalize_features():
    # load data from file
    with open(OUTPUT_FILE_NAME, mode='r', newline='') as _out_file:
        df_file = pd.read_csv(_out_file)

    # get only the values that need to be scaled
    df_scale = df_file.drop(columns=one_hot_features+label)

    # scale values
    scaled_values = StandardScaler().fit_transform(df_scale.values)

    # put values back in file dataframe
    df_scale = pd.DataFrame(scaled_values, columns=df_scale.columns)
    df_file.update(df_scale)

    # write scaled values to file
    with open(OUTPUT_FILE_NAME, mode='w', newline='') as _out_file:
        df_file.to_csv(_out_file, header=True, index=False, mode='a')


# MAIN
if __name__ == '__main__':
    # open raw stats file and put into dataframe
    with open(RAW_DATA_FILE_NAME) as in_file:

        # load file into dataframe
        df_raw = pd.read_csv(in_file)

    # put teams into season stats df
    with open(TEAM_DATA_FILE_NAME) as in_file:
        df_team_ids = pd.read_csv(in_file)['TeamID']
        df_season_stats_for['team_id'] = df_team_ids
        df_season_stats_agst['team_id'] = df_team_ids
        # fill empty values with zeros
        df_season_stats_for = df_season_stats_for.fillna(0)
        df_season_stats_agst = df_season_stats_agst.fillna(0)
        # set index to team_id
        df_season_stats_for = df_season_stats_for.set_index('team_id')
        df_season_stats_agst = df_season_stats_agst.set_index('team_id')

    # go through every game chronologically and
    # calculate features to put into features dataframe
    # then update stats

    with open(OUTPUT_FILE_NAME, mode='w', newline='') as out_file:
        # give the file a header
        pd.DataFrame([data_headers]).to_csv(out_file, header=False, index=False, mode='a')

        # ---------- main loop ----------
        for index, row in df_raw.iloc[
            range(DATA_START_POINT,
                  DATA_START_POINT + NUM_OF_OUTPUTS if NUM_OF_OUTPUTS > 0 else len(df_raw))
        ].iterrows():

            # if new season reset season team stats
            check_new_season(row)

            # get team stats
            w_stats = get_stats(row, 'W')
            l_stats = get_stats(row, 'L')

            # parse stats to features
            feature1 = stats_to_feature(w_stats, l_stats, True)
            feature2 = stats_to_feature(l_stats, w_stats, False)

            # write features to file
            write_feature(feature1, out_file)
            write_feature(feature2, out_file)

            # update team stats df
            update_stats(row)

            # show progress
            if index % 100 == 0:
                print(index)

    # load data from file
    with open(OUTPUT_FILE_NAME, mode='r', newline='') as _out_file:
        df_file = pd.read_csv(_out_file)

    # get rid of teams first game of the season
    df_file = df_file[df_file['t1_games_played'] > 0]
    df_file = df_file[df_file['t2_games_played'] > 0]
    df_file = df_file.reset_index(drop=True)

    # write df back into file
    with open(OUTPUT_FILE_NAME, mode='w', newline='') as _out_file:
        df_file.to_csv(_out_file, header=True, index=False, mode='a')

    # normalize data that needs to be normalized
    # normalize_features()
