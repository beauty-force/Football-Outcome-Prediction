import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.__version__)

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
matches = pd.read_csv('https://www.football-data.co.uk/mmz4281/2223/E0.csv')
upcomings = pd.read_csv('https://www.football-data.co.uk/fixtures.csv')
matches.head()
pd.isnull(matches).sum()
#Load a part of the data to able to do calculations faster
#matches = matches.sample(n=5000)

def get_test_match_sets(matches):
    b365_matches = matches.dropna(subset=['B365H', 'B365D', 'B365A'],inplace=False)
    b365_matches.drop(['BWH', 'BWD', 'BWA', 
            'IWH', 'IWD', 'IWA',  
            ], inplace = True, axis = 1)

    b365_matches = b365_matches.dropna(inplace=False)

    bw_matches = matches.dropna(subset=['BWH', 'BWD', 'BWA'],inplace=False)
    bw_matches.drop(['B365H', 'B365D', 'B365A', 
            'IWH', 'IWD', 'IWA',  
            ], inplace=True, axis = 1)

    bw_matches = bw_matches.dropna(inplace=False)

    iw_matches = matches.dropna(subset=['IWH', 'IWD', 'IWA'],inplace=False)
    iw_matches.drop(['B365H', 'B365D', 'B365A', 
            'BWH', 'BWD', 'BWA',  
            ], inplace=True, axis = 1)

    iw_matches = iw_matches.dropna(inplace=False)

    lb_matches = matches#.dropna(subset=['LBH', 'LBD', 'LBA'],inplace=False)
    lb_matches.drop(['B365H', 'B365D', 'B365A', 
            'BWH', 'BWD', 'BWA',  
            'IWH', 'IWD', 'IWA',
            ], inplace=True, axis = 1)

    lb_matches = lb_matches.dropna(inplace=False)
    return [b365_matches,bw_matches,iw_matches,lb_matches]

bookeeper_list = get_test_match_sets(matches)
upcoming_list = get_test_match_sets(upcomings)
#Gets a label for a given match.
def get_match_outcome(match):
    home_goals = match['FTHG']
    away_goals = match['FTAG']
     
    outcome = pd.DataFrame()
    #outcome.loc[0,'match_api_id'] = match['match_api_id'] 

    #Detect match outcome  
    if home_goals > away_goals:
        outcome.loc[0,'outcome'] = "Home Team Win"
    if home_goals == away_goals:
        outcome.loc[0,'outcome'] = "Draw"
    if home_goals < away_goals:
        outcome.loc[0,'outcome'] = "Away Team Win"
      
    return outcome.loc[0]


#Get last x matches of a team.
def get_last_matches(matches, Date, team, x = 10):
    #Filter team matches from matches
    team_matches = matches[(matches['HomeTeam'] == team) | (matches['AwayTeam'] == team)]
                           
    #Filter x last matches from team matches
    last_matches = team_matches[team_matches.Date < Date].sort_values(by = 'Date', ascending = False).iloc[0:x,:]
    
    return last_matches
    
    
def get_goals(matches, team):
    home_goals = int(matches.FTHG[matches.HomeTeam == team].sum())
    away_goals = int(matches.FTAG[matches.AwayTeam == team].sum())

    total_goals = home_goals + away_goals
    
    return total_goals


def get_goals_conceided(matches, team):
    home_goals = int(matches.FTHG[matches.AwayTeam == team].sum())
    away_goals = int(matches.FTAG[matches.HomeTeam == team].sum())

    total_goals = home_goals + away_goals

    return total_goals


#Get number of wins of a specfic team from a set of matches.
def get_wins(matches, team):
    #Find home and away wins
    home_wins = int(matches.FTHG[(matches.HomeTeam == team) & (matches.FTHG > matches.FTAG)].count())
    away_wins = int(matches.FTAG[(matches.AwayTeam == team) & (matches.FTAG > matches.FTHG)].count())

    total_wins = home_wins + away_wins

    return total_wins 


#Create match specific features for a given match.
def get_match_features(match, matches, x = 10):
    Date = match.Date
    home_team = match.HomeTeam
    away_team = match.AwayTeam
    
    #Get last x matches of home and away team
    matches_home_team = get_last_matches(matches, Date, home_team, x = 5)
    matches_away_team = get_last_matches(matches, Date, away_team, x = 5)
    
    #Create goal variables
    home_goals = get_goals(matches_home_team, home_team)
    home_goals_conceided = get_goals_conceided(matches_home_team, home_team)
    away_goals = get_goals(matches_away_team, away_team)
    away_goals_conceided = get_goals_conceided(matches_away_team, away_team)
    
    #Define result data frame
    result = pd.DataFrame()
    
    
    result.loc[0, 'home_team_goals_difference'] = home_goals - home_goals_conceided
    result.loc[0, 'away_team_goals_difference'] = away_goals - away_goals_conceided
    result.loc[0, 'games_won_home_team'] = get_wins(matches_home_team, home_team) 
    result.loc[0, 'games_won_away_team'] = get_wins(matches_away_team, away_team)

    if 'B365H' in matches.columns:
        result.loc[0, 'B365H'] = match.B365H
        result.loc[0, 'B365D'] = match.B365D
        result.loc[0, 'B365A'] = match.B365A
    elif 'BWH' in matches.columns:
        result.loc[0, 'BWH'] = match.BWH
        result.loc[0, 'BWD'] = match.BWD
        result.loc[0, 'BWA'] = match.BWA
    elif 'IWH' in matches.columns:
        result.loc[0, 'IWH'] = match.IWH
        result.loc[0, 'IWD'] = match.IWD
        result.loc[0, 'IWA'] = match.IWA

    return result.loc[0]

#Create and combine features and labels for all matches
def get_features(matches, x = 10):
    
    #Get match features for all matches
    match_stats = matches.apply(lambda i: get_match_features(i, matches, x = 10), axis = 1)
    
    #Create dummies for league_id feature
    # dummies = pd.get_dummies(match_stats['league_id']).rename(columns = lambda x: 'League_' + str(x))
    # match_stats = pd.concat([match_stats, dummies], axis = 1)
    # match_stats.drop(['league_id'], inplace = True, axis = 1)
    
    #Create match outcomes
    outcomes = matches.apply(get_match_outcome, axis = 1)

    #Merge features, outcomes into one frame
    features = pd.concat([match_stats, outcomes], axis = 1)
    #Drop NA values
    # features.dropna(inplace = True)
    
    return features
from sklearn.preprocessing import Normalizer

viables = [None] * len(bookeeper_list)
outcomes = [None] * len(bookeeper_list)
features = [None] * len(bookeeper_list)
up_viables = [None] * len(bookeeper_list)
up_outcomes = [None] * len(bookeeper_list)
up_features = [None] * len(bookeeper_list)

for i in range(len(bookeeper_list)):
    #Create features and labels based on the provided data
    viables[i] = get_features(bookeeper_list[i], 10)
    
    outcomes[i] = viables[i].loc[:, 'outcome']
    
    features[i] = viables[i].drop('outcome', axis=1)
    
    up_viables[i] = get_features(upcoming_list[i], 10)
    
    up_outcomes[i] = up_viables[i].loc[:, 'outcome']
    
    up_features[i] = up_viables[i].drop('outcome', axis=1)
    
    #Normalize values
    features[i].iloc[:,:] = Normalizer(norm='l1').fit_transform(features[i])
    up_features[i].iloc[:,:] = Normalizer(norm='l1').fit_transform(up_features[i])
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score

lr_monitor = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", patience=10, factor=0.3, cooldown=1)
    
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                  patience=20,
                                                  restore_best_weights=True,
                                                  mode='min')

def train_nn_model(clf, data, labels, upcoming):
    #Set up Cross Validation
    cv_folds = KFold(n_splits=10, shuffle=False)
    prediction_folds = []
    
    #Set up training for each fold
    for train, test in cv_folds.split(data):
        X_train, X_test = data[data.index.isin(train)], data[data.index.isin(test)]
        y_train, y_test = labels[data.index.isin(train)], labels[data.index.isin(test)]

        clf.fit(X_train, y_train, epochs=250, batch_size=128, verbose=2, callbacks=[lr_monitor, early_stopping])
        
        prediction_folds.append(clf.predict(X_test))
        
    y_predict = prediction_folds[0]
    
    for i in range(1, 10):
        y_predict = np.append(y_predict, prediction_folds[i], axis=0)

    return y_predict, clf.predict(upcoming)


def convert_predictions(clf, data, outcomes, upcoming):
    #Encoder for transformations
    encoder = LabelEncoder()
    y_outcomes = encoder.fit_transform(outcomes)
    y_outcomes = tf.keras.utils.to_categorical(y_outcomes)
    
    #Get predictions
    y_predict, upcoming_predict = train_nn_model(clf, data, y_outcomes, upcoming)
    
    #Normalize values
    y_predict_reverse = [np.argmax(y, axis=None, out=None) for y in y_predict]
    y_predict_decoded = encoder.inverse_transform(y_predict_reverse)
    
    upcoming_predict_reverse = [np.argmax(y, axis=None, out=None) for y in upcoming_predict]
    upcoming_predict_decoded = encoder.inverse_transform(upcoming_predict_reverse)
    return outcomes[:(len(y_predict_decoded) - len(outcomes))], y_predict_decoded, upcoming_predict_decoded


def prediction_metrics(y_test, y_predict):
    #Labels for each result
    display_ls = ['Home Team Win', 'Draw', 'Away Team Win']
    
    #Create confusion matrix to evaluate accuracy
    confusion_m = confusion_matrix(y_test, y_predict, labels= display_ls)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_m, display_labels=display_ls)
    disp.plot(include_values=True, values_format='d')
    plt.show()
    
    print(classification_report(y_test, y_predict, target_names=display_ls))
    print("\nAccuracy: ", accuracy_score(y_test, y_predict))
    print("Recall: ", recall_score(y_test, y_predict, average='weighted'))
    print("Precision: ", precision_score(y_test, y_predict, average='weighted', zero_division=1))
    print("\n")
    print("----------------------------------------------------------")
clf_title = ["B365","BW","IW","LB"]

result_outcomes = [None] * len(features)
result_y_predict_decoded = [None] * len(features)
upcoming_predict_result = [None] * len(up_features)

#Train module with linear Neural Network
for i in range(len(features)):
    col_features = list(features[i].columns.values)
    features_selected = features[i][col_features].copy(deep=True)
    features_be_predicted = up_features[i][col_features].copy(deep=True)
    
    visible = tf.keras.layers.Input(shape=(features_selected.shape[1]))
    hidden1 = tf.keras.layers.Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(visible)
    output = tf.keras.layers.Dense(3, activation='softmax')(hidden1)

    clf = tf.keras.Model(inputs=visible, outputs=output)
    print("Predictions for " + clf_title[i])

    clf.compile(optimizer=tf.keras.optimizers.Adam(), 
                loss=tf.keras.losses.CategoricalCrossentropy(), 
                metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    result_outcomes[i],result_y_predict_decoded[i],upcoming_predict_result[i] = convert_predictions(clf, features_selected, outcomes[i], features_be_predicted)
#Print results
for i in range(len(result_outcomes)):
    print("Predictions for " + clf_title[i])
    prediction_metrics(result_outcomes[i], result_y_predict_decoded[i])
result_outcomes = [None] * len(features)
result_y_predict_decoded = [None] * len(features)
upcoming_predict_result = [None] * len(up_features)

#Train module with Multiple layer Neural Network
for i in range(len(features)):
    col_features = list(features[i].columns.values)
#     del col_features[20:23]
    features_selected = features[i][col_features].copy(deep=True)
    features_be_predicted = up_features[i][col_features].copy(deep=True)
    
    visible = tf.keras.layers.Input(shape=(features_selected.shape[1]))
    hidden1 = tf.keras.layers.Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(visible)
    hidden2 = tf.keras.layers.Dense(200, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(hidden1)
    hidden3 = tf.keras.layers.Dense(300, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(hidden2)
    hidden4 = tf.keras.layers.Dense(200, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(hidden3)
    hidden5 = tf.keras.layers.Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(hidden4)
    output = tf.keras.layers.Dense(3, activation='softmax')(hidden5)

    clf = tf.keras.Model(inputs=visible, outputs=output)
    print("Predictions for " + clf_title[i])

    clf.compile(optimizer=tf.keras.optimizers.Adam(), 
                loss=tf.keras.losses.CategoricalCrossentropy(), 
                metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    result_outcomes[i],result_y_predict_decoded[i],upcoming_predict_result[i] = convert_predictions(clf, features_selected, outcomes[i], features_be_predicted)
#Print results
for i in range(len(result_outcomes)):
    print("Predictions for " + clf_title[i])
    prediction_metrics(result_outcomes[i], result_y_predict_decoded[i])

