from __future__ import division
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('filename',type=str)
parser.add_argument('output',type=str)
csvfile = parser.parse_args()

data = pd.read_csv(csvfile.filename)

columns = ['player_id','player','game_id', 'event_type_id', 'event_type', 'period_id', 'period_min', 'period_second',
           'outcome', 'team_id', 'x', 'y', 'cross', 'free_kick_taken', 'corner_taken',
           'pass_end_x', 'pass_end_y', 'regular_play','team','score']

data = data[columns]
action_types = [1, 3, 13, 14, 15, 16]
fdata = data[np.isin(data.event_type_id, action_types)]
fdata = fdata[fdata.x >= (75 / 115) * 100]
fdata = fdata[fdata.free_kick_taken.isnull()]
fdata = fdata[fdata.corner_taken.isnull()]
#fdata = fdata[fdata.player_id == 179018]

attacking = fdata[fdata.x >= 75 / 115 * 100]
attacking.groupby('event_type').size()
attacking.loc[np.isin(attacking.event_type, ["Attempt saved", "Goal", "Miss", "Post"]),
              "event_type"] = "Shot"
attacking.groupby('event_type').size()

attacking = attacking.copy()
attacking['x_real'] = attacking.x / 100 * 115
attacking['y_real'] = attacking.y / 100 * 80
#attacking['player_id'] = np.ones(len(attacking.player_id),dtype=int)

#X = attacking[['player_id','player','x_real', 'y_real','event_type','score']]
X = attacking[['team_id','team','x_real', 'y_real','event_type','score']]

#X, toss = train_test_split(X, test_size=0.8)
X_train, X_test = train_test_split(X, test_size=0.4)
X_test, X_val= train_test_split(X_test, test_size=0.5, random_state=1)

X_train = X_train.assign(train=pd.DataFrame(np.ones((X_train.shape[0],1),dtype=int),index=X_train.index))
X_train = X_train.assign(test=pd.DataFrame(np.zeros((X_train.shape[0],1),dtype=int),index=X_train.index))
X_train = X_train.assign(validate=pd.DataFrame(np.zeros((X_train.shape[0],1),dtype=int),index=X_train.index))

X_test = X_test.assign(train=pd.DataFrame(np.zeros((X_test.shape[0],1),dtype=int),index=X_test.index))
X_test = X_test.assign(test=pd.DataFrame(np.ones((X_test.shape[0],1),dtype=int),index=X_test.index))
X_test = X_test.assign(validate=pd.DataFrame(np.zeros((X_test.shape[0],1),dtype=int),index=X_test.index))

X_val = X_val.assign(train=pd.DataFrame(np.zeros((X_val.shape[0],1),dtype=int),index=X_val.index))
X_val = X_val.assign(test=pd.DataFrame(np.zeros((X_val.shape[0],1),dtype=int),index=X_val.index))
X_val = X_val.assign(validate=pd.DataFrame(np.ones((X_val.shape[0],1),dtype=int),index=X_val.index))


output = pd.concat([X_train,X_test,X_val])

output.to_csv(csvfile.output,index=False)
