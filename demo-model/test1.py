import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
import networkx as nx

# ------------ 1. Load JSON Data ------------ #
with open('new_match_data.json', 'r') as f:
    data = json.load(f)

# ------------ 2. Extract Ball-by-Ball Data ------------ #
innings = data['innings']
rows = []

match_id = 200001  # Dummy match ID for tracking

for inning in innings:
    team = inning['team']
    batting_team = team
    bowling_team = [team for team in data['info']['teams'] if team != batting_team][0]
    overs = inning['overs']

    for over_data in overs:
        over_num = over_data['over']
        for delivery in over_data['deliveries']:
            row = {
                'Match ID': match_id,
                'Venue': data['info']['venue'],
                'Bat First': batting_team,
                'Bat Second': bowling_team,
                'Innings': 1 if batting_team == data['info']['teams'][0] else 2,
                'Over': over_num,
                'Ball': over_data['deliveries'].index(delivery) + 1,
                'Batter': delivery['batter'],
                'Non Striker': delivery['non_striker'],
                'Bowler': delivery['bowler'],
                'Batter Runs': delivery['runs']['batter'],
                'Extra Runs': delivery['runs']['extras'],
                'Runs From Ball': delivery['runs']['total'],
                'Ball Rebowled': 0,
                'Extra Type': 'NA',
                'Wicket': 1 if 'wickets' in delivery else 0,
                'Method': delivery['wickets'][0]['kind'] if 'wickets' in delivery else 'NA',
                'Player Out': delivery['wickets'][0]['player_out'] if 'wickets' in delivery else 'NA',
                'Innings Runs': sum(d['runs']['total'] for d in over_data['deliveries'][:over_data['deliveries'].index(delivery)+1]),
                'Innings Wickets': sum(1 if 'wickets' in d else 0 for d in over_data['deliveries'][:over_data['deliveries'].index(delivery)+1]),
                'Target Score': 0,
                'Runs to Get': 0,
                'Balls Remaining': 0,
                'Winner': data['info']['outcome']['winner'],
                'Chased Successfully': 1 if data['info']['outcome']['winner'] == bowling_team else 0,
                'Total Batter Runs': row['Batter Runs'],
                'Total Non Striker Runs': 0,
                'Batter Balls Faced': 1,
                'Non Striker Balls Faced': 0,
                'Player Out Runs': row['Batter Runs'] if row['Wicket'] else 0,
                'Player Out Balls Faced': 1 if row['Wicket'] else 0,
                'Bowler Runs Conceded': row['Runs From Ball'],
                'Valid Ball': 1
            }

            rows.append(row)

df = pd.DataFrame(rows)

# ------------ 3. Preprocess Data for Model ------------ #
categorical_cols = ['Venue', 'Bat First', 'Bat Second', 'Batter', 'Non Striker', 'Bowler', 'Extra Type', 'Method', 'Player Out', 'Winner']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

numerical_cols = ['Over', 'Ball', 'Batter Runs', 'Extra Runs', 'Runs From Ball', 'Ball Rebowled',
                  'Wicket', 'Innings Runs', 'Innings Wickets', 'Target Score', 'Runs to Get',
                  'Balls Remaining', 'Chased Successfully', 'Total Batter Runs', 'Total Non Striker Runs',
                  'Batter Balls Faced', 'Non Striker Balls Faced', 'Player Out Runs', 'Player Out Balls Faced',
                  'Bowler Runs Conceded', 'Valid Ball']

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# ------------ 4. Create Sequences for Prediction ------------ #
feature_cols = categorical_cols + numerical_cols
sequence_length = 30

X_data = df[feature_cols].values
if len(X_data) < sequence_length:
    padding = np.zeros((sequence_length - len(X_data), len(feature_cols)))
    X_padded = np.vstack((padding, X_data))
else:
    X_padded = X_data[-sequence_length:]

X_input = np.expand_dims(X_padded, axis=0)

# ------------ 5. Load LSTM Model ------------ #
lstm_model = load_model('cricket_momentum_lstm_model.h5')

# ------------ 6. Load GNN Model ------------ #
class GNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

gnn_model = GNNModel(input_size=X_input.shape[2], hidden_size=64, output_size=1)
gnn_model.load_state_dict(torch.load('cricket_momentum_gnn_model.pth'))
gnn_model.eval()

# ------------ 7. Predict Scores ------------ #
lstm_score = lstm_model.predict(X_input)[0][0]

with torch.no_grad():
    gnn_score = gnn_model(torch.tensor(X_input, dtype=torch.float32)).item()

# Combined Momentum Score
momentum_score_mi = (0.6 * lstm_score) + (0.4 * gnn_score)
momentum_score_csk = 100 - momentum_score_mi

# ------------ 8. Display Results ------------ #
print(f"MI Momentum Score: {momentum_score_mi:.2f}")
print(f"CSK Momentum Score: {momentum_score_csk:.2f}")

# ------------ 9. Key Factors Analysis ------------ #
key_factors_mi = [
    "Strong batting partnerships",
    "High run rate maintained",
    "Key bowler struggles to contain runs"
]

key_factors_csk = [
    "Bowler's success in limiting runs",
    "Frequent wickets taken",
    "Pressure situations created by dot balls"
]

# Suggested Actions
actions_mi = [
    "Focus on strike rotation",
    "Target weaker bowlers",
    "Accelerate scoring in death overs"
]

actions_csk = [
    "Maintain dot ball pressure",
    "Use spinners to build pressure",
    "Place aggressive fielders for better wicket chances"
]

# ------------ 10. Display Insights ------------ #
print("\nKey Factors for MI Momentum Shift:")
print("\n".join(key_factors_mi))

print("\nWhat MI Can Do to Improve:")
print("\n".join(actions_mi))

print("\nKey Factors for CSK Momentum Shift:")
print("\n".join(key_factors_csk))

print("\nWhat CSK Can Do to Improve:")
print("\n".join(actions_csk))
