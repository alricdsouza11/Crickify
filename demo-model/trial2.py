import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load Dataset
df = pd.read_csv('ball_by_ball_ipl.csv')

df.fillna({'Extra Type': 'None', 'Wicket': 0, 'Method': 'None', 'Player Out': 'None'}, inplace=True)

# Encode Categorical Variables
label_encoders = {}
categorical_columns = ['Venue', 'Bat First', 'Bat Second', 'Batter', 'Non Striker', 'Bowler', 'Extra Type', 'Method', 'Player Out', 'Winner']
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Feature Engineering
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by=['Match ID', 'Innings', 'Over', 'Ball'], inplace=True)

df['Cumulative Runs'] = df.groupby(['Match ID', 'Innings'])['Batter Runs'].cumsum()
df['Cumulative Wickets'] = df.groupby(['Match ID', 'Innings'])['Wicket'].cumsum()
df['Balls Bowled'] = df.groupby(['Match ID', 'Innings']).cumcount() + 1
df['Run Rate'] = df['Cumulative Runs'] / (df['Balls Bowled'] / 6)
df['Remaining Runs'] = df['Target Score'] - df['Cumulative Runs']
df['Remaining Overs'] = df['Balls Remaining'] / 6
df['Required Run Rate'] = df['Remaining Runs'] / df['Remaining Overs']
df.replace([np.inf, -np.inf], 0, inplace=True)

def calculate_momentum_score(row):
    run_rate_diff = row['Run Rate'] - row['Required Run Rate']
    momentum = (row['Batter Runs'] - (row['Wicket'] * 5)) + run_rate_diff
    return momentum

df['Momentum Score'] = df.apply(calculate_momentum_score, axis=1)

# Feature Selection and Scaling
features = ['Over', 'Ball', 'Batter', 'Non Striker', 'Bowler', 'Batter Runs', 'Extra Runs', 'Wicket',
            'Cumulative Runs', 'Cumulative Wickets', 'Run Rate', 'Required Run Rate', 'Balls Remaining']

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Sequence Data Preparation
X = df[features].values
y = df['Momentum Score'].values

X_sequences = []
y_sequences = []
sequence_length = 30

for match_id in df['Match ID'].unique():
    match_data = df[df['Match ID'] == match_id]
    for i in range(len(match_data) - sequence_length):
        X_sequences.append(match_data[features].iloc[i:i + sequence_length].values)
        y_sequences.append(match_data['Momentum Score'].iloc[i + sequence_length])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

# LSTM Model
def build_lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, len(features))))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

lstm_model = build_lstm_model()

lstm_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)
lstm_model.save('momentum_model.h5')

print("Model trained and saved successfully!")
