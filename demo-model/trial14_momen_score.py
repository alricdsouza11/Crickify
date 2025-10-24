import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ----------- 1. Load Data ----------- #
df = pd.read_csv('ball_by_ball_ipl.csv')

# ----------- 2. Handle Missing Values ----------- #
df.fillna({'Extra Type': 'None', 'Wicket': 0, 'Method': 'None', 'Player Out': 'None'}, inplace=True)

# ----------- 3. Label Encoding ----------- #
label_encoders = {}
categorical_columns = ['Venue', 'Bat First', 'Bat Second', 'Batter', 'Non Striker', 'Bowler',
                       'Extra Type', 'Method', 'Player Out', 'Winner']

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le
    joblib.dump(le, f'label_encoder_{column}.pkl')  # Save encoders

# ----------- 4. Feature Engineering ----------- #
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

# ----------- 5. Scaling ----------- #
features = ['Over', 'Ball', 'Batter', 'Non Striker', 'Bowler', 'Batter Runs', 'Extra Runs', 'Wicket',
            'Cumulative Runs', 'Cumulative Wickets', 'Run Rate', 'Required Run Rate', 'Balls Remaining']

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
joblib.dump(scaler, 'scaler.pkl')

# ----------- 6. Sequence Preparation ----------- #
X = df[features].values
y = df['Momentum Score'].values

sequence_length = 30
X_sequences = []
y_sequences = []

for match_id in df['Match ID'].unique():
    match_data = df[df['Match ID'] == match_id]
    for i in range(len(match_data) - sequence_length):
        X_sequences.append(match_data[features].iloc[i:i+sequence_length].values)
        y_sequences.append(match_data['Momentum Score'].iloc[i+sequence_length])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# ----------- 7. Train-Test Split ----------- #
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)

# ----------- 8. Model Building ----------- #
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, len(features))))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ----------- 9. Model Training ----------- #
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# ----------- 10. Evaluation ----------- #
test_loss = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

print(f'Test Loss: {test_loss}')
print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
print(f'Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred))}')
print(f'R^2 Score: {r2_score(y_test, y_pred)}')

# ----------- 11. Save Model ----------- #
model.save('cricket_momentum_lstm_model.h5')
