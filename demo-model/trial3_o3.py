"""# model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Lambda, Input, Concatenate, Embedding, Flatten
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

def create_sequences(df, seq_length, ball_features, cum_features, ctx_features):
    X_seq, X_cum, X_ctx, y_target = [], [], [], []
    for i in range(len(df) - seq_length):
        seq = df.iloc[i:i+seq_length][ball_features].values
        X_seq.append(seq)
        X_cum.append(df.iloc[i+seq_length][cum_features].values)
        X_ctx.append(df.iloc[i+seq_length][ctx_features].values)
        y_target.append(df.iloc[i+seq_length]['Target Momentum'])
    # Convert arrays to float32 to ensure numeric types
    return (np.array(X_seq, dtype=np.float32),
            np.array(X_cum, dtype=np.float32),
            np.array(X_ctx, dtype=np.float32),
            np.array(y_target, dtype=np.float32))

def load_and_preprocess_data(csv_path, seq_length=10):
    # Load data and sort by Over and Ball.
    data = pd.read_csv(csv_path)
    data.sort_values(by=['Over', 'Ball'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Define the contextual numeric columns we expect.
    ctx_numeric = ['Batter_Historical_Avg', 'Bowler_Historical_Economy', 
                   'Batter_vs_Bowler_Avg', 'Team_Powerplay_Performance', 'Match_Phase']
    # Add missing contextual numeric columns with default value 0.
    for col in ctx_numeric:
        if col not in data.columns:
            data[col] = 0
    
    # Compute cumulative match features.
    data['Cumulative Runs'] = data['Runs From Ball'].cumsum()
    data['Cumulative Wickets'] = data['Wicket'].astype(float).cumsum()
    data['Balls Bowled'] = np.arange(1, len(data) + 1)
    data['Overs Completed'] = data['Balls Bowled'] / 6.0
    data['Current Run Rate'] = data['Cumulative Runs'] / data['Overs Completed']
    
    # Compute an example target momentum.
    # Example: base 50, bonus for run rate above 7, penalty for wickets.
    data['Target Momentum'] = 50 + (data['Current Run Rate'] - 7) * 5 - (data['Cumulative Wickets'] * 5)
    data['Target Momentum'] = data['Target Momentum'].clip(0, 100)
    
    # Define feature groups.
    ball_features = ['Batter Runs', 'Extra Runs', 'Runs From Ball', 'Balls Remaining']
    cum_features = ['Cumulative Runs', 'Cumulative Wickets', 'Current Run Rate']
    # Use "Venue" as the contextual categorical feature.
    # The contextual features list: first element is the encoded Venue, then the numeric factors.
    ctx_features = ['Venue_enc'] + ctx_numeric
    
    # Preprocess ball-level features.
    ball_scaler = StandardScaler()
    data[ball_features] = ball_scaler.fit_transform(data[ball_features])
    
    # Preprocess cumulative features.
    cum_scaler = StandardScaler()
    data[cum_features] = cum_scaler.fit_transform(data[cum_features])
    
    # Process contextual features.
    # Use the column "Venue" instead of "Stadium"
    if 'Venue' not in data.columns:
        data['Venue'] = "Unknown"
    le = LabelEncoder()
    data['Venue_enc'] = le.fit_transform(data['Venue'])
    
    # Scale the contextual numeric features.
    ctx_scaler = StandardScaler()
    data[ctx_numeric] = ctx_scaler.fit_transform(data[ctx_numeric])
    
    # Drop any rows with missing values.
    data = data.dropna(subset=ball_features + cum_features + ctx_features + ['Target Momentum'])
    
    # Create training sequences.
    X_seq, X_cum, X_ctx, y_target = create_sequences(data, seq_length, ball_features, cum_features, ctx_features)
    return X_seq, X_cum, X_ctx, y_target, ball_scaler, cum_scaler, ctx_scaler, le, ball_features, cum_features, ctx_features

def build_model(seq_length, num_ball_features, num_cum_features, num_ctx_numeric, vocab_size, ctx_embed_dim=4):
    # Branch 1: Ball-level sequence features (LSTM branch)
    input_seq = Input(shape=(seq_length, num_ball_features), name="seq_input")
    x = LSTM(64, return_sequences=True)(input_seq)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    
    # Branch 2: Cumulative match state features.
    input_cum = Input(shape=(num_cum_features,), name="cum_input")
    y_branch = Dense(8, activation='relu')(input_cum)
    y_branch = Dense(4, activation='relu')(y_branch)
    
    # Branch 3a: Contextual categorical feature (Venue)
    input_ctx_cat = Input(shape=(1,), dtype='int32', name="ctx_cat_input")
    z_cat = Embedding(input_dim=vocab_size, output_dim=ctx_embed_dim)(input_ctx_cat)
    z_cat = Flatten()(z_cat)
    
    # Branch 3b: Contextual numeric features (remaining factors)
    input_ctx_num = Input(shape=(num_ctx_numeric,), name="ctx_num_input")
    z_num = Dense(8, activation='relu')(input_ctx_num)
    z_num = Dense(4, activation='relu')(z_num)
    
    # Merge the contextual branches.
    ctx_combined = Concatenate()([z_cat, z_num])
    ctx_out = Dense(8, activation='relu')(ctx_combined)
    
    # Combine all branches.
    combined = Concatenate()([x, y_branch, ctx_out])
    z = Dense(8, activation='relu')(combined)
    # Final output: apply sigmoid and scale to [0, 100].
    output = Dense(1, activation='sigmoid')(z)
    output = Lambda(lambda x: x * 100)(output)
    
    model = Model(inputs=[input_seq, input_cum, input_ctx_cat, input_ctx_num], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(csv_path, seq_length=10, epochs=20, batch_size=32):
    (X_seq, X_cum, X_ctx, y_target,
     ball_scaler, cum_scaler, ctx_scaler, le,
     ball_features, cum_features, ctx_features) = load_and_preprocess_data(csv_path, seq_length)
    
    # Split the data into training and validation sets.
    X_seq_train, X_seq_val, X_cum_train, X_cum_val, X_ctx_train, X_ctx_val, y_train, y_val = train_test_split(
        X_seq, X_cum, X_ctx, y_target, test_size=0.2, random_state=42)
    
    num_ball_features = X_seq.shape[-1]
    num_cum_features = X_cum.shape[-1]
    # For contextual numeric input, exclude the first element (encoded Venue).
    num_ctx_numeric = len(ctx_features) - 1  
    vocab_size = len(le.classes_)
    
    model = build_model(seq_length, num_ball_features, num_cum_features, num_ctx_numeric, vocab_size)
    model.summary()
    
    history = model.fit([X_seq_train, X_cum_train, X_ctx_train[:, 0].reshape(-1,1), X_ctx_train[:, 1:]],
                        y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=([X_seq_val, X_cum_val, X_ctx_val[:, 0].reshape(-1,1), X_ctx_val[:, 1:]], y_val))
    
    model.save('momentum_model.h5')
    with open('ball_scaler.pkl', 'wb') as f:
        pickle.dump(ball_scaler, f)
    with open('cum_scaler.pkl', 'wb') as f:
        pickle.dump(cum_scaler, f)
    with open('ctx_scaler.pkl', 'wb') as f:
        pickle.dump(ctx_scaler, f)
    with open('stadium_le.pkl', 'wb') as f:
        # Note: even though the variable is named "stadium_le", it encodes Venue.
        pickle.dump(le, f)
    with open('ball_features.pkl', 'wb') as f:
        pickle.dump(ball_features, f)
    with open('cum_features.pkl', 'wb') as f:
        pickle.dump(cum_features, f)
    with open('ctx_features.pkl', 'wb') as f:
        pickle.dump(ctx_features, f)
    
    print("Training complete. Model and preprocessing objects saved.")
    return model, ball_scaler, cum_scaler, ctx_scaler, le, ball_features, cum_features, ctx_features

if __name__ == "__main__":
    csv_path = 'ball_by_ball_ipl.csv'  # Update this to the path of your CSV file.
    train_model(csv_path, seq_length=10, epochs=20, batch_size=32)
"""
# model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Lambda, Input, Concatenate, Embedding, Flatten
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

TOTAL_BALLS = 120  # For a T20 match

def load_and_preprocess_data(csv_path, seq_length=10, total_balls=TOTAL_BALLS):
    """
    Loads historical ball-by-ball data from csv_path.
    Computes cumulative features and (if available) chase-specific features based on innings.
    Ensures that contextual numeric columns exist; if missing, adds them with default 0.
    Computes:
      - Cumulative Runs, Cumulative Wickets, Overs Completed, Current Run Rate.
      - If the row is in innings 2 (chasing), computes:
            Balls Remaining, Required Run Rate, Chase Differential.
      - Also creates an "Is_Chasing" flag.
    Returns the processed DataFrame along with lists of feature names and fitted scalers.
    """
    data = pd.read_csv(csv_path)
    data.sort_values(by=['Innings', 'Over', 'Ball'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Define expected contextual numeric columns.
    ctx_numeric = ['Batter_Historical_Avg', 'Bowler_Historical_Economy', 
                   'Batter_vs_Bowler_Avg', 'Team_Powerplay_Performance', 'Match_Phase']
    for col in ctx_numeric:
        if col not in data.columns:
            data[col] = 0  # default value

    # Compute cumulative match features.
    data['Cumulative Runs'] = data['Runs From Ball'].cumsum()
    data['Cumulative Wickets'] = data['Wicket'].astype(float).cumsum()
    data['Balls Bowled'] = np.arange(1, len(data) + 1)
    data['Overs Completed'] = data['Balls Bowled'] / 6.0
    data['Current Run Rate'] = data['Cumulative Runs'] / data['Overs Completed']
    
    # Compute chase-specific features.
    # Assumes that the column 'Innings' exists (1 for setting, 2 for chasing)
    # And that 'Target Score' is provided.
    def compute_chase_features(row):
        if row['Innings'] == 2:
            balls_remaining = total_balls - row['Balls Bowled']
            # Avoid division by zero.
            overs_remaining = balls_remaining / 6.0 if balls_remaining > 0 else 0.1
            req_rr = (row['Target Score'] - row['Cumulative Runs']) / overs_remaining
            chase_diff = row['Current Run Rate'] - req_rr
            return pd.Series([1, req_rr, chase_diff])
        else:
            return pd.Series([0, 0, 0])
    
    data[['Is_Chasing', 'Required Run Rate', 'Chase Differential']] = data.apply(compute_chase_features, axis=1)
    
    # Define feature groups.
    ball_features = ['Batter Runs', 'Extra Runs', 'Runs From Ball', 'Balls Remaining']
    cum_features = ['Cumulative Runs', 'Cumulative Wickets', 'Current Run Rate']
    ctx_features = ['Venue_enc'] + ctx_numeric  # contextual: first is encoded venue.
    chase_features = ['Is_Chasing', 'Required Run Rate', 'Chase Differential']
    
    # For contextual categorical feature, use "Venue" (if missing, set to "Unknown").
    if 'Venue' not in data.columns:
        data['Venue'] = "Unknown"
    le = LabelEncoder()
    data['Venue_enc'] = le.fit_transform(data['Venue'])
    
    # Scale numeric groups.
    ball_scaler = StandardScaler()
    data[ball_features] = ball_scaler.fit_transform(data[ball_features])
    
    cum_scaler = StandardScaler()
    data[cum_features] = cum_scaler.fit_transform(data[cum_features])
    
    ctx_scaler = StandardScaler()
    data[ctx_numeric] = ctx_scaler.fit_transform(data[ctx_numeric])
    
    chase_scaler = StandardScaler()
    data[chase_features] = chase_scaler.fit_transform(data[chase_features])
    
    # Compute a target momentum.
    # Example: base 50, add bonus for current run rate above 7, subtract 5 per wicket, and incorporate chase diff.
    data['Target Momentum'] = 50 + (data['Current Run Rate'] - 7)*5 - (data['Cumulative Wickets']*5) + (data['Chase Differential']*2)
    data['Target Momentum'] = data['Target Momentum'].clip(0, 100)
    
    # Drop rows with missing values in our feature groups.
    all_features = ball_features + cum_features + ctx_features + chase_features + ['Target Momentum']
    data = data.dropna(subset=all_features)
    
    return data, ball_features, cum_features, ctx_features, chase_features, ball_scaler, cum_scaler, ctx_scaler, chase_scaler, le

def create_sequences(data, seq_length, ball_features, cum_features, ctx_features, chase_features):
    """
    For each sample, use the previous seq_length balls as sequence input,
    and use the next ball’s cumulative, contextual, and chase features as separate inputs.
    """
    X_seq, X_cum, X_ctx, X_chase, y_target = [], [], [], [], []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length][ball_features].values
        X_seq.append(seq)
        X_cum.append(data.iloc[i+seq_length][cum_features].values)
        X_ctx.append(data.iloc[i+seq_length][ctx_features].values)
        X_chase.append(data.iloc[i+seq_length][chase_features].values)
        y_target.append(data.iloc[i+seq_length]['Target Momentum'])
    return (np.array(X_seq, dtype=np.float32),
            np.array(X_cum, dtype=np.float32),
            np.array(X_ctx, dtype=np.float32),
            np.array(X_chase, dtype=np.float32),
            np.array(y_target, dtype=np.float32))

def train_momentum_model(csv_path, seq_length=10, epochs=10, batch_size=32, total_balls=TOTAL_BALLS):
    data, ball_feats, cum_feats, ctx_feats, chase_feats, ball_scaler, cum_scaler, ctx_scaler, chase_scaler, le = load_and_preprocess_data(csv_path, seq_length, total_balls)
    
    X_seq, X_cum, X_ctx, X_chase, y_target = create_sequences(data, seq_length, ball_feats, cum_feats, ctx_feats, chase_feats)
    
    # Split into training and validation sets.
    (X_seq_train, X_seq_val,
     X_cum_train, X_cum_val,
     X_ctx_train, X_ctx_val,
     X_chase_train, X_chase_val,
     y_train, y_val) = train_test_split(X_seq, X_cum, X_ctx, X_chase, y_target, test_size=0.2, random_state=42)
    
    num_ball_features = len(ball_feats)
    num_cum_features = len(cum_feats)
    num_ctx_numeric = len(ctx_feats) - 1  # excluding the encoded Venue
    num_chase_features = len(chase_feats)
    vocab_size = len(le.classes_)
    
    # Build the model with four inputs.
    # Input 1: Ball-level sequence.
    input_seq = Input(shape=(seq_length, num_ball_features), name="seq_input")
    x = LSTM(64, return_sequences=True)(input_seq)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    
    # Input 2: Cumulative features.
    input_cum = Input(shape=(num_cum_features,), name="cum_input")
    y_branch = Dense(8, activation='relu')(input_cum)
    y_branch = Dense(4, activation='relu')(y_branch)
    
    # Input 3: Contextual features.
    input_ctx_cat = Input(shape=(1,), dtype='int32', name="ctx_cat_input")
    z_cat = Embedding(input_dim=vocab_size, output_dim=4)(input_ctx_cat)
    z_cat = Flatten()(z_cat)
    input_ctx_num = Input(shape=(num_ctx_numeric,), name="ctx_num_input")
    z_num = Dense(8, activation='relu')(input_ctx_num)
    z_num = Dense(4, activation='relu')(z_num)
    ctx_combined = Concatenate()([z_cat, z_num])
    ctx_out = Dense(8, activation='relu')(ctx_combined)
    
    # Input 4: Chase features.
    input_chase = Input(shape=(num_chase_features,), name="chase_input")
    c = Dense(8, activation='relu')(input_chase)
    c = Dense(4, activation='relu')(c)
    
    # Combine all branches.
    combined = Concatenate()([x, y_branch, ctx_out, c])
    z = Dense(8, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(z)
    output = Lambda(lambda t: t * 100)(output)
    
    model = Model(inputs=[input_seq, input_cum, input_ctx_cat, input_ctx_num, input_chase], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    # Train the model.
    model.fit([X_seq_train, X_cum_train, X_ctx_train[:, 0].reshape(-1,1), X_ctx_train[:, 1:], X_chase_train],
              y_train, epochs=epochs, batch_size=batch_size, validation_data=(
                  [X_seq_val, X_cum_val, X_ctx_val[:, 0].reshape(-1,1), X_ctx_val[:, 1:], X_chase_val], y_val))
    
    # Save model and preprocessors.
    model.save('momentum_model.h5')
    with open('ball_scaler.pkl', 'wb') as f:
        pickle.dump(ball_scaler, f)
    with open('cum_scaler.pkl', 'wb') as f:
        pickle.dump(cum_scaler, f)
    with open('ctx_scaler.pkl', 'wb') as f:
        pickle.dump(ctx_scaler, f)
    with open('chase_scaler.pkl', 'wb') as f:
        pickle.dump(chase_scaler, f)
    with open('venue_le.pkl', 'wb') as f:
        pickle.dump(le, f)
    with open('ball_features.pkl', 'wb') as f:
        pickle.dump(ball_feats, f)
    with open('cum_features.pkl', 'wb') as f:
        pickle.dump(cum_feats, f)
    with open('ctx_features.pkl', 'wb') as f:
        pickle.dump(ctx_feats, f)
    with open('chase_features.pkl', 'wb') as f:
        pickle.dump(chase_feats, f)
    
    print("Training complete. Model and preprocessors saved.")
    return model, ball_scaler, cum_scaler, ctx_scaler, chase_scaler, le, ball_feats, cum_feats, ctx_feats, chase_feats

if __name__ == "__main__":
    historical_csv = "ball_by_ball_ipl.csv"  # Historical ball-by-ball data
    train_momentum_model(historical_csv, seq_length=10, epochs=10, batch_size=32)
