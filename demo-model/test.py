# full_pipeline.py

import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Lambda, Input, Concatenate, Embedding, Flatten
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Global constant for a T20 match
TOTAL_BALLS = 120

#########################################
#       TRAINING FUNCTIONS              #
#########################################

def load_and_preprocess_data(csv_path, seq_length=10, total_balls=TOTAL_BALLS):
    """
    Loads historical ball-by-ball data, computes cumulative and chase-specific features,
    and ensures that contextual numeric columns exist.
    """
    data = pd.read_csv(csv_path)
    if 'Innings' in data.columns:
        data.sort_values(by=['Innings', 'Over', 'Ball'], inplace=True)
    else:
        data.sort_values(by=['Over', 'Ball'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Ensure expected contextual numeric columns exist.
    ctx_numeric = ['Batter_Historical_Avg', 'Bowler_Historical_Economy', 
                   'Batter_vs_Bowler_Avg', 'Team_Powerplay_Performance', 'Match_Phase']
    for col in ctx_numeric:
        if col not in data.columns:
            data[col] = 0  # default value

    # Compute cumulative features.
    data['Cumulative Runs'] = data['Runs From Ball'].cumsum()
    data['Cumulative Wickets'] = data['Wicket'].astype(float).cumsum()
    data['Balls Bowled'] = np.arange(1, len(data) + 1)
    data['Overs Completed'] = data['Balls Bowled'] / 6.0
    data['Current Run Rate'] = data['Cumulative Runs'] / data['Overs Completed']
    
    # Compute chase-specific features (if chasing; assume Innings==2).
    def compute_chase(row):
        if row.get('Innings', 1) == 2:
            balls_remaining = total_balls - row['Balls Bowled']
            overs_remaining = balls_remaining / 6.0 if balls_remaining > 0 else 0.1
            req_rr = (row['Target Score'] - row['Cumulative Runs']) / overs_remaining
            chase_diff = row['Current Run Rate'] - req_rr
            return pd.Series([1, req_rr, chase_diff])
        else:
            return pd.Series([0, 0, 0])
    data[['Is_Chasing', 'Required Run Rate', 'Chase Differential']] = data.apply(compute_chase, axis=1)
    
    # Define feature groups.
    ball_features = ['Batter Runs', 'Extra Runs', 'Runs From Ball', 'Balls Remaining']
    cum_features = ['Cumulative Runs', 'Cumulative Wickets', 'Current Run Rate']
    ctx_features = ['Venue_enc'] + ctx_numeric   # contextual: first is encoded Venue
    chase_features = ['Is_Chasing', 'Required Run Rate', 'Chase Differential']
    
    # Process categorical feature "Venue".
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
    
    # Compute target momentum.
    data['Target Momentum'] = 50 + (data['Current Run Rate'] - 7)*5 - (data['Cumulative Wickets']*5) + (data['Chase Differential']*2)
    data['Target Momentum'] = data['Target Momentum'].clip(0, 100)
    
    # Drop rows with missing values in all relevant features.
    all_features = ball_features + cum_features + ctx_features + chase_features + ['Target Momentum']
    data = data.dropna(subset=all_features)
    
    return data, ball_features, cum_features, ctx_features, chase_features, ball_scaler, cum_scaler, ctx_scaler, chase_scaler, le

def create_sequences(data, seq_length, ball_features, cum_features, ctx_features, chase_features):
    """
    Creates training samples:
      - X_seq: Sequence of ball-level features (previous seq_length balls)
      - X_cum: Cumulative features from the next ball
      - X_ctx: Contextual features from the next ball
      - X_chase: Chase features from the next ball
      - y_target: Target Momentum for the next ball
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
    """
    Trains the momentum prediction model on historical data and saves the model and preprocessors.
    """
    data, ball_feats, cum_feats, ctx_feats, chase_feats, ball_scaler, cum_scaler, ctx_scaler, chase_scaler, le = load_and_preprocess_data(csv_path, seq_length, total_balls)
    X_seq, X_cum, X_ctx, X_chase, y_target = create_sequences(data, seq_length, ball_feats, cum_feats, ctx_feats, chase_feats)
    
    (X_seq_train, X_seq_val,
     X_cum_train, X_cum_val,
     X_ctx_train, X_ctx_val,
     X_chase_train, X_chase_val,
     y_train, y_val) = train_test_split(X_seq, X_cum, X_ctx, X_chase, y_target, test_size=0.2, random_state=42)
    
    num_ball_features = len(ball_feats)
    num_cum_features = len(cum_feats)
    num_ctx_numeric = len(ctx_feats) - 1  # excluding encoded Venue
    num_chase_features = len(chase_feats)
    vocab_size = len(le.classes_)
    
    # Build model.
    input_seq = Input(shape=(seq_length, num_ball_features), name="seq_input")
    x = LSTM(64, return_sequences=True)(input_seq)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    
    input_cum = Input(shape=(num_cum_features,), name="cum_input")
    y_branch = Dense(8, activation='relu')(input_cum)
    y_branch = Dense(4, activation='relu')(y_branch)
    
    input_ctx_cat = Input(shape=(1,), dtype='int32', name="ctx_cat_input")
    z_cat = Embedding(input_dim=vocab_size, output_dim=4)(input_ctx_cat)
    z_cat = Flatten()(z_cat)
    input_ctx_num = Input(shape=(num_ctx_numeric,), name="ctx_num_input")
    z_num = Dense(8, activation='relu')(input_ctx_num)
    z_num = Dense(4, activation='relu')(z_num)
    ctx_combined = Concatenate()([z_cat, z_num])
    ctx_out = Dense(8, activation='relu')(ctx_combined)
    
    input_chase = Input(shape=(num_chase_features,), name="chase_input")
    c = Dense(8, activation='relu')(input_chase)
    c = Dense(4, activation='relu')(c)
    
    combined = Concatenate()([x, y_branch, ctx_out, c])
    z = Dense(8, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(z)
    output = Lambda(lambda t: t * 100)(output)
    
    model = Model(inputs=[input_seq, input_cum, input_ctx_cat, input_ctx_num, input_chase], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    model.fit([X_seq_train, X_cum_train, X_ctx_train[:, 0].reshape(-1,1), X_ctx_train[:, 1:], X_chase_train],
              y_train, epochs=epochs, batch_size=batch_size,
              validation_data=([X_seq_val, X_cum_val, X_ctx_val[:, 0].reshape(-1,1), X_ctx_val[:, 1:], X_chase_val], y_val))
    
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

#########################################
#       LIVE MATCH REPORT FUNCTIONS     #
#########################################

def compute_live_features(df, total_balls=TOTAL_BALLS):
    """Compute cumulative and chase-specific features for live match data."""
    df = df.copy()
    if 'Innings' in df.columns:
        df.sort_values(by=['Innings', 'Over', 'Ball'], inplace=True)
    else:
        df.sort_values(by=['Over', 'Ball'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Cumulative Runs'] = df['Runs From Ball'].cumsum()
    df['Cumulative Wickets'] = df['Wicket'].astype(float).cumsum()
    df['Balls Bowled'] = np.arange(1, len(df) + 1)
    df['Overs Completed'] = df['Balls Bowled'] / 6.0
    df['Current Run Rate'] = df['Cumulative Runs'] / df['Overs Completed']
    
    def compute_chase(row):
        if row.get('Innings', 1) == 2:
            balls_remaining = total_balls - row['Balls Bowled']
            overs_remaining = balls_remaining / 6.0 if balls_remaining > 0 else 0.1
            req_rr = (row['Target Score'] - row['Cumulative Runs']) / overs_remaining
            chase_diff = row['Current Run Rate'] - req_rr
            return pd.Series([1, req_rr, chase_diff])
        else:
            return pd.Series([0, 0, 0])
    df[['Is_Chasing', 'Required Run Rate', 'Chase Differential']] = df.apply(compute_chase, axis=1)
    return df

def get_venue_encoded(venue_val, le):
    if venue_val in le.classes_:
        return le.transform([venue_val])[0]
    else:
        print(f"Warning: Venue '{venue_val}' not seen during training. Using default encoding (0).")
        return 0

def compute_historical_stats(csv_path):
    """Computes historical batter and bowler aggregates."""
    df = pd.read_csv(csv_path)
    if 'Innings' in df.columns:
        df.sort_values(by=['Innings', 'Over', 'Ball'], inplace=True)
    else:
        df.sort_values(by=['Over', 'Ball'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    batter_stats = df.groupby('Batter').agg({
        'Batter Runs': ['sum', 'count'],
        'Runs From Ball': 'sum'
    })
    batter_stats.columns = ['Hist_Total_Runs', 'Hist_Balls_Faced', 'Hist_Runs_Contributed']
    batter_stats['Hist_Strike_Rate'] = (batter_stats['Hist_Total_Runs'] / batter_stats['Hist_Balls_Faced'] * 100).round(2)
    
    bowler_stats = df.groupby('Bowler').agg({
        'Runs From Ball': 'sum',
        'Wicket': 'sum',
        'Ball': 'count'
    })
    bowler_stats.rename(columns={'Ball': 'Hist_Balls_Bowled'}, inplace=True)
    bowler_stats['Hist_Overs'] = bowler_stats['Hist_Balls_Bowled'] / 6.0
    bowler_stats['Hist_Economy'] = (bowler_stats['Runs From Ball'] / bowler_stats['Hist_Overs']).round(2)
    bowler_stats['Hist_Wickets'] = bowler_stats['Wicket'].astype(int)
    bowler_stats.drop(columns=['Wicket'], inplace=True)
    return batter_stats, bowler_stats

def predict_momentum_live(df):
    """Predicts momentum from live match data using the last ball's info."""
    with open('ball_scaler.pkl', 'rb') as f:
        ball_scaler = pickle.load(f)
    with open('cum_scaler.pkl', 'rb') as f:
        cum_scaler = pickle.load(f)
    with open('ctx_scaler.pkl', 'rb') as f:
        ctx_scaler = pickle.load(f)
    with open('chase_scaler.pkl', 'rb') as f:
        chase_scaler = pickle.load(f)
    with open('venue_le.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('ball_features.pkl', 'rb') as f:
        ball_features = pickle.load(f)
    with open('cum_features.pkl', 'rb') as f:
        cum_features = pickle.load(f)
    with open('ctx_features.pkl', 'rb') as f:
        ctx_features = pickle.load(f)
    with open('chase_features.pkl', 'rb') as f:
        chase_features = pickle.load(f)
    
    seq_length = 10
    if len(df) < seq_length:
        raise ValueError("Not enough live data to form a sequence.")
    X_seq = df[ball_features].values
    X_seq_scaled = ball_scaler.transform(X_seq)
    sequence = X_seq_scaled[-seq_length:]
    sequence = np.expand_dims(sequence, axis=0)
    
    current_cum = df.iloc[-1][cum_features].values.reshape(1, -1)
    current_cum_scaled = cum_scaler.transform(current_cum)
    
    venue_val = df.iloc[-1]['Venue']
    venue_index = le.transform([venue_val])[0] if venue_val in le.classes_ else 0
    venue_enc = np.array([[venue_index]])
    
    ctx_numeric = df.iloc[-1][ctx_features[1:]].values.reshape(1, -1)
    ctx_numeric_scaled = ctx_scaler.transform(ctx_numeric)
    
    chase_vals = df.iloc[-1][chase_features].values.reshape(1, -1)
    chase_scaled = chase_scaler.transform(chase_vals)
    
    model = load_model('momentum_model.h5', custom_objects={'mse': 'mse'})
    predicted = model.predict([sequence, current_cum_scaled, venue_enc, ctx_numeric_scaled, chase_scaled])
    return predicted[0][0]

def generate_live_report(live_csv, historical_csv, seq_length, ball_scaler, cum_scaler, ctx_scaler, chase_scaler, le, ball_features, cum_features, ctx_features, chase_features):
    # Load live match data.
    live_df = pd.read_csv(live_csv)
    # Ensure that the required contextual numeric columns exist.
    ctx_numeric = ['Batter_Historical_Avg', 'Bowler_Historical_Economy', 'Batter_vs_Bowler_Avg', 'Team_Powerplay_Performance', 'Match_Phase']
    for col in ctx_numeric:
        if col not in live_df.columns:
            live_df[col] = 0
    live_df = compute_live_features(live_df, TOTAL_BALLS)
    
    # Load historical data.
    historical_df = pd.read_csv(historical_csv)
    if 'Innings' in historical_df.columns:
        historical_df.sort_values(by=['Innings', 'Over', 'Ball'], inplace=True)
    else:
        historical_df.sort_values(by=['Over', 'Ball'], inplace=True)
    historical_df.reset_index(drop=True, inplace=True)
    
    hist_batter, hist_bowler = compute_historical_stats(historical_csv)
    
    report_lines = []
    
    # Overall live match summary.
    live_summary = {}
    live_summary['Total Runs'] = live_df['Runs From Ball'].sum()
    live_summary['Total Wickets'] = int(live_df['Wicket'].sum())
    live_summary['Total Balls'] = len(live_df)
    live_summary['Overs'] = round(len(live_df) / 6, 2)
    live_summary['Current Run Rate'] = round(live_summary['Total Runs'] / (len(live_df) / 6) if len(live_df)>0 else 0, 2)
    report_lines.append("=== Live Match Summary ===")
    for key, value in live_summary.items():
        report_lines.append(f"{key}: {value}")
    report_lines.append("")
    
    # Live Batter Analysis.
    live_batter = live_df.groupby('Batter').agg({
        'Batter Runs': ['sum', 'count'],
        'Runs From Ball': 'sum'
    })
    live_batter.columns = ['Live_Total_Runs', 'Live_Balls_Faced', 'Live_Runs_Contributed']
    live_batter['Live_Strike_Rate'] = (live_batter['Live_Total_Runs'] / live_batter['Live_Balls_Faced'] * 100).round(2)
    boundaries = live_df[live_df['Batter Runs'].isin([4,6])].groupby('Batter').size()
    live_batter['Boundaries'] = boundaries.fillna(0).astype(int)
    
    report_lines.append("=== Live Batter Analysis ===")
    for batter in live_batter.index:
        live_info = live_batter.loc[batter]
        if batter in hist_batter.index:
            hist_info = hist_batter.loc[batter]
        else:
            hist_info = None
        report_lines.append(f"Batter: {batter}")
        report_lines.append(f"  Live - Runs: {live_info['Live_Total_Runs']}, Balls: {live_info['Live_Balls_Faced']}, Strike Rate: {live_info['Live_Strike_Rate']}, Boundaries: {live_info['Boundaries']}")
        if hist_info is not None:
            report_lines.append(f"  Historical - Runs: {hist_info['Hist_Total_Runs']}, Strike Rate: {hist_info['Hist_Strike_Rate']}")
            if live_info['Live_Strike_Rate'] < hist_info['Hist_Strike_Rate']:
                report_lines.append("  Insight: Underperforming relative to historical average.")
            else:
                report_lines.append("  Insight: Performing at or above historical norms.")
        else:
            report_lines.append("  Historical data not available.")
        report_lines.append("")
    
    # Live Bowler Analysis.
    live_bowler = live_df.groupby('Bowler').agg({
        'Runs From Ball': 'sum',
        'Wicket': 'sum',
        'Ball': 'count'
    })
    live_bowler.rename(columns={'Ball': 'Live_Balls_Bowled'}, inplace=True)
    live_bowler['Live_Overs'] = live_bowler['Live_Balls_Bowled'] / 6.0
    live_bowler['Live_Economy'] = (live_bowler['Runs From Ball'] / live_bowler['Live_Overs']).round(2)
    live_bowler['Live_Wickets'] = live_bowler['Wicket'].astype(int)
    live_bowler.drop(columns=['Wicket'], inplace=True)
    
    report_lines.append("=== Live Bowler Analysis ===")
    for bowler in live_bowler.index:
        live_info = live_bowler.loc[bowler]
        if bowler in hist_bowler.index:
            hist_info = hist_bowler.loc[bowler]
        else:
            hist_info = None
        report_lines.append(f"Bowler: {bowler}")
        report_lines.append(f"  Live - Runs: {live_info['Runs From Ball']}, Overs: {live_info['Live_Overs']:.1f}, Economy: {live_info['Live_Economy']}, Wickets: {live_info['Live_Wickets']}")
        if hist_info is not None:
            report_lines.append(f"  Historical - Economy: {hist_info['Hist_Economy']}, Wickets: {hist_info['Hist_Wickets']}")
            if live_info['Live_Economy'] > hist_info['Hist_Economy']:
                report_lines.append("  Insight: Bowling economy is higher than historical average.")
            else:
                report_lines.append("  Insight: Bowling is within historical norms.")
        else:
            report_lines.append("  Historical data not available.")
        report_lines.append("")
    
    # Partnership Analysis.
    partnerships = []
    current_partnership = {'Batsmen': set(), 'Runs': 0, 'Balls': 0}
    for idx, row in live_df.iterrows():
        current_partnership['Batsmen'].add(row['Batter'])
        current_partnership['Runs'] += row['Runs From Ball']
        current_partnership['Balls'] += 1
        if row['Wicket'] == 1:
            partnerships.append(current_partnership)
            current_partnership = {'Batsmen': set(), 'Runs': 0, 'Balls': 0}
    if current_partnership['Balls'] > 0:
        partnerships.append(current_partnership)
    
    report_lines.append("=== Partnership Analysis ===")
    if partnerships:
        for p in partnerships:
            batsmen = ", ".join(sorted(p['Batsmen']))
            avg = p['Runs'] / p['Balls'] if p['Balls'] > 0 else 0
            report_lines.append(f"Partnership between {batsmen}: {p['Runs']} runs off {p['Balls']} balls (Average: {round(avg, 2)})")
    else:
        report_lines.append("No partnerships recorded.")
    report_lines.append("")
    
    # Event Probability Calculations.
    total_balls = len(live_df)
    boundary_balls = len(live_df[live_df['Batter Runs'].isin([4,6])])
    wicket_balls = len(live_df[live_df['Wicket'] == 1])
    dot_balls = len(live_df[live_df['Runs From Ball'] == 0])
    prob_boundary = round(boundary_balls / total_balls, 2) if total_balls > 0 else 0
    prob_wicket = round(wicket_balls / total_balls, 2) if total_balls > 0 else 0
    prob_dot = round(dot_balls / total_balls, 2) if total_balls > 0 else 0
    report_lines.append("=== Event Probability Calculations ===")
    report_lines.append(f"Probability of Boundary: {prob_boundary}")
    report_lines.append(f"Probability of Wicket: {prob_wicket}")
    report_lines.append(f"Probability of Dot Ball: {prob_dot}")
    report_lines.append("")
    
    # Model-Based Momentum Prediction.
    report_lines.append("=== Model-Based Momentum Prediction ===")
    model_momentum = predict_momentum_live(live_df)
    report_lines.append(f"Predicted Momentum: {model_momentum:.2f} / 100")
    report_lines.append("")
    
    # Recommendations and Insights.
    report_lines.append("=== Recommendations and Insights ===")
    report_lines.append("• Batters underperforming relative to historical averages may need to adjust shot selection.")
    report_lines.append("• Bowlers with higher live economy than historical figures should consider altering their length/line.")
    report_lines.append("• Partnership breakdowns may suggest reordering the batting lineup.")
    report_lines.append("• Probabilistic trends indicate pressure points—tactical adjustments are recommended accordingly.")
    
    return "\n".join(report_lines)

def predict_momentum_live(df):
    """Predicts momentum from live match data using the last ball's information."""
    with open('ball_scaler.pkl', 'rb') as f:
        ball_scaler = pickle.load(f)
    with open('cum_scaler.pkl', 'rb') as f:
        cum_scaler = pickle.load(f)
    with open('ctx_scaler.pkl', 'rb') as f:
        ctx_scaler = pickle.load(f)
    with open('chase_scaler.pkl', 'rb') as f:
        chase_scaler = pickle.load(f)
    with open('venue_le.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('ball_features.pkl', 'rb') as f:
        ball_features = pickle.load(f)
    with open('cum_features.pkl', 'rb') as f:
        cum_features = pickle.load(f)
    with open('ctx_features.pkl', 'rb') as f:
        ctx_features = pickle.load(f)
    with open('chase_features.pkl', 'rb') as f:
        chase_features = pickle.load(f)
    
    seq_length = 10
    if len(df) < seq_length:
        raise ValueError("Not enough live data to form a sequence.")
    X_seq = df[ball_features].values
    X_seq_scaled = ball_scaler.transform(X_seq)
    sequence = X_seq_scaled[-seq_length:]
    sequence = np.expand_dims(sequence, axis=0)
    
    current_cum = df.iloc[-1][cum_features].values.reshape(1, -1)
    current_cum_scaled = cum_scaler.transform(current_cum)
    
    venue_val = df.iloc[-1]['Venue']
    venue_index = le.transform([venue_val])[0] if venue_val in le.classes_ else 0
    venue_enc = np.array([[venue_index]])
    
    ctx_numeric = df.iloc[-1][ctx_features[1:]].values.reshape(1, -1)
    ctx_numeric_scaled = ctx_scaler.transform(ctx_numeric)
    
    chase_vals = df.iloc[-1][chase_features].values.reshape(1, -1)
    chase_scaled = chase_scaler.transform(chase_vals)
    
    model = load_model('momentum_model.h5', custom_objects={'mse': 'mse'})
    predicted = model.predict([sequence, current_cum_scaled, venue_enc, ctx_numeric_scaled, chase_scaled])
    return predicted[0][0]

#########################################
#              MAIN PIPELINE            #
#########################################

if __name__ == "__main__":
    # Define file paths.
    historical_csv = "ball_by_ball_ipl.csv"   # Historical ball-by-ball data (IPL 2008-23)
    live_csv = "over10.csv"                     # Live match segment CSV (e.g., first 5 overs / 30 balls)
    
    # Define sequence length.
    seq_length = 10
    
    # Step 1: Train the momentum model on historical data.
    model, ball_scaler, cum_scaler, ctx_scaler, chase_scaler, le, ball_features, cum_features, ctx_features, chase_features = train_momentum_model(
        historical_csv, seq_length=seq_length, epochs=10, batch_size=32, total_balls=TOTAL_BALLS)
    
    # Step 2: Compute historical aggregates.
    historical_batter_stats, historical_bowler_stats = compute_historical_stats(historical_csv)
    
    # Step 3: Generate the live match report.
    report = generate_live_report(live_csv, historical_csv, seq_length, ball_scaler, cum_scaler, ctx_scaler, chase_scaler, le, ball_features, cum_features, ctx_features, chase_features)
    
    print("\n================ FULL LIVE MATCH REPORT ================\n")
    print(report)
