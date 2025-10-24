# match_report_pipeline.py

import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Lambda, Input, Concatenate, Embedding, Flatten
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

#########################################
#        TRAINING & HISTORICAL STATS    #
#########################################

def load_and_preprocess_data(csv_path, seq_length=10):
    """
    Loads ball-by-ball historical data from csv_path,
    sorts it by Over and Ball, adds any missing contextual columns,
    computes cumulative features, and returns the processed DataFrame.
    """
    data = pd.read_csv(csv_path)
    data.sort_values(by=['Over', 'Ball'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Ensure contextual numeric columns exist; if not, add them with default 0.
    ctx_numeric = ['Batter_Historical_Avg', 'Bowler_Historical_Economy', 
                   'Batter_vs_Bowler_Avg', 'Team_Powerplay_Performance', 'Match_Phase']
    for col in ctx_numeric:
        if col not in data.columns:
            data[col] = 0

    # Compute cumulative features.
    data['Cumulative Runs'] = data['Runs From Ball'].cumsum()
    data['Cumulative Wickets'] = data['Wicket'].astype(float).cumsum()
    data['Balls Bowled'] = np.arange(1, len(data) + 1)
    data['Overs Completed'] = data['Balls Bowled'] / 6.0
    data['Current Run Rate'] = data['Cumulative Runs'] / data['Overs Completed']
    
    # Compute an example target momentum.
    # (For instance: base 50 plus bonus for run rate above 7, minus 5 points per wicket lost.)
    data['Target Momentum'] = 50 + (data['Current Run Rate'] - 7) * 5 - (data['Cumulative Wickets'] * 5)
    data['Target Momentum'] = data['Target Momentum'].clip(0, 100)
    
    # For the contextual categorical feature, use "Venue".
    if 'Venue' not in data.columns:
        data['Venue'] = "Unknown"
    le = LabelEncoder()
    data['Venue_enc'] = le.fit_transform(data['Venue'])
    
    # Scale numeric groups.
    ball_features = ['Batter Runs', 'Extra Runs', 'Runs From Ball', 'Balls Remaining']
    cum_features = ['Cumulative Runs', 'Cumulative Wickets', 'Current Run Rate']
    ctx_features = ['Venue_enc'] + ctx_numeric  # contextual: first is encoded Venue
    
    ball_scaler = StandardScaler()
    data[ball_features] = ball_scaler.fit_transform(data[ball_features])
    
    cum_scaler = StandardScaler()
    data[cum_features] = cum_scaler.fit_transform(data[cum_features])
    
    ctx_scaler = StandardScaler()
    data[ctx_numeric] = ctx_scaler.fit_transform(data[ctx_numeric])
    
    # Drop rows with missing values in our feature groups.
    data = data.dropna(subset=ball_features + cum_features + ctx_features + ['Target Momentum'])
    return data, ball_features, cum_features, ctx_features, ball_scaler, cum_scaler, ctx_scaler, le

def train_momentum_model(csv_path, seq_length=10, epochs=10, batch_size=32):
    """
    Trains a momentum prediction model on historical ball-by-ball data.
    Returns the trained model and the preprocessing objects.
    """
    data, ball_features, cum_features, ctx_features, ball_scaler, cum_scaler, ctx_scaler, le = load_and_preprocess_data(csv_path, seq_length)
    
    # Create training sequences.
    X_seq, X_cum, X_ctx, y_target = [], [], [], []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length][ball_features].values
        X_seq.append(seq)
        X_cum.append(data.iloc[i+seq_length][cum_features].values)
        X_ctx.append(data.iloc[i+seq_length][ctx_features].values)
        y_target.append(data.iloc[i+seq_length]['Target Momentum'])
    
    X_seq = np.array(X_seq, dtype=np.float32)
    X_cum = np.array(X_cum, dtype=np.float32)
    X_ctx = np.array(X_ctx, dtype=np.float32)
    y_target = np.array(y_target, dtype=np.float32)
    
    # Build a multi-input model.
    num_ball_features = len(ball_features)
    num_cum_features = len(cum_features)
    num_ctx_numeric = len(ctx_features) - 1  # excluding the encoded Venue
    vocab_size = len(le.classes_)
    
    # Define inputs.
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
    
    combined = Concatenate()([x, y_branch, ctx_out])
    z = Dense(8, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(z)
    output = Lambda(lambda t: t * 100)(output)
    
    model = Model(inputs=[input_seq, input_cum, input_ctx_cat, input_ctx_num], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    # Train the model.
    model.fit([X_seq, X_cum, X_ctx[:, 0].reshape(-1, 1), X_ctx[:, 1:]],
              y_target, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    # Save the model and preprocessors.
    model.save('momentum_model.h5')
    with open('ball_scaler.pkl', 'wb') as f:
        pickle.dump(ball_scaler, f)
    with open('cum_scaler.pkl', 'wb') as f:
        pickle.dump(cum_scaler, f)
    with open('ctx_scaler.pkl', 'wb') as f:
        pickle.dump(ctx_scaler, f)
    with open('venue_le.pkl', 'wb') as f:
        pickle.dump(le, f)
    with open('ball_features.pkl', 'wb') as f:
        pickle.dump(ball_features, f)
    with open('cum_features.pkl', 'wb') as f:
        pickle.dump(cum_features, f)
    with open('ctx_features.pkl', 'wb') as f:
        pickle.dump(ctx_features, f)
    
    return model, ball_scaler, cum_scaler, ctx_scaler, le, ball_features, cum_features, ctx_features

def compute_historical_player_stats(csv_path):
    """
    Computes aggregated historical performance statistics from the full historical ball-by-ball data.
    Returns two DataFrames: one for batters and one for bowlers.
    """
    df = pd.read_csv(csv_path)
    df.sort_values(by=['Over', 'Ball'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Batter aggregated stats.
    batter_stats = df.groupby('Batter').agg({
        'Batter Runs': ['sum', 'count'],
        'Runs From Ball': 'sum',
        'Wicket': 'sum'
    })
    batter_stats.columns = ['Total Runs', 'Balls Faced', 'Runs Contributed', 'Wickets']
    batter_stats['Strike Rate'] = (batter_stats['Total Runs'] / batter_stats['Balls Faced'] * 100).round(2)
    boundaries = df[df['Batter Runs'].isin([4, 6])].groupby('Batter').size()
    batter_stats['Boundaries'] = boundaries.fillna(0).astype(int)
    
    # Bowler aggregated stats.
    bowler_stats = df.groupby('Bowler').agg({
        'Runs From Ball': 'sum',
        'Wicket': 'sum',
        'Ball': 'count'
    })
    bowler_stats.rename(columns={'Ball': 'Balls Bowled'}, inplace=True)
    bowler_stats['Overs'] = bowler_stats['Balls Bowled'] / 6.0
    bowler_stats['Economy'] = (bowler_stats['Runs From Ball'] / bowler_stats['Overs']).round(2)
    bowler_stats['Wickets'] = bowler_stats['Wicket'].astype(int)
    bowler_stats.drop(columns=['Wicket'], inplace=True)
    
    return batter_stats, bowler_stats

#########################################
#         LIVE MATCH REPORT             #
#########################################

def compute_live_match_features(df):
    """For live match data, compute cumulative features."""
    df = df.copy()
    df.sort_values(by=['Over', 'Ball'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Cumulative Runs'] = df['Runs From Ball'].cumsum()
    df['Cumulative Wickets'] = df['Wicket'].astype(float).cumsum()
    df['Balls Bowled'] = np.arange(1, len(df) + 1)
    df['Overs Completed'] = df['Balls Bowled'] / 6.0
    df['Current Run Rate'] = df['Cumulative Runs'] / df['Overs Completed']
    return df

def get_venue_encoded(venue_val, le):
    if venue_val in le.classes_:
        return le.transform([venue_val])[0]
    else:
        print(f"Warning: Venue '{venue_val}' not seen during training. Using default encoding (0).")
        return 0

def generate_live_match_report(live_csv, historical_batter_stats, historical_bowler_stats):
    """
    Loads live match data (ball-by-ball up to current over),
    computes current aggregated statistics, compares with historical averages,
    and produces a comprehensive textual report.
    """
    df = pd.read_csv(live_csv)
    df = compute_live_match_features(df)
    
    report_lines = []
    
    # Overall live match summary.
    live_summary = {}
    live_summary['Total Runs'] = df['Runs From Ball'].sum()
    live_summary['Total Wickets'] = int(df['Wicket'].sum())
    live_summary['Total Balls'] = len(df)
    live_summary['Overs'] = round(len(df)/6, 2)
    live_summary['Current Run Rate'] = round(live_summary['Total Runs'] / (len(df)/6) if len(df)>0 else 0, 2)
    
    report_lines.append("=== Live Match Summary ===")
    for key, value in live_summary.items():
        report_lines.append(f"{key}: {value}")
    report_lines.append("")
    
    # Batter analysis for live match.
    live_batter_stats = df.groupby('Batter').agg({
        'Batter Runs': ['sum', 'count'],
        'Runs From Ball': 'sum',
        'Wicket': 'sum'
    })
    live_batter_stats.columns = ['Total Runs', 'Balls Faced', 'Runs Contributed', 'Wickets']
    live_batter_stats['Strike Rate'] = (live_batter_stats['Total Runs'] / live_batter_stats['Balls Faced'] * 100).round(2)
    boundaries = df[df['Batter Runs'].isin([4,6])].groupby('Batter').size()
    live_batter_stats['Boundaries'] = boundaries.fillna(0).astype(int)
    
    report_lines.append("=== Live Batter Analysis ===")
    for batter in live_batter_stats.index:
        live_info = live_batter_stats.loc[batter]
        hist_info = historical_batter_stats.loc[batter] if batter in historical_batter_stats.index else None
        report_lines.append(f"Batter: {batter}")
        report_lines.append(f"  Live - Total Runs: {live_info['Total Runs']}, Balls Faced: {live_info['Balls Faced']}, Strike Rate: {live_info['Strike Rate']}, Boundaries: {live_info['Boundaries']}")
        if hist_info is not None:
            report_lines.append(f"  Historical - Total Runs: {hist_info['Total Runs']}, Strike Rate: {hist_info['Strike Rate']}")
            if live_info['Strike Rate'] < hist_info['Strike Rate']:
                report_lines.append("  Insight: Currently underperforming relative to historical average.")
            else:
                report_lines.append("  Insight: Performing well compared to historical standards.")
        else:
            report_lines.append("  Historical data not available.")
        report_lines.append("")
    
    # Bowler analysis for live match.
    live_bowler_stats = df.groupby('Bowler').agg({
        'Runs From Ball': 'sum',
        'Wicket': 'sum',
        'Ball': 'count'
    })
    live_bowler_stats.rename(columns={'Ball': 'Balls Bowled'}, inplace=True)
    live_bowler_stats['Overs'] = live_bowler_stats['Balls Bowled'] / 6.0
    live_bowler_stats['Economy'] = (live_bowler_stats['Runs From Ball'] / live_bowler_stats['Overs']).round(2)
    live_bowler_stats['Wickets'] = live_bowler_stats['Wicket'].astype(int)
    live_bowler_stats.drop(columns=['Wicket'], inplace=True)
    
    report_lines.append("=== Live Bowler Analysis ===")
    for bowler in live_bowler_stats.index:
        live_info = live_bowler_stats.loc[bowler]
        hist_info = historical_bowler_stats.loc[bowler] if bowler in historical_bowler_stats.index else None
        report_lines.append(f"Bowler: {bowler}")
        report_lines.append(f"  Live - Runs Conceded: {live_info['Runs From Ball']}, Overs: {live_info['Overs']:.1f}, Economy: {live_info['Economy']}, Wickets: {live_info['Wickets']}")
        if hist_info is not None:
            report_lines.append(f"  Historical - Economy: {hist_info['Economy']}, Total Wickets: {hist_info['Wickets']}")
            if live_info['Economy'] > hist_info['Economy']:
                report_lines.append("  Insight: Economy is higher than historical average; may require adjustments.")
            else:
                report_lines.append("  Insight: Bowling is within historical norms.")
        else:
            report_lines.append("  Historical data not available.")
        report_lines.append("")
    
    # Partnership Analysis.
    partnerships = []
    current_partnership = {'Batsmen': set(), 'Runs': 0, 'Balls': 0}
    for idx, row in df.iterrows():
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
            report_lines.append(f"Partnership between {batsmen}: {p['Runs']} runs off {p['Balls']} balls (Average: {round(avg,2)})")
    else:
        report_lines.append("No partnerships recorded.")
    report_lines.append("")
    
    # Probability Calculations.
    total_balls = len(df)
    boundary_balls = len(df[df['Batter Runs'].isin([4,6])])
    wicket_balls = len(df[df['Wicket'] == 1])
    dot_balls = len(df[df['Runs From Ball'] == 0])
    prob_boundary = round(boundary_balls/total_balls, 2) if total_balls else 0
    prob_wicket = round(wicket_balls/total_balls, 2) if total_balls else 0
    prob_dot = round(dot_balls/total_balls, 2) if total_balls else 0
    report_lines.append("=== Event Probability Calculations ===")
    report_lines.append(f"Probability of Boundary: {prob_boundary}")
    report_lines.append(f"Probability of Wicket: {prob_wicket}")
    report_lines.append(f"Probability of Dot Ball: {prob_dot}")
    report_lines.append("")
    
    # You may also add the model prediction of momentum (if using the trained model).
    # Here we assume you have saved the momentum model and corresponding preprocessors.
    # (For brevity, this example report focuses on aggregated statistics and insights.)
    
    report_lines.append("=== Recommendations and Key Insights ===")
    report_lines.append("• Batters underperforming relative to historical averages may need to adjust shot selection.")
    report_lines.append("• Bowlers with high live economy compared to historical figures should consider tighter lines.")
    report_lines.append("• Partnership breakdowns suggest strategic changes in batting order.")
    report_lines.append("• Probabilistic trends indicate where pressure is building; focus on defensive or aggressive strategies accordingly.")
    
    full_report = "\n".join(report_lines)
    return full_report

#########################################
#               MAIN PIPELINE           #
#########################################

if __name__ == "__main__":
    # Step 1: Train the model on historical ball-by-ball data (IPL 2008-23)
    historical_csv = "ball_by_ball_ipl.csv"  # Historical data file path
    # (You can adjust epochs/batch_size as needed.)
    model, ball_scaler, cum_scaler, ctx_scaler, le, ball_features, cum_features, ctx_features = train_momentum_model(historical_csv, seq_length=10, epochs=10, batch_size=32)
    
    # Step 2: Compute historical player aggregates for deep analysis.
    historical_batter_stats, historical_bowler_stats = compute_historical_player_stats(historical_csv)
    
    # Step 3: Generate a detailed report on a live match segment.
    live_csv = "over10.csv"  # This CSV should contain ball-by-ball data for the current match up to the desired over (e.g., first 30 balls)
    report = generate_live_match_report(live_csv, historical_batter_stats, historical_bowler_stats)
    
    print("\n================ FULL MATCH REPORT ================\n")
    print(report)
