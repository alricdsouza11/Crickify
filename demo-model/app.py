import io
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS

# -----------------------------
# Global Constants and Preloads
# -----------------------------
TOTAL_BALLS = 120
SEQ_LENGTH = 10
HISTORICAL_CSV = "ball_by_ball_ipl.csv"  # path to your historical ball-by-ball CSV

# Preload your saved model and preprocessors so they are loaded only once.
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
# Define chase features
chase_features = ['Is_Chasing', 'Required Run Rate', 'Chase Differential']

# Load the pre-trained momentum model.
model = load_model('momentum_model.h5', custom_objects={'mse': 'mse'})

# Precompute historical aggregates (for batter and bowler stats).
def compute_historical_stats(csv_path):
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

historical_batter_stats, historical_bowler_stats = compute_historical_stats(HISTORICAL_CSV)

# -----------------------------
# Pipeline Functions (Live Report)
# -----------------------------
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

def predict_momentum_live(df):
    """Predicts momentum from live match data using the last ball's information."""
    if len(df) < SEQ_LENGTH:
        raise ValueError("Not enough live data to form a sequence.")
    X_seq = df[ball_features].values
    X_seq_scaled = ball_scaler.transform(X_seq)
    sequence = X_seq_scaled[-SEQ_LENGTH:]
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
    
    predicted = model.predict([sequence, current_cum_scaled, venue_enc, ctx_numeric_scaled, chase_scaled])
    return predicted[0][0]

def generate_live_report_from_df(live_df):
    """
    Generates a full live match report (including momentum score) from a live DataFrame.
    """
    # Ensure required contextual numeric columns exist.
    ctx_numeric = ['Batter_Historical_Avg', 'Bowler_Historical_Economy', 
                   'Batter_vs_Bowler_Avg', 'Team_Powerplay_Performance', 'Match_Phase']
    for col in ctx_numeric:
        if col not in live_df.columns:
            live_df[col] = 0
    live_df = compute_live_features(live_df, TOTAL_BALLS)
    
    # Use precomputed historical aggregates.
    # (historical_batter_stats and historical_bowler_stats were preloaded above.)
    
    report_lines = []
    # Overall match summary.
    live_summary = {
        'Total Runs': live_df['Runs From Ball'].sum(),
        'Total Wickets': int(live_df['Wicket'].sum()),
        'Total Balls': len(live_df),
        'Overs': round(len(live_df) / 6, 2),
        'Current Run Rate': round(live_df['Runs From Ball'].sum() / (len(live_df) / 6) if len(live_df) > 0 else 0, 2)
    }
    report_lines.append("=== Live Match Summary ===")
    for k, v in live_summary.items():
        report_lines.append(f"{k}: {v}")
    report_lines.append("")
    
    # Live Batter Analysis.
    live_batter = live_df.groupby('Batter').agg({
        'Batter Runs': ['sum', 'count'],
        'Runs From Ball': 'sum'
    })
    live_batter.columns = ['Live_Total_Runs', 'Live_Balls_Faced', 'Live_Runs_Contributed']
    live_batter['Live_Strike_Rate'] = (live_batter['Live_Total_Runs'] / live_batter['Live_Balls_Faced'] * 100).round(2)
    boundaries = live_df[live_df['Batter Runs'].isin([4,6])].groupby('Batter').size()
    live_batter['Boundaries'] = boundaries.reindex(live_batter.index).fillna(0).astype(int)
    
    report_lines.append("=== Live Batter Analysis ===")
    for batter in live_batter.index:
        live_info = live_batter.loc[batter]
        hist_info = None
        if batter in historical_batter_stats.index:
            hist_info = historical_batter_stats.loc[batter]
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
        hist_info = None
        if bowler in historical_bowler_stats.index:
            hist_info = historical_bowler_stats.loc[bowler]
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
            report_lines.append(f"Partnership between {batsmen}: {p['Runs']} runs off {p['Balls']} balls (Avg: {round(avg, 2)})")
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
    try:
        report_lines.append("=== Model-Based Momentum Prediction ===")
        model_momentum = predict_momentum_live(live_df)
        report_lines.append(f"Predicted Momentum: {model_momentum:.2f} / 100")
    except ValueError as e:
        report_lines.append(f"Momentum Prediction Error: {str(e)}")
    report_lines.append("")
    
    # Recommendations.
    report_lines.append("=== Recommendations and Insights ===")
    report_lines.append("• Batters underperforming relative to historical averages may need to adjust shot selection.")
    report_lines.append("• Bowlers with higher live economy than historical figures should consider altering their length/line.")
    report_lines.append("• Partnership breakdowns may suggest reordering the batting lineup.")
    report_lines.append("• Probabilistic trends indicate pressure points—tactical adjustments are recommended accordingly.")
    
    return "\n".join(report_lines)

# -----------------------------
# Flask API
# -----------------------------
app = Flask(__name__)
CORS(app)

@app.route('/report', methods=['GET'])
def report_api():
    """
    GET endpoint that reads data from a specified CSV file path on the server.
    Returns a structured JSON response with organized report sections.
    """
    try:
        live_df = pd.read_csv("converted_match_innings1_over15.csv")
    except Exception as e:
        return jsonify({"error": f"Error reading CSV: {str(e)}"}), 400
    
    try:
        # Compute features for the dataframe
        live_df = compute_live_features(live_df, TOTAL_BALLS)
        
        # Ensure required contextual numeric columns exist
        ctx_numeric = ['Batter_Historical_Avg', 'Bowler_Historical_Economy', 
                   'Batter_vs_Bowler_Avg', 'Team_Powerplay_Performance', 'Match_Phase']
        for col in ctx_numeric:
            if col not in live_df.columns:
                live_df[col] = 0
        
        # --- Match Summary ---
        match_summary = {
            'total_runs': int(live_df['Runs From Ball'].sum()),
            'total_wickets': int(live_df['Wicket'].sum()),
            'total_balls': len(live_df),
            'overs': round(len(live_df) / 6, 2),
            'current_run_rate': round(live_df['Runs From Ball'].sum() / (len(live_df) / 6) if len(live_df) > 0 else 0, 2)
        }
        
        # --- Batter Analysis ---
        live_batter = live_df.groupby('Batter').agg({
            'Batter Runs': ['sum', 'count'],
            'Runs From Ball': 'sum'
        })
        live_batter.columns = ['Live_Total_Runs', 'Live_Balls_Faced', 'Live_Runs_Contributed']
        live_batter['Live_Strike_Rate'] = (live_batter['Live_Total_Runs'] / live_batter['Live_Balls_Faced'] * 100).round(2)
        boundaries = live_df[live_df['Batter Runs'].isin([4,6])].groupby('Batter').size()
        live_batter['Boundaries'] = boundaries.reindex(live_batter.index).fillna(0).astype(int)
        
        batters_analysis = []
        for batter in live_batter.index:
            live_info = live_batter.loc[batter]
            batter_data = {
                'name': batter,
                'live_stats': {
                    'runs': int(live_info['Live_Total_Runs']),
                    'balls_faced': int(live_info['Live_Balls_Faced']),
                    'strike_rate': float(live_info['Live_Strike_Rate']),
                    'boundaries': int(live_info['Boundaries'])
                }
            }
            
            if batter in historical_batter_stats.index:
                hist_info = historical_batter_stats.loc[batter]
                batter_data['historical_stats'] = {
                    'runs': int(hist_info['Hist_Total_Runs']),
                    'strike_rate': float(hist_info['Hist_Strike_Rate'])
                }
                
                if live_info['Live_Strike_Rate'] < hist_info['Hist_Strike_Rate']:
                    batter_data['insight'] = "Underperforming relative to historical average."
                else:
                    batter_data['insight'] = "Performing at or above historical norms."
            else:
                batter_data['historical_stats'] = None
                batter_data['insight'] = "Historical data not available."
                
            batters_analysis.append(batter_data)
        
        # --- Bowler Analysis ---
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
        
        bowlers_analysis = []
        for bowler in live_bowler.index:
            live_info = live_bowler.loc[bowler]
            bowler_data = {
                'name': bowler,
                'live_stats': {
                    'runs_conceded': int(live_info['Runs From Ball']),
                    'overs': float(live_info['Live_Overs']),
                    'economy': float(live_info['Live_Economy']),
                    'wickets': int(live_info['Live_Wickets'])
                }
            }
            
            if bowler in historical_bowler_stats.index:
                hist_info = historical_bowler_stats.loc[bowler]
                bowler_data['historical_stats'] = {
                    'economy': float(hist_info['Hist_Economy']),
                    'wickets': int(hist_info['Hist_Wickets'])
                }
                
                if live_info['Live_Economy'] > hist_info['Hist_Economy']:
                    bowler_data['insight'] = "Bowling economy is higher than historical average."
                else:
                    bowler_data['insight'] = "Bowling is within historical norms."
            else:
                bowler_data['historical_stats'] = None
                bowler_data['insight'] = "Historical data not available."
                
            bowlers_analysis.append(bowler_data)
        
        # --- Partnership Analysis ---
        partnerships = []
        current_partnership = {'batsmen': set(), 'runs': 0, 'balls': 0}
        for idx, row in live_df.iterrows():
            current_partnership['batsmen'].add(row['Batter'])
            current_partnership['runs'] += row['Runs From Ball']
            current_partnership['balls'] += 1
            if row['Wicket'] == 1:
                if current_partnership['balls'] > 0:
                    partnerships.append({
                        'batsmen': sorted(list(current_partnership['batsmen'])),
                        'runs': int(current_partnership['runs']),
                        'balls': int(current_partnership['balls']),
                        'run_rate': round(current_partnership['runs'] / current_partnership['balls'], 2) if current_partnership['balls'] > 0 else 0
                    })
                current_partnership = {'batsmen': set(), 'runs': 0, 'balls': 0}
        
        if current_partnership['balls'] > 0:
            partnerships.append({
                'batsmen': sorted(list(current_partnership['batsmen'])),
                'runs': int(current_partnership['runs']),
                'balls': int(current_partnership['balls']),
                'run_rate': round(current_partnership['runs'] / current_partnership['balls'], 2) if current_partnership['balls'] > 0 else 0
            })
        
        # --- Event Probability Calculations ---
        total_balls = len(live_df)
        boundary_balls = len(live_df[live_df['Batter Runs'].isin([4,6])])
        wicket_balls = len(live_df[live_df['Wicket'] == 1])
        dot_balls = len(live_df[live_df['Runs From Ball'] == 0])
        
        probabilities = {
            'boundary': round(boundary_balls / total_balls, 2) if total_balls > 0 else 0,
            'wicket': round(wicket_balls / total_balls, 2) if total_balls > 0 else 0,
            'dot_ball': round(dot_balls / total_balls, 2) if total_balls > 0 else 0
        }
        
        # --- Model-Based Momentum Prediction ---
        momentum = None
        try:
            momentum = float(predict_momentum_live(live_df))
        except ValueError as e:
            momentum = str(e)
        
        # --- Recommendations ---
        recommendations = [
            "Batters underperforming relative to historical averages may need to adjust shot selection.",
            "Bowlers with higher live economy than historical figures should consider altering their length/line.",
            "Partnership breakdowns may suggest reordering the batting lineup.",
            "Probabilistic trends indicate pressure points—tactical adjustments are recommended accordingly."
        ]
        
        # Construct the final structured response
        response = {
            "match_summary": match_summary,
            "batters_analysis": batters_analysis,
            "bowlers_analysis": bowlers_analysis,
            "partnerships": partnerships,
            "probabilities": probabilities,
            "momentum": 50 if pd.isna(momentum) else momentum,
            "recommendations": recommendations
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Error generating report: {str(e)}"}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
