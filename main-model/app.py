from flask import Flask, jsonify
import os
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import json
import openai
import re
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# =============================================================================
# GLOBAL CONSTANTS & FILE PATHS
# =============================================================================
TOTAL_BALLS = 120       # T20: 20 overs x 6 balls
SEQ_LENGTH = 10
HISTORICAL_CSV = "ball_by_ball_ipl.csv"          # Historical ball-by-ball data CSV
LIVE_CSV = "converted_match_innings2_over10.csv"  # Live match CSV (set to innings 1 data)
TEAM_CSV = "team.csv"                             # Starting XI CSV

# Default team names (will be overridden by TEAM_CSV if available)
DEFAULT_TEAM1 = "Mumbai Indians"
DEFAULT_TEAM2 = "Sunrisers Hyderabad"

# =============================================================================
# LOAD PRE-TRAINED OBJECTS & FEATURE LISTS
# =============================================================================
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
chase_features = ['Is_Chasing', 'Required Run Rate', 'Chase Differential']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
openai.api_key = os.getenv("OPENAI_API_KEY")
def generate_gpt_recommendations(match_report, batting_team, bowling_team):
    prompt = (
        f"Assume you are a cricket analytics coach. Based on the following match report, "
        f"provide 5 actionable recommendations and tactics for the batting team and 5 for the bowling team. "
        f"Focus only on batting advice when discussing the batting team and only on bowling advice for the fielding team. "
        f"Format your response as a numbered list, where each recommendation includes a main topic, actions, and purpose.\n\n"
        f"Team XI:\n"
        f"Batting Team: {batting_team}\n"
        f"Bowling Team: {bowling_team}\n\n"
        f"Match Report:\n{match_report}\n\n"
        f"Provide recommendations based solely on the latest data in the report (e.g. last ball details, momentum scores, current player performance)."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert cricket analytics coach."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message["content"]

def standardize_columns(df):
    df.columns = df.columns.str.strip()
    mapping = {
        "batter": "Batter",
        "bowler": "Bowler",
        "non_striker": "Non Striker",
        "runs_batter": "Batter Runs",
        "runs_extras": "Extra Runs",
        "runs_total": "Runs From Ball"
    }
    df.rename(columns=mapping, inplace=True)
    return df

def add_team_info(df):
    if "Innings" in df.columns:
        df["Innings"] = df["Innings"].astype(int)
        df["BattingTeam"] = df["Innings"].apply(lambda x: DEFAULT_TEAM1 if x == 1 else DEFAULT_TEAM2)
        df["BowlingTeam"] = df["Innings"].apply(lambda x: DEFAULT_TEAM2 if x == 1 else DEFAULT_TEAM1)
    return df

def ensure_ctx_columns(df):
    ctx_numeric = ['Batter_Historical_Avg', 'Bowler_Historical_Economy',
                   'Batter_vs_Bowler_Avg', 'Team_Powerplay_Performance', 'Match_Phase']
    for col in ctx_numeric:
        if col not in df.columns:
            df[col] = 0
    return df

def is_valid_delivery(row):
    if "extras_wides" in row and pd.notna(row["extras_wides"]):
        try:
            return float(row["extras_wides"]) == 0
        except:
            return True
    return True

def parse_cricket_recommendations(input_text):
    sections = input_text.strip().split("\n\n")
    recommendations = {}

    current_team = None

    for section in sections:
        lines = section.strip().split("\n")

        for line in lines:
            if "Team:" in line:
                current_team = re.sub(r"(Batting Team:|Bowling Team:)", "", line).strip()
                recommendations[current_team] = {"team_name": current_team, "recommendations": []}
            elif "Main Topic:" in line:
                topic = line.replace("Main Topic:", "").strip()
            elif "Action:" in line:
                action = line.replace("Action:", "").strip()
            elif "Purpose:" in line:
                purpose = line.replace("Purpose:", "").strip()
                recommendations[current_team]["recommendations"].append({
                    "topic": topic,
                    "action": action,
                    "purpose": purpose
                })

    return recommendations

def compute_live_features(df, total_balls=TOTAL_BALLS):
    df = df.copy()
    df = standardize_columns(df)
    # If "Over" or "Ball" are missing, create them from the index
    if "Over" not in df.columns:
        df["Over"] = (df.index // 6) + 1
    if "Ball" not in df.columns:
        df["Ball"] = (df.index % 6) + 1
    df = add_team_info(df)
    df = ensure_ctx_columns(df)
    if "Innings" in df.columns:
        df.sort_values(by=["Innings", "Over", "Ball"], inplace=True)
    else:
        df.sort_values(by=["Over", "Ball"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Runs From Ball"] = pd.to_numeric(df["Runs From Ball"], errors="coerce").fillna(0)
    df["Wicket"] = pd.to_numeric(df["Wicket"], errors="coerce").fillna(0)
    df["Cumulative Runs"] = df["Runs From Ball"].cumsum()
    df["Cumulative Wickets"] = df["Wicket"].cumsum()
    df["Balls Bowled"] = np.arange(1, len(df) + 1)
    df["Overs Completed"] = df["Balls Bowled"] / 6.0
    df["Current Run Rate"] = df["Cumulative Runs"] / df["Overs Completed"]
    if "Balls Remaining" not in df.columns:
        df["Balls Remaining"] = total_balls - df["Balls Bowled"]
    for col in ["Is_Chasing", "Required Run Rate", "Chase Differential"]:
        if col not in df.columns:
            df[col] = 0
    return df

def compute_over_metrics(df):
    metrics_list = []
    # Group by Over, and record the actual over number for display (starting at 1)
    for over, group in df.groupby("Over"):
        over_runs = group["Runs From Ball"].sum()
        wickets = group["Wicket"].sum()
        valid_balls = group.apply(lambda row: 1 if is_valid_delivery(row) else 0, axis=1).sum()
        over_run_rate = (over_runs / valid_balls * 6) if valid_balls > 0 else 0
        metrics_list.append({
            "Over": int(over),  # Store the actual over number
            "Over_Runs": float(over_runs),
            "Valid_Balls": int(valid_balls),
            "Wickets": int(wickets),
            "Over_Run_Rate": float(over_run_rate)
        })
    return metrics_list

def read_starting_xi(filename):
    df = pd.read_csv(filename, header=None)
    teams = df.iloc[0].tolist()
    team1 = teams[0].strip()
    team2 = teams[1].strip()
    team1_players = []
    team2_players = []
    for i in range(1, len(df)):
        row = df.iloc[i].tolist()
        if pd.notna(row[0]):
            team1_players.append(row[0].strip())
        if pd.notna(row[1]):
            team2_players.append(row[1].strip())
    return {team1: team1_players, team2: team2_players}

def compute_historical_stats(csv_path):
    df = pd.read_csv(csv_path)
    df = standardize_columns(df)
    df = add_team_info(df)
    if "Innings" in df.columns:
        df.sort_values(by=["Innings", "Over", "Ball"], inplace=True)
    else:
        df.sort_values(by=["Over", "Ball"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    batter_stats = df.groupby(["Batter", "BattingTeam"]).agg({
        "Batter Runs": ["sum", "count"],
        "Runs From Ball": "sum"
    })
    batter_stats.columns = ["Hist_Total_Runs", "Hist_Balls_Faced", "Hist_Runs_Contributed"]
    batter_stats["Hist_Strike_Rate"] = (batter_stats["Hist_Total_Runs"] / batter_stats["Hist_Balls_Faced"] * 100).round(2)
    bowler_stats = df.groupby(["Bowler", "BowlingTeam"]).agg({
        "Runs From Ball": "sum",
        "Wicket": "sum",
        "Ball": "count"
    })
    bowler_stats.rename(columns={"Ball": "Hist_Balls_Bowled"}, inplace=True)
    bowler_stats["Hist_Overs"] = bowler_stats["Hist_Balls_Bowled"] / 6.0
    bowler_stats["Hist_Economy"] = (bowler_stats["Runs From Ball"] / bowler_stats["Hist_Overs"]).round(2)
    bowler_stats["Hist_Wickets"] = bowler_stats["Wicket"].astype(int)
    bowler_stats.drop(columns=["Wicket"], inplace=True)
    return batter_stats, bowler_stats

def predict_momentum_live(df, ball_features, cum_features, ctx_features, chase_features):
    if len(df) < SEQ_LENGTH:
        raise ValueError("Not enough live data to form a sequence.")
    X_seq = df[ball_features].values
    X_seq_scaled = ball_scaler.transform(X_seq)
    sequence = X_seq_scaled[-SEQ_LENGTH:]
    sequence = np.expand_dims(sequence, axis=0)
    current_cum = df.iloc[-1][cum_features].values.reshape(1, -1)
    current_cum_scaled = cum_scaler.transform(current_cum)
    venue_val = df.iloc[-1]["Venue"] if "Venue" in df.columns else "Unknown"
    venue_index = le.transform([venue_val])[0] if venue_val in le.classes_ else 0
    venue_enc = np.array([[venue_index]])
    ctx_numeric = df.iloc[-1][ctx_features[1:]].values.reshape(1, -1)
    ctx_numeric_scaled = ctx_scaler.transform(ctx_numeric)
    chase_vals = df.iloc[-1][chase_features].values.reshape(1, -1)
    chase_scaled = chase_scaler.transform(chase_vals)
    loaded_model = load_model("momentum_model.h5", custom_objects={"mse": "mse"})
    predicted = loaded_model.predict([sequence, current_cum_scaled, venue_enc, ctx_numeric_scaled, chase_scaled])
    return float(predicted[0][0])

def compute_recent_stats(df, player_col, stat_col, window=5):
    recent_stats = {}
    for player, group in df.groupby(player_col):
        group = group.sort_values(by=["Over", "Ball"])
        recent_stats[player] = float(group[stat_col].tail(window).mean())
    return recent_stats

def get_last_ball_details(df):
    if df.empty:
        return {"last_over": None, "last_bowler": None, "on_strike_batter": None, "non_striker": None}
    latest_over = df["Over"].max()
    latest_df = df[df["Over"] == latest_over].sort_values(by="Ball")
    if latest_df.empty:
        return {"last_over": None, "last_bowler": None, "on_strike_batter": None, "non_striker": None}
    last_ball = latest_df.iloc[-1]
    return {
        "last_over": int(latest_over),
        "last_bowler": last_ball["Bowler"],
        "on_strike_batter": last_ball["Batter"],
        "non_striker": last_ball["Non Striker"]
    }

def compute_partnership_stats(df):
    if df.empty:
        return []
    
    partnerships = []
    current_pair = (df.iloc[0]["Batter"], df.iloc[0]["Non Striker"])
    current_runs = 0
    current_balls = 0
    
    for _, row in df.iterrows():
        batter = row["Batter"]
        non_striker = row["Non Striker"]
        
        # If we have a new partnership
        if set([batter, non_striker]) != set(current_pair):
            if current_balls > 0:
                run_rate = current_runs / current_balls * 6
                partnerships.append({
                    "batsmen": list(current_pair),
                    "runs": int(current_runs),
                    "balls": int(current_balls),
                    "run_rate": float(run_rate)
                })
            
            current_pair = (batter, non_striker)
            current_runs = 0
            current_balls = 0
        
        current_runs += row["Runs From Ball"]
        current_balls += 1
    
    # Add the last partnership
    if current_balls > 0:
        run_rate = current_runs / current_balls * 6
        partnerships.append({
            "batsmen": list(current_pair),
            "runs": int(current_runs),
            "balls": int(current_balls), 
            "run_rate": float(run_rate)
        })
    
    return partnerships

def calculate_probabilities(df, window=10):
    """Calculate predictive probabilities based on recent balls"""
    if len(df) < window:
        window = len(df)
    
    recent_df = df.tail(window)
    boundary_prob = ((recent_df["Runs From Ball"] == 4) | (recent_df["Runs From Ball"] == 6)).mean()
    dot_ball_prob = (recent_df["Runs From Ball"] == 0).mean()
    wicket_prob = (recent_df["Wicket"] > 0).mean()
    
    return {
        "boundary": float(boundary_prob),
        "dot_ball": float(dot_ball_prob),
        "wicket": float(wicket_prob)
    }

def format_batter_data(df, historical_batter_stats, starting_batters, user_team):
    batters_data = []
    recent_avg = compute_recent_stats(df, "Batter", "Batter Runs", window=5)
    
    for batter in starting_batters:
        batter_data = {"name": batter}
        
        # Historical data
        try:
            hist_info = historical_batter_stats.loc[(batter, user_team)]
            batter_data["historical_stats"] = {
                "runs": int(hist_info["Hist_Total_Runs"]),
                "strike_rate": float(hist_info["Hist_Strike_Rate"])
            }
        except Exception:
            batter_data["historical_stats"] = {"runs": 0, "strike_rate": 0}
        
        # Live data
        if batter in df["Batter"].unique():
            group = df[df["Batter"] == batter]
            total_runs = int(group["Batter Runs"].sum())
            balls = int(group.shape[0])
            strike_rate = float((total_runs / balls * 100) if balls > 0 else 0)
            fours = int((group["Batter Runs"] == 4).sum())
            sixes = int((group["Batter Runs"] == 6).sum())
            
            batter_data["live_stats"] = {
                "runs": total_runs,
                "balls_faced": balls,
                "strike_rate": strike_rate,
                "boundaries": fours + sixes  # Total boundaries (fours + sixes)
            }
            
            # Recent performance
            if batter in recent_avg:
                batter_data["recent_performance"] = float(recent_avg[batter])
        else:
            batter_data["live_stats"] = {
                "runs": 0,
                "balls_faced": 0,
                "strike_rate": 0,
                "boundaries": 0
            }
        
        # Insight
        if "historical_stats" in batter_data and "live_stats" in batter_data:
            hist_sr = batter_data["historical_stats"]["strike_rate"]
            live_sr = batter_data["live_stats"]["strike_rate"]
            
            if hist_sr > 0 and live_sr > 0:
                if live_sr < (hist_sr - 30):
                    batter_data["insight"] = "Performing below historical average."
                else:
                    batter_data["insight"] = "Performing at or above historical norms."
            else:
                batter_data["insight"] = "Insufficient data for comparison."
        
        batters_data.append(batter_data)
    
    return batters_data

def format_bowler_data(df, historical_bowler_stats, starting_bowlers, opponent_team):
    bowlers_data = []
    
    for bowler in starting_bowlers:
        bowler_data = {"name": bowler}
        
        # Historical data
        try:
            hist_info = historical_bowler_stats.loc[(bowler, opponent_team)]
            bowler_data["historical_stats"] = {
                "economy": float(hist_info["Hist_Economy"]),
                "wickets": int(hist_info["Hist_Wickets"])
            }
        except Exception:
            bowler_data["historical_stats"] = {"economy": 0, "wickets": 0}
        
        # Live data
        if bowler in df["Bowler"].unique():
            group = df[df["Bowler"] == bowler]
            runs_conceded = int(group["Runs From Ball"].sum())
            valid_balls = int(group.apply(lambda r: 1 if is_valid_delivery(r) else 0, axis=1).sum())
            overs = valid_balls / 6.0
            economy = float((runs_conceded / overs) if overs > 0 else 0)
            wickets = int(group["Wicket"].sum())
            
            bowler_data["live_stats"] = {
                "runs_conceded": runs_conceded,
                "overs": overs,
                "economy": economy,
                "wickets": wickets
            }
            
            # Insight
            if "historical_stats" in bowler_data:
                hist_economy = bowler_data["historical_stats"]["economy"]
                if hist_economy > 0 and economy > hist_economy + 1.5:
                    bowler_data["insight"] = "Bowling economy is higher than historical average."
                elif hist_economy > 0 and economy < hist_economy - 1.5:
                    bowler_data["insight"] = "Bowling economy is better than historical average."
                else:
                    bowler_data["insight"] = "Bowling in line with historical performance."
        else:
            bowler_data["live_stats"] = {
                "runs_conceded": 0,
                "overs": 0,
                "economy": 0,
                "wickets": 0
            }
            bowler_data["insight"] = "No bowling data available."
        
        bowlers_data.append(bowler_data)
    
    return bowlers_data

# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/report', methods=['GET'])
def get_detailed_report():
    try:
        # Read the live data and preprocess
        df_live = pd.read_csv(LIVE_CSV)
        df_live = standardize_columns(df_live)
        df_live = add_team_info(df_live)
        
        # Process innings separately if available
        innings_data = []
        
        if "Innings" in df_live.columns:
            for innings_num in df_live["Innings"].unique():
                innings_df = df_live[df_live["Innings"] == innings_num].copy()
                innings_df = compute_live_features(innings_df, TOTAL_BALLS)
                
                batting_team = DEFAULT_TEAM1 if innings_num == 1 else DEFAULT_TEAM2
                bowling_team = DEFAULT_TEAM2 if innings_num == 1 else DEFAULT_TEAM1
                
                # Read team info
                starting_dict = read_starting_xi(TEAM_CSV)
                starting_batters = starting_dict.get(batting_team, [])
                starting_bowlers = starting_dict.get(bowling_team, [])
                
                # Get historical stats
                historical_batter_stats, historical_bowler_stats = compute_historical_stats(HISTORICAL_CSV)
                
                # Format data
                batters_data = format_batter_data(innings_df, historical_batter_stats, starting_batters, batting_team)
                bowlers_data = format_bowler_data(innings_df, historical_bowler_stats, starting_bowlers, bowling_team)
                
                # Calculate momentum
                try:
                    batting_momentum = predict_momentum_live(
                        innings_df, ball_features, cum_features, ctx_features, chase_features
                    )
                except Exception:
                    batting_momentum = 50.0
                
                # Other metrics
                partnerships = compute_partnership_stats(innings_df)
                over_metrics = compute_over_metrics(innings_df)
                last_ball_details = get_last_ball_details(innings_df)
                
                # Match summary
                if not innings_df.empty:
                    last_row = innings_df.iloc[-1]
                    match_summary = {
                        "total_runs": int(last_row["Cumulative Runs"]),
                        "total_wickets": int(last_row["Cumulative Wickets"]),
                        "overs": float(last_row["Overs Completed"]),
                        "current_run_rate": float(last_row["Current Run Rate"])
                    }
                else:
                    match_summary = {
                        "total_runs": 0,
                        "total_wickets": 0,
                        "overs": 0,
                        "current_run_rate": 0
                    }
                
                innings_data.append({
                    "innings_number": int(innings_num),
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "match_summary": match_summary,
                    "batting_momentum": float(batting_momentum),
                    "bowling_momentum": float(100 - batting_momentum),
                    "batters_analysis": batters_data,
                    "bowlers_analysis": bowlers_data,
                    "partnerships": partnerships,
                    "over_metrics": over_metrics,
                    "last_ball_details": last_ball_details
                })
        else:
            # If no innings column, process as a single innings
            df_live = compute_live_features(df_live, TOTAL_BALLS)
            
            batting_team = DEFAULT_TEAM1
            bowling_team = DEFAULT_TEAM2
            
            # Read team info
            starting_dict = read_starting_xi(TEAM_CSV)
            starting_batters = starting_dict.get(batting_team, [])
            starting_bowlers = starting_dict.get(bowling_team, [])
            
            # Get historical stats
            historical_batter_stats, historical_bowler_stats = compute_historical_stats(HISTORICAL_CSV)
            
            # Format data
            batters_data = format_batter_data(df_live, historical_batter_stats, starting_batters, batting_team)
            bowlers_data = format_bowler_data(df_live, historical_bowler_stats, starting_bowlers, bowling_team)
            
            # Calculate momentum
            try:
                batting_momentum = predict_momentum_live(
                    df_live, ball_features, cum_features, ctx_features, chase_features
                )
            except Exception:
                batting_momentum = 50.0
            
            # Other metrics
            partnerships = compute_partnership_stats(df_live)
            over_metrics = compute_over_metrics(df_live)
            last_ball_details = get_last_ball_details(df_live)
            
            # Match summary
            if not df_live.empty:
                last_row = df_live.iloc[-1]
                match_summary = {
                    "total_runs": int(last_row["Cumulative Runs"]),
                    "total_wickets": int(last_row["Cumulative Wickets"]),
                    "overs": float(last_row["Overs Completed"]),
                    "current_run_rate": float(last_row["Current Run Rate"])
                }
            else:
                match_summary = {
                    "total_runs": 0,
                    "total_wickets": 0,
                    "overs": 0,
                    "current_run_rate": 0
                }
            
            innings_data.append({
                "innings_number": 1,
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "match_summary": match_summary,
                "batting_momentum": float(batting_momentum),
                "bowling_momentum": float(100 - batting_momentum),
                "batters_analysis": batters_data,
                "bowlers_analysis": bowlers_data,
                "partnerships": partnerships,
                "over_metrics": over_metrics,
                "last_ball_details": last_ball_details
            })
        
        # Generate match predictions if both innings have data
        match_prediction = None
        if len(innings_data) > 1 and "Is_Chasing" in df_live.columns:
            team1_score = innings_data[0]["match_summary"]["total_runs"]
            team2_score = innings_data[1]["match_summary"]["total_runs"]
            team2_wickets = innings_data[1]["match_summary"]["total_wickets"]
            team2_balls_faced = innings_data[1]["match_summary"]["overs"] * 6
            
            target = team1_score + 1
            runs_needed = target - team2_score
            balls_remaining = TOTAL_BALLS - team2_balls_faced
            wickets_remaining = 10 - team2_wickets
            
            if balls_remaining > 0:
                required_rate = runs_needed / (balls_remaining / 6)
                
                match_prediction = {
                    "target": int(target),
                    "runs_needed": int(runs_needed),
                    "balls_remaining": int(balls_remaining),
                    "wickets_remaining": int(wickets_remaining),
                    "required_run_rate": float(required_rate)
                }
        
        # Determine current innings and teams for recommendations
        last_innings = int(df_live["Innings"].iloc[-1]) if "Innings" in df_live.columns and len(df_live) > 0 else 1
        if last_innings == 1:
            batting_team = DEFAULT_TEAM1
            bowling_team = DEFAULT_TEAM2
        else:
            batting_team = DEFAULT_TEAM2
            bowling_team = DEFAULT_TEAM1
        
        # Create a match report for GPT
        match_report = f"Match Summary: {batting_team} vs {bowling_team}\n"
        
        # Add current innings details
        current_innings = next((inn for inn in innings_data if inn["innings_number"] == last_innings), None)
        if current_innings:
            match_report += f"\nCurrent Score: {current_innings['match_summary']['total_runs']}/{current_innings['match_summary']['total_wickets']}"
            match_report += f"\nOvers: {current_innings['match_summary']['overs']:.1f}"
            match_report += f"\nRun Rate: {current_innings['match_summary']['current_run_rate']:.2f}"
            match_report += f"\nBatting Momentum: {current_innings['batting_momentum']:.1f}"
            match_report += f"\nBowling Momentum: {current_innings['bowling_momentum']:.1f}"
            
            # Add top batters
            if current_innings['batters_analysis']:
                match_report += "\n\nTop Batters:"
                for batter in current_innings['batters_analysis'][:3]:  # Top 3 batters
                    if 'live_stats' in batter and batter['live_stats']['runs'] > 0:
                        match_report += f"\n{batter['name']}: {batter['live_stats']['runs']} runs ({batter['live_stats']['strike_rate']:.1f} SR)"
            
            # Add top bowlers
            if current_innings['bowlers_analysis']:
                match_report += "\n\nTop Bowlers:"
                for bowler in current_innings['bowlers_analysis'][:3]:  # Top 3 bowlers
                    if 'live_stats' in bowler and bowler['live_stats']['wickets'] > 0:
                        match_report += f"\n{bowler['name']}: {bowler['live_stats']['wickets']} wickets (Econ: {bowler['live_stats']['economy']:.2f})"
            
            # Add partnerships
            if current_innings['partnerships']:
                match_report += "\n\nCurrent Partnership:"
                current_partnership = current_innings['partnerships'][-1]
                match_report += f"\n{' & '.join(current_partnership['batsmen'])}: {current_partnership['runs']} runs ({current_partnership['balls']} balls)"
            
            # Add last ball details
            if current_innings['last_ball_details']['last_over']:
                match_report += "\n\nLast Ball Details:"
                match_report += f"\nOver: {current_innings['last_ball_details']['last_over']}"
                match_report += f"\nBowler: {current_innings['last_ball_details']['last_bowler']}"
                match_report += f"\nBatter: {current_innings['last_ball_details']['on_strike_batter']}"
                match_report += f"\nNon-Striker: {current_innings['last_ball_details']['non_striker']}"
        
        # Add match prediction if available
        if match_prediction:
            match_report += "\n\nMatch Prediction:"
            match_report += f"\nTarget: {match_prediction['target']}"
            match_report += f"\nRuns Needed: {match_prediction['runs_needed']}"
            match_report += f"\nBalls Remaining: {match_prediction['balls_remaining']}"
            match_report += f"\nRequired Run Rate: {match_prediction['required_run_rate']:.2f}"

        # Generate GPT recommendations
        gpt_response = generate_gpt_recommendations(match_report, batting_team, bowling_team)

        # gpt_rec = parse_cricket_recommendations(gpt_response)

        # Prepare final response with structured recommendations
        response = {
            "innings_data": innings_data,
            "match_prediction": match_prediction,
            # "recommendations": gpt_rec,
            "raw_recommendations": gpt_response  # Optional: keep the original text if needed
        }

        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)