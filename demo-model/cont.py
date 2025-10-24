import json
import pandas as pd

def convert_json_to_csv(json_file, csv_file):
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    info = data.get('info', {})
    meta = data.get('meta', {})
    
    # Determine Bat First and Bat Second based on toss decision.
    toss = info.get('toss', {})
    teams = info.get('teams', [])
    toss_winner = toss.get('winner')
    toss_decision = toss.get('decision', '').lower()
    
    if toss_winner and toss_decision == 'bat':
        bat_first = toss_winner
        bat_second = teams[0] if teams[0] != toss_winner else teams[1]
    elif toss_winner and toss_decision == 'field':
        # When toss decision is field, the other team bats first.
        bat_first = teams[0] if teams[0] != toss_winner else teams[1]
        bat_second = toss_winner
    else:
        bat_first = teams[0] if teams else ''
        bat_second = teams[1] if len(teams) > 1 else ''
    
    # Use available match info
    match_id = info.get('event', {}).get('match_number', '')
    # Use the first date from the dates list or meta.created
    date = info.get('dates', [meta.get('created', '')])[0]
    venue = info.get('venue', '')
    winner = info.get('outcome', {}).get('winner', '')
    
    rows = []
    innings_list = data.get('innings', [])
    innings1_total_runs = None  # to store first innings total runs for target calculation

    # Process each innings
    for innings_index, innings in enumerate(innings_list):
        innings_number = innings_index + 1
        cumulative_runs = 0
        cumulative_wickets = 0
        ball_counter = 0  # total balls bowled in the innings

        overs = innings.get('overs', [])
        for over in overs:
            over_number = over.get('over', 0)
            deliveries = over.get('deliveries', [])
            for ball_index, delivery in enumerate(deliveries):
                ball_counter += 1

                batter = delivery.get('batter', '')
                bowler = delivery.get('bowler', '')
                non_striker = delivery.get('non_striker', '')
                
                # Runs information from delivery
                runs_info = delivery.get('runs', {})
                batter_runs = runs_info.get('batter', 0)
                extra_runs = runs_info.get('extras', 0)
                total_runs = runs_info.get('total', 0)
                cumulative_runs += total_runs

                # Check for wicket event
                wicket = 0
                method = ''
                player_out = ''
                if 'wickets' in delivery and delivery['wickets']:
                    wicket = 1
                    wicket_details = delivery['wickets'][0]  # taking the first wicket if multiple
                    method = wicket_details.get('kind', '')
                    player_out = wicket_details.get('player_out', '')
                    cumulative_wickets += 1

                # Fields not available in JSON are set to default values
                ball_rebowled = 0
                extra_type = "[]"  # default value as in sample CSV

                # For innings runs, wickets and target, compute cumulative values.
                target_score = ''
                runs_to_get = ''
                balls_remaining = ''
                if innings_number == 2:
                    # For the second innings, compute target if available
                    if innings1_total_runs is None:
                        # Use innings 1 cumulative runs if already computed
                        innings1_total_runs = cumulative_runs  # fallback if not set
                    target_score = innings1_total_runs + 1
                    runs_to_get = target_score - cumulative_runs
                    total_balls = info.get('overs', 20) * 6  # default T20: 20 overs * 6 balls
                    balls_remaining = total_balls - ball_counter
                
                # If processing the first innings, capture the total runs at the end
                if innings_number == 1:
                    innings1_total_runs = cumulative_runs

                # For additional columns not in the JSON, we use placeholder defaults.
                chased_successfully = 0
                total_batter_runs = '' 
                total_non_striker_runs = ''
                batter_balls_faced = ''
                non_striker_balls_faced = ''
                player_out_runs = ''
                player_out_balls_faced = ''
                bowler_runs_conceded = total_runs  # ball-wise value; cumulative could be computed separately
                valid_ball = 1

                row = {
                    "Match ID": match_id,
                    "Date": date,
                    "Venue": venue,
                    "Bat First": bat_first,
                    "Bat Second": bat_second,
                    "Innings": innings_number,
                    "Over": over_number,
                    "Ball": ball_index + 1,  # Ball number within the over
                    "Batter": batter,
                    "Non Striker": non_striker,
                    "Bowler": bowler,
                    "Batter Runs": batter_runs,
                    "Extra Runs": extra_runs,
                    "Runs From Ball": total_runs,
                    "Ball Rebowled": ball_rebowled,
                    "Extra Type": extra_type,
                    "Wicket": wicket,
                    "Method": method,
                    "Player Out": player_out,
                    "Innings Runs": cumulative_runs,
                    "Innings Wickets": cumulative_wickets,
                    "Target Score": target_score,
                    "Runs to Get": runs_to_get,
                    "Balls Remaining": balls_remaining,
                    "Winner": winner,
                    "Chased Successfully": chased_successfully,
                    "Total Batter Runs": total_batter_runs,
                    "Total Non Striker Runs": total_non_striker_runs,
                    "Batter Balls Faced": batter_balls_faced,
                    "Non Striker Balls Faced": non_striker_balls_faced,
                    "Player Out Runs": player_out_runs,
                    "Player Out Balls Faced": player_out_balls_faced,
                    "Bowler Runs Conceded": bowler_runs_conceded,
                    "Valid Ball": valid_ball
                }
                rows.append(row)
    
    # Create DataFrame with desired column order
    columns_order = ["Match ID", "Date", "Venue", "Bat First", "Bat Second", "Innings", "Over", "Ball", "Batter", "Non Striker",
                     "Bowler", "Batter Runs", "Extra Runs", "Runs From Ball", "Ball Rebowled", "Extra Type", "Wicket", "Method",
                     "Player Out", "Innings Runs", "Innings Wickets", "Target Score", "Runs to Get", "Balls Remaining", "Winner",
                     "Chased Successfully", "Total Batter Runs", "Total Non Striker Runs", "Batter Balls Faced", "Non Striker Balls Faced",
                     "Player Out Runs", "Player Out Balls Faced", "Bowler Runs Conceded", "Valid Ball"]
    
    df = pd.DataFrame(rows)
    df = df[columns_order]
    df.to_csv(csv_file, index=False)
    print(f"CSV file saved to {csv_file}")

if __name__ == "__main__":
    json_file = "1426303.json"           # Input JSON file containing your match data
    csv_file = "trial_convt.csv"      # Output CSV file
    convert_json_to_csv(json_file, csv_file)
