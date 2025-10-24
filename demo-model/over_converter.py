import pandas as pd

def filter_csv_by_ball_count(input_csv, output_csv, max_balls):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Filter for the desired innings if needed (assuming we work with innings 1 here)
    df_innings1 = df[df['Innings'] == 1].copy()
    
    # Compute the global ball count:
    # GlobalBall = (Over - 1) * 6 + Ball
    df_innings1['GlobalBall'] = (df_innings1['Over'] * 6) + df_innings1['Ball']
    
    # Filter rows where the global ball count is less than or equal to max_balls
    df_filtered = df_innings1[df_innings1['GlobalBall'] <= max_balls].copy()
    
    # Drop the helper column before saving
    df_filtered.drop(columns=['GlobalBall'], inplace=True)
    
    # Save the filtered DataFrame to a new CSV file
    df_filtered.to_csv(output_csv, index=False)
    print(f"Saved filtered CSV with first {max_balls} balls to {output_csv}")

if __name__ == '__main__':
    input_csv = 'trial_convt.csv'
    filter_csv_by_ball_count(input_csv, 'over1.csv', 6)   # First 1 over (6 balls)
    filter_csv_by_ball_count(input_csv, 'over2.csv', 12)  # First 2 overs (12 balls)
    filter_csv_by_ball_count(input_csv, 'over5.csv', 30)  # First 5 overs (30 balls)
    filter_csv_by_ball_count(input_csv, 'over6.csv', 36)  # First 6 overs (36 balls)
    filter_csv_by_ball_count(input_csv, 'over10.csv', 60)  # First 10 overs (60 balls)
    filter_csv_by_ball_count(input_csv, 'over20.csv', 120)  # First 20 overs (120 balls)
