import csv
import json
import os

def csv_file_to_json(csv_file_path, json_file_path=None, orientation='records'):
    """
    Convert a CSV file from your computer to JSON format.
    
    Args:
        csv_file_path (str): Path to the CSV file on your computer
        json_file_path (str, optional): Path to save the JSON file. If None, won't save to file.
        orientation (str, optional): Format of the JSON output. Options:
            - 'records': List of dictionaries (default)
            - 'dict': Dictionary with first column as keys
            - 'split': Dictionary with keys as column names and values as lists
    
    Returns:
        data: JSON data (as Python object)
    """
    # Verify file exists
    if not os.path.isfile(csv_file_path):
        raise FileNotFoundError(f"The file {csv_file_path} was not found.")
    
    # Read CSV file
    data = []
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        # Try to detect dialect
        try:
            dialect = csv.Sniffer().sniff(csv_file.read(1024))
            csv_file.seek(0)
        except:
            dialect = 'excel'  # Default dialect
            csv_file.seek(0)
        
        # Read as dictionary
        csv_reader = csv.DictReader(csv_file, dialect=dialect)
        for row in csv_reader:
            # Convert empty strings to None if needed
            clean_row = {k: (v if v != "" else None) for k, v in row.items()}
            data.append(clean_row)
    
    # Convert to specified JSON orientation
    if orientation == 'records':
        json_data = data
    elif orientation == 'dict':
        # Use first column as key
        if data:
            first_key = list(data[0].keys())[0]
            json_data = {row[first_key]: row for row in data}
        else:
            json_data = {}
    elif orientation == 'split':
        if not data:
            json_data = {"columns": [], "data": []}
        else:
            columns = list(data[0].keys())
            values = []
            for row in data:
                values.append([row[col] for col in columns])
            json_data = {"columns": columns, "data": values}
    else:
        raise ValueError("Invalid orientation. Choose 'records', 'dict', or 'split'")
    
    # Write JSON file if path is provided
    if json_file_path:
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=4)
        print(f"JSON data saved to {json_file_path}")
    
    return json_data

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    csv_path = "over1.csv"
    
    # Optional: path to save the JSON output
    json_path = "over1.json"
    
    try:
        # Convert without saving
        json_data = csv_file_to_json(csv_path)
        print("CSV successfully converted to JSON in memory")
        
        # Display the first 2 records (if available)
        if isinstance(json_data, list) and len(json_data) > 0:
            print(f"\nFirst {min(2, len(json_data))} records:")
            for i, record in enumerate(json_data[:2]):
                print(f"Record {i+1}:", json.dumps(record, indent=2))
        
        # Convert and save to file
        json_data = csv_file_to_json(csv_path, json_path)
        print(f"CSV converted and saved to {json_path}")
        
    except Exception as e:
        print(f"Error: {e}")