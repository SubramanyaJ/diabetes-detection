import pandas as pd

def remove_percentage_of_target(file_path, percentage, target_value, output_file="../../datasets/new.csv"):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Filter rows matching the target value
    target_rows = data[data['Target'] == target_value]
    
    # Calculate the number of rows to remove
    num_to_remove = int(len(target_rows) * (percentage / 100))
    
    # Randomly select rows to remove
    rows_to_remove = target_rows.sample(n=num_to_remove, random_state=42)
    
    # Create the new dataset by excluding the selected rows
    modified_data = data.drop(rows_to_remove.index)
    
    # Save the modified dataset to a new file
    modified_data.to_csv(output_file, index=False)
    print(f"Modified dataset saved to: {output_file}")

# Get inputs from the user
file_path = "../../datasets/master.csv" 
percentage = float(input("Enter the percentage of rows to remove: "))
target_value = input("Enter the target value to filter rows (e.g., 'Prediabetic'): ")

# Call the function
remove_percentage_of_target(file_path, percentage, target_value)

