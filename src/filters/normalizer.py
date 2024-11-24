import pandas as pd

def balance_dataset(input_file, output_file):
    # Load the data from the CSV file
    data = pd.read_csv(input_file)
    
    # Check the distribution of the target variable
    target_counts = data['Target'].value_counts()
    print("Target distribution before balancing:")
    print(target_counts)
    
    # Separate the data by target value
    target_values = data['Target'].unique()
    data_balanced = []
    
    # Find the smallest class size to balance around
    min_class_size = target_counts.min()
    
    for target_value in target_values:
        # Select rows with the current target value
        target_data = data[data['Target'] == target_value]
        
        # Down-sample the class to the size of the smallest class
        target_data_resampled = target_data.sample(n=min_class_size, random_state=42)
        
        # Add the resampled data to the balanced dataset list
        data_balanced.append(target_data_resampled)
    
    # Concatenate all the resampled data
    balanced_data = pd.concat(data_balanced)
    
    # Shuffle the rows to avoid any ordering bias
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the balanced dataset to a new CSV file
    balanced_data.to_csv(output_file, index=False)
    
    # Check the distribution of the target variable after balancing
    print("Target distribution after balancing:")
    print(balanced_data['Target'].value_counts())

# Example usage
input_file = '../../datasets/data.csv'  # Replace with your input file path
output_file = '../../datasets/balanced_data.csv'  # Replace with your desired output file path
balance_dataset(input_file, output_file)
