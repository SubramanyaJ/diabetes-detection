import pandas as pd

# Input and output file paths
input_file = '/diabetes_detection/datasets/master.csv'
final_output_file = '/diabetes_detection/datasets/new.csv'

# List of columns to keep
columns_to_keep = [
    'Target',
    'Insulin Levels',
    'Age',
    'BMI',
    'Cholesterol Levels',
    'Blood Glucose Levels',
    'Pancreatic Health',
    'Digestive Enzyme Levels'
]

# Mapping for the Target column
target_mapping = {
    'Prediabetic': 0,
    'Type 1 Diabetes': 1,
    'Type 2 Diabetes': 2,
    'Type 3c Diabetes (Pancreatogenic Diabetes)': 3
}

def process_data(input_file, final_output_file, columns_to_keep, target_mapping):
    try:
        # Step 1: Read the original CSV file
        df = pd.read_csv(input_file)

        # Step 2: Filter columns
        filtered_df = df[columns_to_keep]

        # Step 3: Filter rows based on user input
        attribute_name = "Target"
        values_list = ['Prediabetic', 'Type 1 Diabetes', 'Type 2 Diabetes', 'Type 3c Diabetes (Pancreatogenic Diabetes)']

        if attribute_name in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[attribute_name].isin(values_list)]
        else:
            print(f'The attribute "{attribute_name}" does not exist in the dataset.')
            return

        # Step 4: Convert Target column to numerical values
        filtered_df['Target'] = filtered_df['Target'].map(target_mapping)

        # Step 5: Save the final DataFrame to the specified output file
        filtered_df.to_csv(final_output_file, index=False)
        print(f'Processed data has been saved to {final_output_file}.')

    except FileNotFoundError:
        print(f'The file {input_file} was not found. Please check the file path.')
    except Exception as e:
        print(f'An error occurred: {e}')

# Run the data processing function
process_data(input_file, final_output_file, columns_to_keep, target_mapping)
