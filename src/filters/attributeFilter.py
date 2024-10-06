# Program to filter data to include only important components

import pandas as pd

input_file = '../../datasets/master.csv' 
output_file = '../../datasets/main.csv' 

# List of columns to keep
columns_to_keep = [
    'Target',
    'Insulin Levels',
    'Age',
    'BMI',
    'Waist Circumference',
    'Cholesterol Levels',
    'Blood Glucose Levels',
    'Pancreatic Health'
]

try:
    df = pd.read_csv(input_file)

    # Filter the DataFrame to keep only the specified columns
    filtered_df = df[columns_to_keep]
    filtered_df.to_csv(output_file, index=False)
    
    print(f'Filtered data has been saved to {output_file}.')

except FileNotFoundError:
    print(f'The file {input_file} was not found. Please check the file path.')
except Exception as e:
    print(f'An error occurred: {e}')
