import pandas as pd

# Read the CSV file
df = pd.read_csv('diabetes_dataset00.csv')

# Filter rows where 'Age' is below 35
filtered_df = df[df['Age'] < 35]

# Write the filtered data to a new CSV file
filtered_df.to_csv('filtered_data_below_35.csv', index=False)