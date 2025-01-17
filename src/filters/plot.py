import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_csv(file_path):
    # Load data
    data = pd.read_csv(file_path)

    # Display available columns
    print("Available columns in the dataset:")
    print(data.columns.tolist())

    # Ask user for required fields
    required_fields = input("Enter the required fields (comma-separated): ").split(',')
    required_fields = [field.strip() for field in required_fields]

    # Validate fields
    for field in required_fields:
        if field not in data.columns:
            print(f"Field '{field}' not found in the dataset.")
            return

    # Filter data for required fields
    selected_data = data[required_fields]

    # Display statistical measures
    print("\nStatistical Measures:")
    print(selected_data.describe())

    # Plot distribution curves for each required field
    for field in required_fields:
        plt.figure(figsize=(10, 6))
        sns.histplot(selected_data[field], kde=True, color='blue', bins=30)

        # Calculate statistical measures
        mean = selected_data[field].mean()
        median = selected_data[field].median()
        mode = selected_data[field].mode()[0]
        std_dev = selected_data[field].std()

        # Add statistical measures to the plot
        plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
        plt.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')
        plt.axvline(mode, color='orange', linestyle='--', label=f'Mode: {mode:.2f}')
        plt.text(mean, plt.ylim()[1]*0.9, f"Std Dev: {std_dev:.2f}", color='purple')

        plt.title(f"Distribution Curve for {field}")
        plt.xlabel(field)
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.show()

analyze_csv('../../datasets/data.csv')
