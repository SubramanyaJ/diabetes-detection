# Program to list range of values for an attribute

import pandas as pd

input_file = '../../datasets/master.csv'

# Function to get unique values for a given attribute
def get_unique_values(attribute):
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)

        # Check if the attribute exists in the DataFrame
        if attribute in df.columns:
            # Get unique values for the specified attribute
            unique_values = df[attribute].unique()
            return unique_values
        else:
            print(f'The attribute "{attribute}" does not exist in the CSV file.')
            return None
    except FileNotFoundError:
        print(f'The file {input_file} was not found. Please check the file path.')
        return None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None

# Main program
if __name__ == "__main__":
    # Read attribute name from user
    attribute_name = input("Enter the attribute name: ")
    unique_values = get_unique_values(attribute_name)

    # Print unique values if found
    if unique_values is not None:
        print(f'Unique values for "{attribute_name}":')
        for value in unique_values:
            print(value)

