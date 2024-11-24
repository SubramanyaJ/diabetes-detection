# Program to filter tuples based on Target values

import pandas as pd

# Specify the input file name
input_file = '../../datasets/attributesFiltered.csv'  # Change this to your actual input file path

# Function to filter rows based on user input
def filter_by_attribute(attribute, values):
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)

        # Check if the attribute exists in the DataFrame
        if attribute in df.columns:
            # Filter the DataFrame based on the values
            filtered_df = df[df[attribute].isin(values)]
            return filtered_df
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

    # Read values to filter by, separated by commas
    values_input = input("Enter the values to filter by, separated by commas: ")
    values_list = [value.strip() for value in values_input.split(",")]

    # Get the filtered DataFrame
    filtered_df = filter_by_attribute(attribute_name, values_list)

    # Write the filtered DataFrame to a new CSV file if not None
    if filtered_df is not None and not filtered_df.empty:
        output_file = '../../datasets/targetsFiltered.csv'  # Specify your desired output file name
        filtered_df.to_csv(output_file, index=False)
        print(f'Filtered data has been saved to {output_file}.')
    else:
        print("No matching records found or an error occurred.")

