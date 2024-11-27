# Program to filter data to include only important components

import pandas as pd

input_file = 'master.csv' 
output_file = 'attributesFiltered.csv' 

# List of columns to keep
columns_to_keep = [
    'Target',
    'Insulin Levels',
    'Age',
    'BMI',
    # 'Waist Circumference',
    'Cholesterol Levels',
    'Blood Glucose Levels',
    'Pancreatic Health',
    # 'Neurological Assessments',
    # 'Glucose Tolerance Test',
    'Digestive Enzyme Levels'
    
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

# Program to filter tuples based on Target values

import pandas as pd

# Specify the input file name
input_file = 'attributesFiltered.csv'  # Change this to your actual input file path

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
        output_file = 'newtargetsFiltered.csv'  # Specify your desired output file name
        filtered_df.to_csv(output_file, index=False)
        print(f'Filtered data has been saved to {output_file}.')
    else:
        print("No matching records found or an error occurred.")

# Converts the string data in the 'Type' field into numerical values

import pandas as pd

def convert_and_save(input_file, output_file):
    df = pd.read_csv(input_file)
    
    # Define the mapping for the Target column
    target_mapping = {
#       'No Diabetes': 0,
        'Prediabetic': 0,
        'Type 1 Diabetes': 1,
        'Type 2 Diabetes': 2,
        'Type 3c Diabetes (Pancreatogenic Diabetes)' : 3,
        #'Secondary Diabetes': 4
        
    }
    
    # Convert the Target column using the mapping
    df['Target'] = df['Target'].map(target_mapping)
    
    # gtt_mapping = {
    #     'Normal' : 0,
    #     'Abnormal' : 1
    # }

    #  df['Glucose Tolerance Test'] = df['Glucose Tolerance Test'].map(gtt_mapping)    

    # Save the modified dataframe to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

    

# Example usage
input_file_path = 'newtargetsFiltered.csv'  
output_file_path = 'newdata1.csv' 
convert_and_save(input_file_path, output_file_path)

# Target
# Prediabetic, Type 1 Diabetes, Type 2 Diabetes, Secondary Diabetes,Type 3c Diabetes (Pancreatogenic Diabetes)