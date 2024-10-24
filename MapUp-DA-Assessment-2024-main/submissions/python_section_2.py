import pandas as pd
import datetime


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here

     # Load the dataset
    #df = pd.read_csv(file_path)

    # Extract unique IDs
    unique_ids = pd.Series(df['id_start'].unique())
    num_ids = len(unique_ids)

    # Initialize a distance matrix with zeros
    distance_matrix = pd.DataFrame(0, index=unique_ids, columns=unique_ids)

    # Fill the distance matrix with known distances
    for _, row in df.iterrows():
        id_a = row['id_start']
        id_b = row['id_end']
        distance = row['distance']  # Assuming there's a 'distance' column in the dataset

        # Set the distance for A to B and B to A
        distance_matrix.at[id_a, id_b] = distance
        distance_matrix.at[id_b, id_a] = distance

    # Calculate cumulative distances (Floyd-Warshall algorithm)
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    return distance_matrix


    #return df
df1 = pd.read_csv('C:/Users/r/Documents/GitHub/Mapup_Assessment_2024/MapUp-DA-Assessment-2024-main/datasets/dataset-2.csv')
distance_df = calculate_distance_matrix(df1)
print(distance_df)


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here

    # Reset index to turn the distance matrix into a long format
    unrolled_df = df.reset_index().melt(id_vars='index', var_name='id_end', value_name='distance')
    
    # Rename the 'index' column to 'id_start'
    unrolled_df.rename(columns={'index': 'id_start'}, inplace=True)
    
    # Filter out rows where distance is zero (if needed)
    unrolled_df = unrolled_df[unrolled_df['distance'] != 0]

    return unrolled_df

    #return df

df2 = distance_df
unroll_dst_df = unroll_distance_matrix(df2)
print(unroll_dst_df)


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here

    # Calculate the average distance for each ID
    avg_distances = df.groupby('id_start')['distance'].mean().reset_index()
    avg_distances.rename(columns={'distance': 'avg_distance'}, inplace=True)
    
    # Get the average distance of the reference ID
    reference_avg_distance = avg_distances.loc[avg_distances['id_start'] == reference_id, 'avg_distance']
    
    if reference_avg_distance.empty:
        return pd.DataFrame()  # Return an empty DataFrame if the reference ID is not found
    
    reference_avg_distance = reference_avg_distance.values[0]
    
    # Calculate the 10% threshold
    lower_bound = reference_avg_distance * 0.9
    upper_bound = reference_avg_distance * 1.1
    
    # Filter the IDs within the threshold
    filtered_ids = avg_distances[(avg_distances['avg_distance'] >= lower_bound) & 
                                  (avg_distances['avg_distance'] <= upper_bound)]
    
    return filtered_ids
    #return df

df3 = unroll_dst_df
result_df = find_ids_within_ten_percentage_threshold(df3, reference_id=1)
print(result_df)


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type and add as new columns
    for vehicle_type, coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * coefficient

    return df


df4 = unroll_dst_df
toll_rate_result = calculate_toll_rate(df4)
print(toll_rate_result)



def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    # Add time columns
    df['start_day'] = 'Monday'  # Placeholder for start day (you can modify as needed)
    df['end_day'] = 'Sunday'     # Placeholder for end day (you can modify as needed)
    df['start_time'] = datetime.time(0, 0)  # Starting at midnight
    df['end_time'] = datetime.time(23, 59, 59)  # Ending just before midnight

    # Define discount factors based on time intervals
    def apply_discount(row):
        start_day = row['start_day']
        start_time = row['start_time']
        
        if start_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            if datetime.time(0, 0) <= start_time < datetime.time(10, 0):
                factor = 0.8
            elif datetime.time(10, 0) <= start_time < datetime.time(18, 0):
                factor = 1.2
            else:
                factor = 0.8
        else:  # Saturday and Sunday
            factor = 0.7
        
        # Adjust rates for all vehicle types based on the discount factor
        for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
            row[vehicle] *= factor
            
        return row

    # Apply the discount factor to each row
    df = df.apply(apply_discount, axis=1)

    return df

df5 = toll_rate_result
time_based_toll_rate_result = calculate_time_based_toll_rates(df5)
print(time_based_toll_rate_result)
