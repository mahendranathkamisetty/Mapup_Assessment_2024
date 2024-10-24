import re
from typing import Dict, Any, List
import pandas as pd
import math
import polyline
import numpy as np


def reverse_by_n_elements(lst: list[int], n: int) -> list[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    res = []
    for i in range(0, len(lst), n):
        group = []

        for j in range(n):
            if i+j < len(lst):
                group.append(lst[i+j])

        for k in range(len(group)-1, -1, -1):
            res.append(group[k])
    
    
    return res

# lst = [10, 20, 30, 40, 50, 60, 70]
# n=4
# print(reverse_by_n_elements(lst, n))



def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here

    length_dict = {}
    for string in lst:
        length = len(string)

        if length not in length_dict:
            length_dict[length] = []

        length_dict[length].append(string)
    
    # return length_dict

    sorted_dict = {k: length_dict[k] for k in sorted(length_dict)}
    return sorted_dict
    #return dict

# list_strings = ["apple", "bat", "car", "elephant", "dog", "bear"]
# print(group_by_length(list_strings))


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here

    dict = {}


    if not nested_dict:
        return dict

    def _flatten(current_dict: Dict[str, Any], parent_key: str):
        for key, value in current_dict.items():
           
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, Dict):
                _flatten(value, new_key)
            elif isinstance(value, list):
                
                for index, item in enumerate(value):
                    _flatten({f"{new_key}[{index}]": item}, '')
            else:
                dict[new_key] = value

    _flatten(nested_dict, '')
    
    return dict

# nested_input = {
#     "road": {
#         "name": "Highway 1",
#         "length": 350,
#         "sections": [
#             {
#                 "id": 1,
#                 "condition": {
#                     "pavement": "good",
#                     "traffic": "moderate"
#                 }
#             }
#         ]
#     }
# }

# flattened_result = flatten_dict(nested_input)
# print(flattened_result)


def unique_permutations(nums: list[int]) -> list[list[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: list of integers (may contain duplicates)
    :return: list of unique permutations
    """
    # Your code here
    def backtrack(path, counter):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for num in counter:
            if counter[num] > 0:
                path.append(num)
                counter[num] -= 1
                backtrack(path, counter)
                path.pop()
                counter[num] += 1

    result = []
    counter = {}
    for num in nums:
        counter[num] = counter.get(num, 0) + 1
    backtrack([], counter)
    return result
    pass

# list = [1, 1, 2]
# print(unique_permutations(list))



def find_all_dates(text: str) -> list[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    list[str]: A list of valid dates in the formats specified.
    """
    pattern = r'''(\d{2}-\d{2}-\d{4} | \d{2}/\d{2}/\d{4} | \d{4}\.\d{2}\.\d{2})'''
    matches = re.findall(pattern, text, re.VERBOSE)
    return matches
    #pass

# text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
# print(find_all_dates(text))


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str)

    # Create a DataFrame with latitude and longitude
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    # Calculate distances
    distances = [0.0]  # First point has no previous point
    R = 6371000  # Radius of the Earth in meters

    for i in range(1, len(df)):
        lat1, lon1 = np.radians(df.iloc[i-1][['latitude', 'longitude']])
        lat2, lon2 = np.radians(df.iloc[i][['latitude', 'longitude']])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        distance = R * c
        distances.append(distance)

    # Add distance column to the DataFrame
    df['distance'] = distances

    return df
    #return pd.Dataframe()


# polyline_str = "_p~iF~wvy@_@~fJv~i@kB"      #give your polyline encoded string
# df = polyline_to_dataframe(polyline_str)
# print(df)



def rotate_and_multiply_matrix(matrix: list[list[int]]) -> list[list[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (list[list[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - list[list[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    # Step 2: Create a new matrix for the final result
    final_matrix = [[0] * n for _ in range(n)]
    
    # Calculate the row and column sums
    row_sums = [sum(rotated_matrix[i]) for i in range(n)]

    col_sums = [sum(rotated_matrix[j][i] for j in range(n)) for i in range(n)]
    
    
    # Step 3: Replace each element by sum of all rows and cols excluding itself (adding two times self number so remove two times)
    for i in range(n):
        for j in range(n):
            final_matrix[i][j] = row_sums[i] + col_sums[j] - rotated_matrix[i][j] - rotated_matrix[i][j]
    
    return final_matrix

    #return []


# def rotate_and_transform_matrix(matrix: list[list[int]]) -> list[list[int]]:
#     """
#     Rotate the given matrix by 90 degrees clockwise, then replace each element
#     with the sum of all elements in the same row and column, excluding itself.
    
#     Args:
#     - matrix (list[list[int]]): 2D list representing the matrix to be transformed.
    
#     Returns:
#     - list[list[int]]: The final transformed matrix.
#     """

#     n = len(matrix)
    
#     # Step 1: Rotate the matrix by 90 degrees clockwise
#     rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
#     # Step 2: Create a new matrix for the final result
#     final_matrix = [[0] * n for _ in range(n)]
    
#     # Calculate the row and column sums
#     row_sums = [sum(rotated_matrix[i]) for i in range(n)]

#     col_sums = [sum(rotated_matrix[j][i] for j in range(n)) for i in range(n)]
    
    
#     # Step 3: Replace each element by sum of all rows and cols excluding itself (adding two times self number so remove two times)
#     for i in range(n):
#         for j in range(n):
#             final_matrix[i][j] = row_sums[i] + col_sums[j] - rotated_matrix[i][j] - rotated_matrix[i][j]
    
#     return final_matrix


# matrix = [[1, 2, 3],
#           [4, 5, 6],
#           [7, 8, 9]]
# result = rotate_and_multiply_matrix(matrix)
# #result = rotate_and_transform_matrix(matrix)
# print(result)



def time_check(df) -> pd.Series:
    """
    Use shared dataset-1 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair 
    cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    grouped = df.groupby(['id', 'id_2'])

    def check_group(group):
        # Create a set for days of the week
        required_days = {0, 1, 2, 3, 4, 5, 6}  # Monday to Sunday
        days_covered = set(group['start_datetime'].dt.dayofweek.unique())
        
        # Check if all days are covered
        if not required_days.issubset(days_covered):
            return False
        
        # Check each day for the 24-hour coverage
        for day in range(7):  # Check for all 7 days
            day_entries = group[group['start_datetime'].dt.dayofweek == day]
            if not day_entries.empty:
                day_start = day_entries['start_datetime'].min()
                day_end = day_entries['end_datetime'].max()
                if (day_end - day_start) < pd.Timedelta(hours=24):
                    return False

        return True
    
    result = grouped.apply(check_group)

    return result


    #return pd.Series()

# df = pd.read_csv('C:/Users/r/Documents/GitHub/Mapup_Assessment_2024/MapUp-DA-Assessment-2024-main/datasets/dataset-1.csv')
# result = time_check(df)
# print(result)
