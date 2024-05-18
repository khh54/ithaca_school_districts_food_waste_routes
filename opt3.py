import numpy as np
import csv

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

'''has weight optimized code'''

# read pathpoints csv
data = []
with open('pathpoints.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        data.append(row)

# sort by order (of)
data.sort(key=lambda x: int(x[1]))

# extract school names, orders, and weights
school_names = np.array([row[0] for row in data])
school_order = np.array([int(row[1]) for row in data])
school_weight = np.array([float(row[2]) for row in data])

print("School Names:", school_names)
print("School Order:", school_order)
print("School Weight:", school_weight)

# Read CSV file
data = []
with open('distancematrix.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        data.append(row)

# Create a dictionary to map locations to indices
location_to_index = {'1. dewitt middle school': 1, '10. maint garage': 10, '2. northeast elementary': 2,
                     '3. cayuga heights elementary': 3, '4. belle sherman': 4, '5. caroline elementary': 5,
                     '6. south hill elementary': 6, '7. bjm elementary': 7, '8. fall creek elementary': 8,
                     '9. boynton middle school': 9, '11. bus garage': 11, '12. enfield elementary': 12,
                     '13. lehman alternative': 13, '14. tompkins recycling': 14, 'tst boces tompkins': 0}

# Create an empty distance matrix
num_locations = len(location_to_index)
distance_matrix = np.zeros((num_locations, num_locations))

# Populate the distance matrix
for row in data:
    start_loc = row[0].strip()
    end_loc = row[1].strip()
    distance = float(row[2])
    start_index = location_to_index[start_loc]
    end_index = location_to_index[end_loc]
    distance_matrix[start_index][end_index] = distance

print(distance_matrix)

# Print the distance matrix
# print("Distance Matrix:")
# print(distance_matrix)

# Convert distance matrix to a weighted distance matrix
weighted_distance_matrix = distance_matrix.copy()
for i in range(num_locations):
    for j in range(num_locations):
        weight = school_weight[i] if school_weight[i] > 0 else 1
        weighted_distance_matrix[i][j] *= weight

sorted_indices = np.argsort(school_weight)
print(sorted_indices)

# Sort schools by weight in ascending order
sorted_indices = np.argsort(school_weight)

# Construct route
route = [school_names[0]]  # Start with the first school
for idx in sorted_indices:
    if school_names[idx] not in route:
        route.append(school_names[idx])
route.append(school_names[-1])  # Add the last school

# Calculate total cost of the route
total_cost = 0
for i in range(len(route) - 1):
    start_index = np.where(school_names == route[i])[0][0]
    end_index = np.where(school_names == route[i + 1])[0][0]
    total_cost += distance_matrix[start_index][end_index] * \
        school_weight[start_index]

# Print the optimized route and total cost
print("Optimized Route:", route)
print("Total Cost:", total_cost)
