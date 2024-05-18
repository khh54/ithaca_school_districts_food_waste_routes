import numpy as np
import csv

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# read pathpoints csv
data = []
with open('pathpoints.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip header
    for row in reader:
        data.append(row)

# sort by order (of)
data.sort(key=lambda x: int(x[1]))

# extract school names, orders, and weights
school_names = np.array([row[0] for row in data])
school_order = np.array([int(row[1]) for row in data])


def create_data_model(csv_string):
    # read CSV file
    data = []
    with open(csv_string, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            data.append(row)

    # create a dictionary to map locations to indices
    location_to_index = {'1. dewitt middle school': 1, '10. maint garage': 10, '2. northeast elementary': 2,
                         '3. cayuga heights elementary': 3, '4. belle sherman': 4, '5. caroline elementary': 5,
                         '6. south hill elementary': 6, '7. bjm elementary': 7, '8. fall creek elementary': 8,
                         '9. boynton middle school': 9, '11. bus garage': 11, '12. enfield elementary': 12,
                         '13. lehman alternative': 13, '14. tompkins recycling': 14, 'tst boces tompkins': 0}

    # create an empty distance matrix
    num_locations = len(location_to_index)
    distance_matrix = np.zeros((num_locations, num_locations))

    # populate the distance matrix
    for row in data:
        start_loc = row[0].strip()
        end_loc = row[1].strip()
        distance = float(row[2])
        start_index = location_to_index[start_loc]
        end_index = location_to_index[end_loc]
        distance_matrix[start_index][end_index] = distance

    num_nodes = len(distance_matrix)
    # add an edge from the end node to the start node with 0 cost (for tsp)
    distance_matrix[num_nodes - 1][0] = 0

    # add edges from every other node to the start node with very high cost (for tsp)
    high_cost = float('inf')
    for i in range(1, num_nodes - 1):
        distance_matrix[i][0] = high_cost

    data = {}
    distance_matrix = distance_matrix.astype(int)

    data["distance_matrix"] = distance_matrix.tolist()
    data["num_vehicles"] = 1
    data["depot"] = 0  # start point
    return data


def return_solution(manager, routing, solution):
    """Prints solution on console."""
    print(f"Emissions: {solution.ObjectiveValue()}")
    index = routing.Start(0)
    plan_output = "Route:\n"
    route_distance = 0
    route_indices = []
    while not routing.IsEnd(index):
        plan_output += f" {school_names[manager.IndexToNode(index)]} ->"
        route_indices.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(
            previous_index, index, 0)
    plan_output += f" {school_names[manager.IndexToNode(index)]}\n"
    route_indices.append(manager.IndexToNode(index))
    print(plan_output)
    plan_output += f"Route distance: {route_distance}miles\n"
    return route_indices


def calculate_cost(route, distance_matrix):
    """calculates cost of a given route"""
    cost = 0
    for i in range(len(route) - 1):
        cost += distance_matrix[route[i]][route[i+1]]
    return cost


def find_opt_route(csv_name):
    # instantiate the data problem
    data = create_data_model(csv_name)

    # create the routing index manager
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # create routing model
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # returns solution
    opt_indices = return_solution(manager, routing, solution)

    return opt_indices


def main():
    """Entry point of the program."""

    emissions_data = create_data_model('emissions.csv')
    emissions_data = emissions_data["distance_matrix"]
    distance_data = create_data_model('distance.csv')
    distance_data = distance_data["distance_matrix"]

    opt_emissions = find_opt_route('emissions.csv')
    opt_emissions.pop()
    opt_dist = find_opt_route('distance.csv')
    opt_dist.pop()
    opt_weight = find_opt_route('weight_matrix.csv')
    opt_weight.pop()
    original_route = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    print("emissions optimized route: ")
    print(opt_emissions)
    print("emissions route cost: ")
    print(calculate_cost(opt_emissions, emissions_data))
    print(calculate_cost(opt_emissions, distance_data))

    print("\ndistance optimized route: ")
    print(opt_dist)
    print("dist route cost: ")
    print(calculate_cost(opt_dist, emissions_data))
    print(calculate_cost(opt_dist, distance_data))

    def calculate_average(low, high):
        return (low + high) / 2

    # read CSV and store data in a dictionary indexed by school index
    data = {}
    with open('weights.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            school_name = row[0]
            # extract index from school name
            index = int(school_name.split('.')[0])
            low_weight = float(row[1])
            high_weight = float(row[2])
            average_weight = calculate_average(low_weight, high_weight)
            data[index] = average_weight

    # Sort the dictionary by weight
    sorted_data = sorted(data.items(), key=lambda x: x[1])

    # Get a list of indices corresponding to the sorted weights
    opt_weight = [None] * 15
    opt_weight = [index for index, _ in sorted_data]
    opt_weight.append(14)

    print("\nweight optimized route: ")
    print(opt_weight)
    print("weight route cost: ")
    print(calculate_cost(opt_weight, emissions_data))
    print(calculate_cost(opt_weight, distance_data))

    print("\noriginal route: ")
    print(original_route)
    print("original cost: ")
    print(calculate_cost(original_route, emissions_data))


if __name__ == "__main__":
    main()
