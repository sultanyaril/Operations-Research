import pandas as pd
import math
from collections import namedtuple, defaultdict
import time
from ortools.sat.python import cp_model

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def solve_it(input_data, required_solution, use_memory=True):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    # build a trivial solution
    # Define the problem parameters
    distances = defaultdict(dict)
    if use_memory:
        for i in range(facility_count):
            for j in range(customer_count):
                distances[i][j] = length(facilities[i].location, customers[j].location)

    # Create the model
    model = cp_model.CpModel()
    
    # Define the decision variables
    facilities_var = [model.NewIntVar(0, 1, f'facility_{i}') for i in range(facility_count)]
    customers_var = [[model.NewIntVar(0, 1, f'customer_{j}{i}') for i in range(customer_count)] for j in range(facility_count)]
    
    # Define the constraints
    for i in range(customer_count):
        model.Add(sum(customers_var[j][i] for j in range(facility_count)) == 1)
        
    for i in range(facility_count):
        model.Add(sum(customers_var[i][j] * customers[j].demand for j in range(customer_count)) <= facilities[i].capacity)

    for i in range(facility_count):
        for j in range(customer_count):
            model.Add(customers_var[i][j] <= facilities_var[i])
        
    # Define the objective function
    fixed_cost = sum(facilities_var[i] * facilities[i].setup_cost for i in range(facility_count))
    if use_memory:
        transport_cost = sum(distances[i][j] * customers_var[i][j] for i in range(facility_count) for j in range(customer_count))
    else:
        transport_cost = sum(length(facilities[i].location, customers[j].location) * customers_var[i][j] for i in range(facility_count) for j in range(customer_count))
    objective = model.Minimize(fixed_cost + transport_cost)    
    
    # Solve the problem
    
    class SolutionCallback(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.curr_min = float('inf')

        def on_solution_callback(self):
            if self.ObjectiveValue() < self.curr_min:
                self.curr_min = self.ObjectiveValue()
                print('NEW MIN', self.curr_min)
                if self.curr_min < required_solution:
                    self.StopSearch()

    solver = cp_model.CpSolver()    
    solution_callback = SolutionCallback()
    status = solver.Solve(model, solution_callback)

    # Print the solution
    solution = []
    for j in range(customer_count):
        for i in range(facility_count):
            if solver.Value(customers_var[i][j]):
                solution.append(i)
    
    return solution


import sys

def get_result(file_name, reqired_solution, use_memory=True):
    with open('data/'+file_name, 'r') as input_data_file:
        input_data = input_data_file.read()
    solution = solve_it(input_data, reqired_solution, use_memory)
    print(solution)
    res = pd.DataFrame(solution, columns=['Order'])
    res.to_excel(file_name+'_res.xlsx', index=False, header=True)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 3:
        file_name = sys.argv[1].strip()
        required_solution = int(sys.argv[2].strip())
        use_memory = bool(sys.argv[3].strip())
        get_result(file_name, required_solution, use_memory)
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')