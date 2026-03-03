import numpy as np
import matplotlib.pyplot as plt

def cvrp_read(file_name):
    coords = np.asarray([])
    cap_dem = np.asarray([])

    with open(file_name, "r") as file:
        for line in file:
            if len(line.strip().split()) == 1:
                veh_cap = line.strip()
            elif len(line.strip().split()) == 2:
                a, b = line.strip().split()
                coords = np.concatenate([coords, [a, b]], axis=0)
                cap_dem = np.append(cap_dem, veh_cap)
            else:
                a, b, dem = line.strip().split()
                coords = np.concatenate([coords, [a, b]], axis=0)
                cap_dem = np.append(cap_dem, dem)
        coords = np.reshape(coords, (int(len(coords)/2), 2)).astype("int")
        cap_dem = cap_dem.astype("int")
    return coords, cap_dem

def cvrp_plot(coords, cap_dem):
    plt.scatter(coords[0,0], coords[0,0], c="#e00000", s=cap_dem[0])
    plt.scatter(coords[1:,0], coords[1:,1], c="#120cbf", s=cap_dem[1:])
    plt.show()

def calculate_distance(coords, method="euclidean"):
    distance_matrix = np.zeros((len(coords), len(coords)))
    for j in range(len(coords)):
        for i in range(len(coords)):
            
            if method == "euclidean":
                distance_matrix[i][j] = ((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)**0.5
            else:
                distance_matrix[i][j] = np.abs(coords[i, 0] - coords[j, 0]) + np.abs(coords[i, 1] - coords[j, 1])
    return distance_matrix

coords, cap_dem = cvrp_read("instances/cvrp.txt")

complete_solution = {
    0: [0]
 }

count = 0
coords_to_keep = 5

while ((np.isin(np.nonzero(calculate_distance(coords)[0] != 0)[0], np.concatenate([complete_solution[i] for i in range(len(complete_solution))])) == True) != True).any():
    current_solution = np.asarray([0])
    curr_sol_cap = 0

    last_coord = current_solution[-1]

    while True:
        # Get all unvisited nodes (all except node 0 and those already in complete_solution or current_solution)
        visited_nodes = np.concatenate([complete_solution[i] for i in range(len(complete_solution))])
        visited_nodes = np.concatenate([visited_nodes, current_solution])
        all_nodes = np.arange(len(coords))
        unvisited_nodes = np.setdiff1d(all_nodes, visited_nodes)
        
        # Get distances from last_coord to all unvisited nodes
        distances_to_unvisited = calculate_distance(coords)[last_coord][unvisited_nodes]
        
        # If no unvisited nodes left, close this route
        if len(unvisited_nodes) == 0:
            current_solution = np.append(current_solution, 0)
            complete_solution[count] = current_solution
            count += 1
            break
        
        # Select k-nearest unvisited nodes
        num_candidates = min(coords_to_keep, len(unvisited_nodes))
        best_distances_indices = np.argsort(distances_to_unvisited)[:num_candidates]
        best_unvisited_nodes = unvisited_nodes[best_distances_indices]
        
        # Pick a random node from the k-nearest
        rand_idx = np.random.randint(num_candidates)
        next_node = best_unvisited_nodes[rand_idx]
        
        # Check capacity constraint
        if curr_sol_cap + cap_dem[next_node] > cap_dem[0]:
            current_solution = np.append(current_solution, 0)
            complete_solution[count] = current_solution
            count += 1
            break
        
        current_solution = np.append(current_solution, next_node)
        curr_sol_cap += cap_dem[next_node]
        last_coord = next_node
    # print(complete_solution)