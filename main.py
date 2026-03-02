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

current_solution = np.asarray([0])
curr_sol_cap = 0

# pick the best n solutions
coords_to_keep = 5
last_coord = current_solution[-1]

# print(calculate_distance)

while True:
    dist_bw_coords = calculate_distance(coords)[last_coord]
    best_dists = np.sort(dist_bw_coords)[1:coords_to_keep+1] # starting from 1 because we want to ignore dist 0
    best_dists_idx = np.nonzero(np.isin(dist_bw_coords, best_dists) == True)[0]
    # pick a random solution
    rand_idx = np.random.randint(coords_to_keep)
    while best_dists_idx[rand_idx] in current_solution:
        rand_idx = np.random.randint(coords_to_keep)
    if curr_sol_cap + cap_dem[best_dists_idx[rand_idx]] > cap_dem[0]:
        current_solution = np.append(current_solution, 0)
        break
    current_solution = np.append(current_solution, best_dists_idx[rand_idx])
    curr_sol_cap += cap_dem[best_dists_idx[rand_idx]]
    last_coord = current_solution[-1]
print(current_solution, curr_sol_cap)