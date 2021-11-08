import os
import sys
import math
import random
import matplotlib.pyplot as plt

def read(input_name):

    '''
        This function detects whether
    '''
    isGeo = False
    lines = []
    with open(input_name, "r") as file:
        lines = file.readlines()
        for i in range(len(lines)):
            if lines[i].strip()  == "EDGE_WEIGHT_TYPE: GEO":
                isGeo = True
            if lines[i].strip() == "NODE_COORD_SECTION":
                start = i + 1
            if lines[i].strip() == "EOF":
                end = i
    coords = read_coords(lines, start, end)

    if isGeo:
        return read_GEO(coords)
    
def read_GEO(coords):
    # create distance matrix

    length = len(coords)
    distance = [[0 for i in range(length)] for j in range(length)]

    RRR = 6378.388

    for i in range(length):
        for j in range(length):
            if i == j:
                distance[i][j] = float("inf")
            else:
                q1 = math.cos(convertRadiant(coords[i][2]) - convertRadiant(coords[j][2]))
                q2 = math.cos(convertRadiant(coords[i][1]) - convertRadiant(coords[j][1]))
                q3 = math.cos(convertRadiant(coords[i][1]) + convertRadiant(coords[j][1]))
                distance[i][j] = int(RRR*math.acos(0.5*((1.0+q1)*q2 - (1.0-q1)*q3)) + 1.0)
    return distance

def convertRadiant(x):
    PI = 3.141592
    deg = int(x)
    minimum = x - deg
    rad = PI * (deg + 5.0 * minimum/ 3.0) / 180.0
    return rad

def read_coords(lines, start, end):
    coords = []
    for i in range(start, end):
        if lines[i][0] == ' ':
            lines[i] = lines[i][1:]
        node_id, x, y = lines[i].split(" ")[:3]
        coords.append([int(node_id),float(x),float(y)])
    return coords

def greedy(self):
        cur_node = random.randint(0, self.N-1)
        solution = [cur_node]

        remain_nodes = set(self.nodes)
        remain_nodes.remove(cur_node)

        while remain_nodes:
            next_node = min(remain_nodes, key=lambda x: self.distance[cur_node][x])
            remain_nodes.remove(next_node)
            solution.append(next_node)
            cur_node = next_node

        cur_total_dis = get_total_dist(self, solution)
        if cur_total_dis < self.best_cost:
            self.best_cost = cur_total_dis
            self.best_tour = solution
            self.cost_history.append(cur_total_dis)
        return solution, cur_total_dis

def get_total_dist(self, tour):
        cur_total_dis = 0
        for i in range(self.N):
            cur_total_dis += self.distance[tour[i % self.N]][tour[(i + 1) % self.N]]
        return cur_total_dis


def plotTSP(paths, points, num_iters=1):

    """
    path: List of lists with the different orders in which the nodes are visited
    points: coordinates for the different nodes
    num_iters: number of paths that are in the path list

    """

    # Unpack the primary TSP path and transform it into a list of ordered
    # coordinates

    x = []; y = []
    for i in paths[0]:
        x.append(points[i][0])
        y.append(points[i][1])

    plt.plot(x, y, 'co')

    # Set a scale for the arrow heads
    a_scale = float(max(x))/float(100)

    # Draw the older paths, if provided
    if num_iters > 1:

        for i in range(1, num_iters):

            # Transform the old paths into a list of coordinates
            xi = []; yi = [];
            for j in paths[i]:
                xi.append(points[j][0])
                yi.append(points[j][1])

            plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]),
                    head_width = a_scale, color = 'r',
                    length_includes_head = True, ls = 'dashed',
                    width = 0.001/float(num_iters))
            for i in range(0, len(x) - 1):
                plt.arrow(xi[i], yi[i], (xi[i+1] - xi[i]), (yi[i+1] - yi[i]),
                        head_width = a_scale, color = 'r', length_includes_head = True,
                        ls = 'dashed', width = 0.001/float(num_iters))

    # Draw the primary path for the TSP problem
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = a_scale,
            color ='g', length_includes_head=True)
    for i in range(0,len(x)-1):
        plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = a_scale,
                color = 'g', length_includes_head = True)

    #Set axis too slitghtly larger than the set of x and y
    plt.xlim(min(x)*1.1, max(x)*1.1)
    plt.ylim(min(y)*1.1, max(y)*1.1)
    plt.show()

def solution(filename, best_quality, best_tour):
    with open(filename, "w") as file:
        file.write(str(best_quality) + "\n")
        file.write(",".join(map(str, best_tour)))

def solution_trace(filename, cost_history):
    with open(filename, "w") as file:
        for i in range(len(cost_history)):
            if isinstance(cost_history[i], tuple):
                file.write(str(cost_history[i][0]) + "," + str(cost_history[i][1]) + "\n")


if __name__ == "__main__":

    input_file = sys.argv[1]
    distance = read(input_file)
    print(distance[0][1])
