import random
import math
import numpy as np
# import simulated_annealing as sa
import iterated_local_search as ils
# import tabu_search as tb
import matplotlib.pyplot as plt
from datetime import datetime
import csv
# from memory_profiler import memory_usage

import os, psutil


def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data

if __name__=='__main__':
    data = read_tsp(r'./input/tsp10.txt')
    f = open('./output_ils/out10.csv', 'w')

    # data = read_tsp(r'./input/tsp100.txt')
    # f = open('./output_ils/out100.csv', 'w')

    # data = read_tsp(r'./input/tsp1000.txt')
    # f = open('./output_ils/out1000.csv', 'w')

    data = np.array(data)
    data = data[:, 1:]
    show_data = np.vstack([data, data[0]])
    Best, Best_path = math.inf, None

    # model = sa.SA(num_city=data.shape[0], data=data.copy())
    # path, path_len = model.run()

    # model = tb.Tabu()
    # path, path_len = model.run()
    
    process = psutil.Process(os.getpid())
    model = ils.ILS(num_city=data.shape[0], data=data.copy())

    no_iterations = 10
    tsp_runtime = []
    tsp_memory = []
    tsp_path_len = []
    sep = "========================="
    writer = csv.writer(f)
    for i in range(no_iterations):
        start_time = datetime.now()

        path, path_len = model.run()
        
        runtime = datetime.now() - start_time
        tsp_runtime.append(runtime.microseconds)
        used_mem = (process.memory_info().rss)/(1024**2)
        tsp_memory.append(used_mem)
        tsp_path_len.append(path_len)
        row = [['iteration ', i], [path_len], path, [runtime.microseconds], [used_mem], model.iter_y]
        writer.writerows(row)
        writer.writerow(sep)
        # if path_len < Best:
        #     Best = path_len
        #     Best_path = path
        Best_path = path 
        Best_path = np.vstack([Best_path, Best_path[0]])

    #     plot1 = plt.figure()
    #     plt.scatter(Best_path[:, 0], Best_path[:,1])
    #     plt.plot(Best_path[:, 0], Best_path[:, 1], 'ro', Best_path[:, 0], Best_path[:, 1])
    #     plt.title('Optimization tour of TSP 10 - ILS')
    #     plt.xlabel(f'Total mileage of the tour: {path_len}')
    #     plt.pause(0.05)
    #     #

    #     plot2 = plt.figure()
    #     iterations = model.iter_x
    #     best_record = model.iter_y
    #     plt.plot(iterations, best_record)
    #     plt.title('Optimization result of TSP 10 - ILS')
    #     plt.xlabel('Iterations')
    #     plt.ylabel('Mileage of tour')
    #     plt.pause(0.05)
        
    # plt.show()
    writer.writerow('==== Summation =======')
    writer.writerows([tsp_path_len, tsp_runtime, tsp_memory])
    f.close()