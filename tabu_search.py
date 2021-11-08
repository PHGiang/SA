import random
import math
import numpy as np
import time
from itertools import combinations
from util import *
import sys

NUMPY_PRECISION = 2

class Tabu:
    def __init__(self, num_city, data, seed=0, limited_time=600):
        # distance input: node_id, x, y
        self.location = data
        self.distance = self.compute_dis_mat(num_city=num_city, location=data)
        self.N = len(self.distance)
        self.best_tour = None
        self.best_cost = float("inf")
        self.seed = seed
        self.limited_time = limited_time
        self.nodes = [i for i in range(self.N)]
        self.cost_history = []

    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = np.round(tmp, NUMPY_PRECISION)
        return dis_mat

    def set_seed(self, seed):
        self.seed = seed

    def tabu(self, curr_tour=None, stopping_criteria=30):
        '''
        Keep tabu list: keep track of recent searches and include
        them into tabu list in order for the algorithm to 'explore'
        different possibilities.

        Steps:
            1. choose a random initial state
            2. enters in a loop checking if a condition to break given
            by the user is met(lower bound)
            3. creates an empty candidate list. Each of the candidates
            in a given neighbor which does not contain a tabu element
            are added to this empty candidate list
            4. It finds the best candidate on this list and if it's cost
            is better than the current best it's marked as a solution.
            5. If the number of tabus on the tabu list have reached the
            maximum number of tabus ( you are defining the number ) a tabu
            expires. The tabus on the list expires in the order they have
            been entered .. first in first out.
        '''
        N = len(self.distance)
        tabu_list = []
        tabu_list_limit = N * 50

        # initialization
        sol_cost = get_total_dist(self, curr_tour)
        neighbor_swap = list(combinations(list(range(N)), 2))

        stop_criterion = 0
        changed = 0
        while time.time() - self.start_time < self.limited_time:
            best_tour, best_cost = [], float("inf")
            # get best solution in the neighbor
            random.shuffle(neighbor_swap)
            for neighbor in neighbor_swap[: len(neighbor_swap) // 3]:
                i, j = neighbor
                # define a neighbor tour
                new_tour = curr_tour.copy()
                new_tour[i: (i + j)] = reversed(new_tour[i: (i + j)])

                new_cost = get_total_dist(self, new_tour)
                if new_cost <= best_cost and new_tour not in tabu_list:
                    best_tour = new_tour
                    best_cost = new_cost

            # stopping criterion:
            if stop_criterion > stopping_criteria and changed <= 10:
                changed += 1
                curr_tour, _ = greedy(self)

            if stop_criterion > stopping_criteria and changed > 10:
                break

            if len(tabu_list) == tabu_list_limit:
                tabu_list.pop()

            if not best_tour:
                best_tour = new_tour  # accpet some worse solution to escape the local maximum
                stop_criterion += 1

            tabu_list.append(best_tour)

            if best_cost < sol_cost:
                curr_tour = best_tour.copy()
                sol_cost = best_cost

        if self.best_cost > sol_cost:
            self.best_cost = sol_cost
            self.best_tour = curr_tour
            self.cost_history.append((round(time.time() - self.start_time, 2), self.best_cost))

    def run(self, times=100, stopping_criteria=10):
        self.start_time = time.time()
        for i in range(1, times + 1):
            if time.time() - self.start_time < self.limited_time:
                print(f"Iteration {i}/{times} -------------------------------")
                greedy_tour, _ = greedy(self)
                self.tabu(curr_tour=greedy_tour, stopping_criteria=stopping_criteria)
                print("Best cost obtained: ", self.best_cost)
                print("Best tour", self.best_tour)

        # return self.best_tour, self.best_cost
        return self.location[self.best_tour], self.best_cost