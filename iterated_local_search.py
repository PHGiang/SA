# -*- coding: utf-8 -*-

import os
import random
import numpy
from itertools import combinations
from datetime import datetime
import simulated_annealing as sa

NUMPY_PRECISION = 2
numpy.set_printoptions(precision=NUMPY_PRECISION)


def randomize_tour(length):
    tour = []
    tour.append(0)
    # random_tour = range(1, length)
    random_tour = [x for x in range(1, length)]
    random.shuffle(random_tour)
    tour += random_tour
    return tour


class ILS:
    def __init__(self, num_city, data):
        self.data = data
        self.num_city = num_city
        self.iteration_limit = 200
        self.alternative = True
        self.is_SA_enabled = False
        self.idle_limit = 50

        self.dist_matrix = None
        self.iterations = 0
        self.runtime = ""
        self.solutions = []
        self.best_solution = {}
        self.alternative_counter = []
        self.iter_x = []
        self.iter_y = []
        self.model = sa.SA(num_city=num_city, data=data)

    def reset(self):
        self.iterations = 0
        self.runtime = ""
        self.solutions = []
        self.best_solution = {}
        self.alternative_counter = []
        self.iter_x = []
        self.iter_y = []

    def set_parameters(self, iteration_limit, alternative, idle_limit):
        self.iteration_limit = iteration_limit
        self.idle_limit = idle_limit
        self.alternative = alternative

    def calc_dist_matrix(self):
        z = numpy.array([[complex(x, y) for x, y in self.data]])
        return numpy.round(abs(z.T - z), NUMPY_PRECISION)

    def run(self):
        if not numpy.any(self.dist_matrix):
            self.dist_matrix = self.calc_dist_matrix()

        self.reset()

        start = datetime.now()
        self.best_solution = self.iterated_local_search(self.iteration_limit, self.idle_limit, start)
        self.runtime = datetime.now() - start

        path_order = self.best_solution['tour']
        path_len = self.best_solution['distance']
        path = [self.data[i] for i in path_order]

        return path, path_len

    def iterated_local_search(self, iteration_limit, idle_limit, start_timestamp):
        """Source: Algorithm3 from http://www.scielo.br/scielo.php?script=sci_arttext&pid=S2238-10312014000400010"""
        solution = {'tour': [], 'distance': 0, 'iteration': 0}
        # initial solution starting at 0
        # solution['tour'] = randomize_tour(len(self.data))
        solution['tour'] = self.model.greedy_init(self.dist_matrix, 100, self.num_city)
        solution['distance'] = self.calculate_tour_distance(solution['tour'])

        solution = self.local_search_wrapper(solution)
        solution['iteration'] = 1
        solution['runtime'] = datetime.now() - start_timestamp
        self.solutions.append(solution)
        self.iterations += 1

        for i in range(1, iteration_limit):
            new_solution = self.perturbation(solution)
            new_solution = self.local_search_wrapper(new_solution)
            if new_solution['distance'] < solution['distance']:
                solution = new_solution
                solution['iteration'] = i + 1
                solution['runtime'] = datetime.now() - start_timestamp
            self.solutions.append(new_solution)
            print(i, solution['distance'])
            self.iterations += 1
            self.iter_x.append(i)
            self.iter_y.append(solution['distance'])
        return solution

    def get_edge_list(self, tour):
        # create all edges as tuples beginning at 0 and ending at 0
        edges = [(tour[i], tour[i + 1]) for i in range(0, len(tour) - 1)]
        edges.append((tour[len(tour) - 1], tour[0]))
        return edges

    def calculate_tour_distance(self, tour):
        edges = self.get_edge_list(tour)
        distance = 0
        for a, b in edges:
            distance += self.dist_matrix[a, b]
        return distance

    def local_search_wrapper(self, solution):
        """this wrapper is used to change local search mode"""
        if self.is_SA_enabled:
            return self.simulated_annealing(solution)
        elif not self.alternative:
            return self.local_search(solution)
        else:
            return self.local_search_alt(solution, self.idle_limit)

    def simulated_annealing(self, solution):
        local_opt = solution
        self.model.reset(solution['tour'])
        path, path_len = self.model.run()
        local_opt['distance'] = path_len
        local_opt['tour'] = path
        return local_opt

    def local_search(self, solution):
        local_opt = solution
        for a, b in combinations(range(len(solution['tour'])), 2):
            if abs(a-b) in (1, len(solution['tour'])-1):
                continue
            tour = self.stochastic_two_opt(solution['tour'], a, b)
            distance = self.calculate_tour_distance(tour)
            if distance < local_opt['distance']:
                local_opt['tour'] = tour
                local_opt['distance'] = distance
        return local_opt

    def local_search_alt(self, solution, idle_limit):
        idle_counter = 0
        total_counter = 0

        while idle_counter < idle_limit:
            tour = self.stochastic_two_opt_random(solution['tour'])
            distance = self.calculate_tour_distance(tour)
            if distance < solution['distance']:
                idle_counter = 0
                solution['tour'] = tour
                solution['distance'] = distance
            else:
                idle_counter += 1
            total_counter += 1
        self.alternative_counter.append(total_counter)
        return solution

    def stochastic_two_opt(self, tour, c1, c2):
        """Delete 2 Edges and reverse everything between them
        Source: http://www.cleveralgorithms.com/nature-inspired/stochastic/iterated_local_search.html"""
        tour = tour[:]
        # make sure c1 < c2
        if c2 < c1:
            c1, c2 = c2, c1
        rev = tour[c1:c2]
        rev.reverse()
        tour[c1:c2] = rev
        return tour

    def stochastic_two_opt_random(self, tour):
        """2-opt by randomly selecting 2 points"""
        tour = tour[:]
        c1 = random.randint(0, len(tour))
        c2 = random.randint(0, len(tour))
        exclude = [c1]
        if c1 == 0:
            exclude.append(len(tour) - 1)
        else:
            exclude.append(c1 - 1)
        if c2 == len(tour) - 1:
            exclude.append(0)
        else:
            exclude.append(c1 + 1)

        while c2 in exclude:
            c2 = random.randint(0, len(tour))

        # make sure c1 < c2
        if c2 < c1:
            c1, c2 = c2, c1
        rev = tour[c1:c2]
        rev.reverse()
        tour[c1:c2] = rev
        return tour

    def perturbation(self, solution):
        new_solution = {}
        new_solution['tour'] = self.double_bridge_move(solution['tour'])
        new_solution['distance'] = self.calculate_tour_distance(new_solution['tour'])
        return new_solution

    def double_bridge_move(self, tour):
        """Split tour in 4 and reorder them.
        (a,b,c,d) --> (a,d,c,b)
        Source: https://www.comp.nus.edu.sg/~stevenha/database/viz/TSP_ILS.cpp"""
        tmp = len(tour)//4
        pos1 = 1 + random.randint(0, tmp)
        pos2 = pos1 + 1 + random.randint(0, tmp)
        pos3 = pos2 + 1 + random.randint(0, tmp)
        return tour[0:pos1] + tour[pos3:] + tour[pos2:pos3] + tour[pos1:pos2]
