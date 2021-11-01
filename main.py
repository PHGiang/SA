import random
import math
import numpy as np
import matplotlib.pyplot as plt


class SA(object):
    def __init__(self, num_city, data):
        self.T0 = 4000
        # self.Tend = 1e-3
        self.Tend = 1e-2
        self.rate = 0.9995
        self.num_city = num_city
        self.scores = []
        self.location = data
        self.fires = []
        self.dis_mat = self.compute_dis_mat(num_city, data)
        self.fire = self.greedy_init(self.dis_mat,100,num_city)
        init_pathlen = 1. / self.compute_pathlen(self.fire, self.dis_mat)
        # init_best = self.location[self.fire]
        self.iter_x = [0]
        self.iter_y = [1. / init_pathlen]

    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x

                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        pathlens = self.compute_paths(result)
        sortindex = np.argsort(pathlens)
        index = sortindex[0]
        return result[index]

    @staticmethod
    def random_init(self, num_city):
        tmp = [x for x in range(num_city)]
        random.shuffle(tmp)
        return tmp

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
                dis_mat[i][j] = tmp
        return dis_mat

    def compute_pathlen(self, path, dis_mat):
        a = path[0]
        b = path[-1]
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    def get_new_fire(self, fire):
        fire = fire.copy()
        t = [x for x in range(len(fire))]
        a, b = np.random.choice(t, 2)
        fire[a:b] = fire[a:b][::-1]
        return fire

    def eval_fire(self, raw, get, temp):
        len1 = self.compute_pathlen(raw, self.dis_mat)
        len2 = self.compute_pathlen(get, self.dis_mat)
        dc = len2 - len1
        p = max(1e-1, np.exp(-dc / temp))
        if len2 < len1:
            return get, len2
        elif np.random.rand() <= p:
            return get, len2
        else:
            return raw, len1

    def sa(self):
        count = 0
        best_path = self.fire
        best_length = self.compute_pathlen(self.fire, self.dis_mat)

        while self.T0 > self.Tend:
            count += 1

            tmp_new = self.get_new_fire(self.fire.copy())

            self.fire, file_len = self.eval_fire(best_path, tmp_new, self.T0)
            if file_len < best_length:
                best_length = file_len
                best_path = self.fire
            self.T0 *= self.rate
            self.iter_x.append(count)
            self.iter_y.append(best_length)
            print(count, best_length)
        return best_length, best_path

    def run(self):
        best_length, best_path = self.sa()
        return self.location[best_path], best_length


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


if __name__ == '__main__':
    print('reading input data')
    # data = read_tsp(r'./tsp10.txt')
    # data = read_tsp(r'./tsp100.txt')
    data = read_tsp(r'./tsp1000.txt')
    print('read input data')

    data = np.array(data)
    data = data[:, 1:]
    show_data = np.vstack([data, data[0]])
    # Best, Best_path = math.inf, None
    Best = []
    Best_path = []
    no_of_realizations = 5
    model = []

    for i in range(no_of_realizations):
        model_i = SA(num_city=data.shape[0], data=data.copy())
        model.append(model_i)
        path, path_len = model.run()
        print(path_len)
        # if path_len < Best:

        Best.append(path_len)
        Best_path.append(path)

        Best_path = np.vstack([Best_path, Best_path[0]])

    fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
    axs[0].scatter(Best_path[:, 0], Best_path[:,1])
    Best_path = np.vstack([Best_path, Best_path[0]])
    axs[0].plot(Best_path[:, 0], Best_path[:, 1])
    axs[0].set_title('Result')
    iterations = model.iter_x
    best_record = model.iter_y
    axs[1].plot(iterations, best_record)
    axs[1].set_title('Pic')
    plt.show()
