# SA-TSP
## CSC06A Team 04
# Three algorithms for Travelling salesman problem

* [Purpose]
* [Installation] 
* [Usage]
* [References]

## Purpose
Find the maximum, minimum, and average performance by running 10 times

## Installation 
Install library by using pip or conda

Example (pip): 
```
$ pip install numpy
$ pip install matplotlib
```

## Usage 
### Tabu search - TA
Steps:
* Select a file 
```
data = read_tsp('inputfile')
```
Run the following command
```
python3 main.py
```
### Simulated annealing 
* Environment: Jupyter makes iPython
* Initial parameter for simulated annealing algorithm
```
T(k)=alfa*T(k-1)
Initial Temperature = 100.0
Stop Temperature = 1
Markov =1000
alfa = 0.98
```
### Iterated local search ILS 
Run main.py file: 
```
Select data file in main.py
python3 main.py
```

### References

[1]https://github.com/tommy3713/TSP-Simulated-Annealing-SA-Python/blob/main/TSP_SA.py
[2]https://www.codestudyblog.com/cnb2105a/0515200057.html
[3]https://qiita.com/fockl/items/ada469c138900caaf0a8
[4]https://qiita.com/pocokhc/items/7f2fb4ee8d83e08e842b
[5]https://github.com/frizzleqq/Python-TSP-Heuristic

### Members 
* LIN Yu-Ling
* SUKEGAWA Takuya
* PHAM Thi Huong Giang
