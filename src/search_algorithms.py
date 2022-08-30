from tensorflow.python.ops.init_ops import Zeros
import math
import numpy as np

import sys
sys.path.append('../aima-python')
from search import Problem

from search import Node
from utils import PriorityQueue
from numpy import array_equal

# hyp: state is an tuple with 2 elements: (row, column)

class TreasureMazeProblem(Problem):
  def __init__(self, initial, grid, k, goal=None):
    """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments.
    """
    super().__init__(initial)
    self.grid = grid
    self.k = k

    self.actions_space = {
        'UP': (-1,0),
        'DOWN': (1,0),
        'LEFT': (0,-1),
        'RIGHT': (0,1)
    }

    self.states_cost = {
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        'X': 5,
        'T': 1,
        'S': 1
    }

  def actions(self, state):
    """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once.
    """
    actions_list = []
    if state[0] > 0:
      actions_list.append('UP')
    if state[0] < len(self.grid)-1:
      actions_list.append('DOWN')
    if state[1] > 0:
      actions_list.append('LEFT')
    if state[1] < len(self.grid[0])-1:
      actions_list.append('RIGHT')
    return actions_list

  def result(self, state, action):
    """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).
    """
    new_state = (state[0] + self.actions_space[action][0], state[1] + self.actions_space[action][1])
    
    

    return new_state

  def goal_test(self, state):
    """ Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough.
    """
    if self.grid[state[0]][state[1]] == 'T':
      self.k = self.k - 1
      print("k:", self.k)

    return self.grid[state[0]][state[1]] == 'T', self.k == 0

  def path_cost(self, c, state1, action, state2):
    """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path.
    """
    value_state2 = self.grid[state2[0]][state2[1]]
    return c + self.states_cost[value_state2]

  def update_grid(self, actions):
    current_state = self.initial
    self.grid[current_state[0]][current_state[1]] = '1'
    for action in actions:
      current_state = self.result(current_state, action)
      if self.grid[current_state[0]][current_state[1]] in ['X', 'T']:
        self.grid[current_state[0]][current_state[1]] = '1'
    self.initial = current_state

def calculate_heuristic_grid(treasure_maze_problem):

  treasure_positions = []
  heuristic_grid = []

  # trovano le coordinate dei tesori
  for i in range(len(treasure_maze_problem.grid)):
    for j in range(len(treasure_maze_problem.grid[i])):
      if treasure_maze_problem.grid[i][j] == 'T':
        treasure_positions.append((i,j))
  
  if len(treasure_positions) == 0:
    # se non ci sono tesori restituisce una griglia con soli 0
    print('THERE ARE NO TREASURES')
    heuristic_grid = [ [0]*len(treasure_maze_problem.grid) for _ in range(len(treasure_maze_problem.grid[0])) ]
  else:
    for i in range(len(treasure_maze_problem.grid)):
      row = []
      for j in range(len(treasure_maze_problem.grid[i])):
        # per ogni cella della griglia calcola tutte i valori h_i-esimi e sceglie quello minore
        h_values = []
        for treasure in treasure_positions:
          h_values.append(abs(treasure[0] - i) + abs(treasure[1] - j))
        row.append(np.min(np.array(h_values)))
      heuristic_grid.append(row)

  return heuristic_grid  

def is_in(node, queue):
  if len(queue.heap) == 0:
    return False
  return any(node.state[0] == x.state[0] and node.state[1] == x.state[1] for _, x in queue.heap)


def dijkstra(problem, g):
  root = Node(problem.initial)
  explored = []
  frontier = PriorityQueue('min', g)
  frontier.append(root)

  while True:
    if len(frontier.heap) == 0:
      print("LA FRONTIER È VUOTA")
      return False, None
    current_node = frontier.pop()
    flag1, flag2 = problem.goal_test(current_node.state)
    if flag1 == True:
      return flag2, current_node.solution()
    
    explored.append(current_node.state)
    for action in problem.actions(current_node.state):
      child = current_node.child_node(problem, action)

      if not any(child.state[0] == i and child.state[1] == j for (i,j) in explored) and not is_in(child, frontier):
        frontier.append(child)
      elif is_in(child, frontier):
        if g(child) < frontier[child]:
          del frontier[child]
          frontier.append(child)

def a_star(problem, h):
  return dijkstra(problem, lambda node: node.path_cost + h[node.state[0]][node.state[1]])

def solve_treasure_maze_dijkstra(treasure_maze_problem):
  flag = False
  f = lambda node: node.path_cost
  tot_solution = []
  while not flag:
    flag, solution = dijkstra(treasure_maze_problem, f)
    print("DOPO DIJKSTRA:", flag, "---", solution)
    if solution is None:
      print('ERROR: can\'t find {} treasures'.format(treasure_maze_problem.k))
      break
    treasure_maze_problem.update_grid(solution)
    for row in treasure_maze_problem.grid:
      print("| {} {} {} {} |\n".format(row[0], row[1], row[2], row[3]))
    tot_solution.append(solution)
  
  return tot_solution

def solve_treasure_maze_a_star(treasure_maze_problem, h):
  flag = False
  f = lambda node: node.path_cost
  tot_solution = []
  while not flag:
    flag, solution = a_star(treasure_maze_problem, h)
    if solution is None:
      print('ERROR: can\'t find {} treasures'.format(treasure_maze_problem.k))
      break
    treasure_maze_problem.update_grid(solution)
    for row in treasure_maze_problem.grid:
      print("| {} {} {} {} |\n".format(row[0], row[1], row[2], row[3]))
    tot_solution.append(solution)
  
  return tot_solution