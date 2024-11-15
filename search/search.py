# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in search_agents.py).
"""
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in obj-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem.
        """
        util.raise_not_defined()

    def is_goal_state(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raise_not_defined()

    def get_successors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raise_not_defined()

    def get_cost_of_actions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raise_not_defined()


def tiny_maze_search(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# def addSuccessors(problem, addCost=True):

class SearchNode:
    def __init__(self, parent, node_info):
        """
            parent: parent SearchNode.

            node_info: tuple with three elements => (coord, action, cost)

            coord: (x,y) coordinates of the node position

            action: Direction of movement required to reach node from
            parent node. Possible values are defined by class Directions from
            game.py

            cost: cost of reaching this node from the starting node.
        """

        self.__state = node_info[0]
        self.action = node_info[1]
        self.cost = node_info[2] if parent is None else node_info[2] + parent.cost
        self.parent = parent

    # The coordinates of a node cannot be modified, se we just define a getter.
    # This allows the class to be hashable.
    @property
    def state(self):
        return self.__state

    def get_path(self):
        path = []
        current_node = self
        while current_node.parent is not None:
            path.append(current_node.action)
            current_node = current_node.parent
        path.reverse()
        return path
    
    # Consider 2 nodes to be equal if their coordinates are equal (regardless of everything else)
    # def __eq__(self, __o: obj) -> bool:
    #     if (type(__o) is SearchNode):
    #         return self.__state == __o.__state
    #     return False

    # # def __hash__(self) -> int:
    # #     return hash(self.__state)

def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.get_start_state())
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    "*** YOUR CODE HERE ***"
    # initialize a stack to manage nodes to explore
    frontier = util.Stack()
    # retrieve the starting state from the problem
    start_state = problem.get_start_state()
    # create the initial node with no parent and starting state
    initial_node = SearchNode(None, (start_state, None, 0))
    # push the initial node onto the stack
    frontier.push(initial_node)
    # list to keep track of visited states
    expanded = []

    # loop until the stack is empty
    while not frontier.is_empty():
        # pop a node from the stack
        current_node = frontier.pop()
        current_state = current_node.state
        # check if the current state is the goal
        if problem.is_goal_state(current_state):
            # return the path to the goal
            return current_node.get_path()
        # check if the state has been visited
        if current_state not in expanded:
            # add the state to the closed list
            expanded.append(current_state)
            # get successors of the current state
            successors = problem.get_successors(current_state)
            # iterate over each successor
            for successor in successors:
                next_state, action, cost = successor
                # check if the successor state has been visited
                if next_state not in expanded:
                    # create a new search node
                    child_node = SearchNode(current_node, (next_state, action, cost))
                    # push the child node onto the stack
                    frontier.push(child_node)

    # return an empty list if no solution is found
    return []


def breadth_first_search(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # create a queue to manage nodes to explore
    frontier = util.Queue()
    # get the starting state from the problem
    starting_state = problem.get_start_state()
    # create the initial search node without a parent
    initial_node = SearchNode(None, (starting_state, None, 0))
    # enqueue the initial node
    frontier.push(initial_node)
    # list to keep track of visited states
    expanded = []

    # loop until there are no nodes left to explore
    while not frontier.is_empty():
        # dequeue a node from the front of the queue
        current_node = frontier.pop()
        current_state = current_node.state
        # check if the current state is the goal
        if problem.is_goal_state(current_state):
            # return the path from the start state to the goal
            return current_node.get_path()
        # if the state hasn't been visited yet
        if current_state not in expanded:
            # add the state to the list of visited states
            expanded.append(current_state)
            # get the successors of the current state
            successors = problem.get_successors(current_state)

            # iterate over each successor
            for successor_state, action, step_cost in successors:
                # if the successor state hasn't been visited
                if successor_state not in expanded:
                    # create a new search node with the successor state
                    child_node = SearchNode(current_node, (successor_state, action, step_cost))
                    # enqueue the child node to the exploration queue
                    frontier.push(child_node)

    # return an empty list if no solution is found
    return []

def uniform_cost_search(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # create a priority queue to manage nodes to explore
    frontier_queue = util.PriorityQueue()
    # get the starting state from the problem
    starting_state = problem.get_start_state()
    # create the initial search node without a parent
    initial_node = SearchNode(None, (starting_state, None, 0))
    # enqueue the initial node with its cost as the priority
    frontier_queue.push(initial_node, initial_node.cost)
    # dictionary to keep track of visited states and their costs
    expanded = {}

    # loop until there are no nodes left to explore
    while not frontier_queue.is_empty():
        # dequeue the node with the lowest cost
        current_node = frontier_queue.pop()
        current_state = current_node.state
        # check if the current state is the goal
        if problem.is_goal_state(current_state):
            # return the path from the start state to the goal
            return current_node.get_path()
        # if the state hasn't been visited or found a cheaper path
        if current_state not in expanded or current_node.cost < expanded[current_state]:
            # update the cost for the current state
            expanded[current_state] = current_node.cost
            # get the successors of the current state
            successors = problem.get_successors(current_state)
            # iterate over each successor
            for successor_state, action, step_cost in successors:
                # calculate the cumulative cost to reach the successor
                cumulative_cost = current_node.cost + step_cost
                # create a new search node with the successor state
                child_node = SearchNode(current_node, (successor_state, action, step_cost))
                child_node.cost = cumulative_cost
                # enqueue the child node with its cumulative cost as priority
                frontier_queue.push(child_node, cumulative_cost)

    # return an empty list if no solution is found
    return []

def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def a_star_search(problem, heuristic=null_heuristic):
    """
    performs A* search to find the optimal path to the goal using the provided heuristic.
    """
    # create a priority queue to manage nodes to explore
    frontier_queue = util.PriorityQueue()
    # get the starting state from the problem
    starting_state = problem.get_start_state()
    # create the initial search node without a parent
    initial_node = SearchNode(None, (starting_state, None, 0))
    # calculate the heuristic value for the starting state
    heuristic_value = heuristic(starting_state, problem)

    # enqueue the initial node with its total estimated cost as priority
    frontier_queue.push(initial_node, initial_node.cost + heuristic_value)
    # dictionary to keep track of visited states and their costs
    # loop until there are no nodes left to explore
    expanded = {}
    
    while not frontier_queue.is_empty():
        # dequeue the node with the lowest estimated total cost
        current_node = frontier_queue.pop()
        current_state = current_node.state
        # check if the current state is the goal
        if problem.is_goal_state(current_state):
            # return the path from the start state to the goal
            return current_node.get_path()
        # if the state hasn't been visited or found a cheaper path
        if current_state not in expanded or current_node.cost < expanded[current_state]:
            # update the cost for the current state
            expanded[current_state] = current_node.cost
            # get the successors of the current state
            successors = problem.get_successors(current_state)
            # iterate over each successor
            for successor_state, action, step_cost in successors:
                # calculate the cumulative cost to reach the successor
                cumulative_cost = current_node.cost + step_cost
                # calculate the heuristic value for the successor state
                heuristic_value = heuristic(successor_state, problem)
                # calculate the total estimated cost f(n) = g(n) + h(n)
                total_estimated_cost = cumulative_cost + heuristic_value
                # create a new search node with the successor state
                child_node = SearchNode(current_node, (successor_state, action, step_cost))
                child_node.cost = cumulative_cost
                # enqueue the child node with its total estimated cost as priority
                frontier_queue.push(child_node, total_estimated_cost)
    # return an empty list if no solution is found
    return []

# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
