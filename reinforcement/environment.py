# environment.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
from abc import abstractmethod


#!/usr/bin/python

class Environment:

    @abstractmethod
    def get_current_state(self):
        """
        Returns the current state of enviornment
        """

    @abstractmethod
    def get_possible_actions(self, state):
        """
          Returns possible actions the agent
          can take in the given state. Can
          return the empty list if we are in
          a terminal state.
        """

    @abstractmethod
    def do_action(self, action):
        """
          Performs the given action in the current
          environment state and updates the enviornment.

          Returns a (reward, next_state) pair
        """

    @abstractmethod
    def reset(self):
        """
          Resets the current state to the start state
        """

    def is_terminal(self):
        """
          Has the enviornment entered a terminal
          state? This means there are no successors
        """
        state = self.get_current_state()
        actions = self.get_possible_actions(state)
        return len(actions) == 0
