# q_learning_agents.py
# ------------------
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


from game import *
from learning_agents import ReinforcementAgent
from feature_extractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - compute_value_from_q_values
        - compute_action_from_q_values
        - get_q_value
        - get_action
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.get_legal_actions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.q_values = util.Counter()

    def get_q_value(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        # verificamos si (state, action) existe en q_values
        if (state, action) in self.q_values:
            return self.q_values[(state, action)]
        else:
            return 0.0


    def compute_value_from_q_values(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.get_legal_actions(state)
        if not legal_actions:
            return 0.0
        
        # inicializamos el valor máximo con un valor muy pequeño
        max = -999999
        
        # iteramos sobre todas las acciones legales para encontrar el Q máximo
        for action in legal_actions:
            Q = self.get_q_value(state, action)
            if Q > max:
                max = Q
        
        return max

    def compute_action_from_q_values(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.get_legal_actions(state)
        if not legal_actions:
            return None
        
        max = self.get_value(state)
        best_actions = []
        
        # encontramos accions que tienen el valor Q máximo
        for action in legal_actions:
            if self.get_q_value(state, action) == max:
                best_actions.append(action)
        
        # seleccionamos una acción al azar entre las mejores acciones
        return random.choice(best_actions)


    def get_action(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flip_coin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        legal_actions = self.get_legal_actions(state)
        action = None
        
        if not legal_actions:
            return action
        
        # decidimos si explorar o explotar
        if util.flip_coin(self.epsilon):
            # random
            action = random.choice(legal_actions)
        else:
            # best policy
            action = self.get_policy(state)
                
        return action  
        

    def update(self, state, action, next_state, reward):
        """
          The parent class calls this to observe a
          state = action => next_state and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # calculamos el valor Q actual para (state, action)
        current_q = self.get_q_value(state, action)
        
        # calculamos el valor Q máximo para el siguiente estado
        max_next_q_value = self.get_value(next_state)
        
        # calculamos el valor de muestra (sample value)
        sample = reward + self.discount * max_next_q_value
        
        # Actualiza el valor Q del par (state, action) usando la fórmula de actualización
        self.q_values[(state, action)] = (1 - self.alpha) * current_q + self.alpha * sample

    def get_policy(self, state):
        return self.compute_action_from_q_values(state)

    def get_value(self, state):
        return self.compute_value_from_q_values(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, num_training=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        num_training - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['num_training'] = num_training
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def get_action(self, state):
        """
        Simply calls the get_action method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.get_action(self, state)
        self.do_action(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite get_q_value
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.feat_extractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def get_weights(self):
        return self.weights

    def get_q_value(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()
        

    def update(self, state, action, next_state, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodes_so_far == self.num_training:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
