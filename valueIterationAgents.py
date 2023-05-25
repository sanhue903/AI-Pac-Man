# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational pu rposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        ## HECHO POR ELLA
        for iteration in range(self.iterations):
            temp = util.Counter()
            for state in self.mdp.getStates():
                #the value for terminal state is 0
                if self.mdp.isTerminal(state):
                    temp[state] = 0
                else:
                    #get actions and rewards
                    maximumValue = -99999
                    #actions for the state
                    actions = self.mdp.getPossibleActions(state)
                    for action in actions:
                        #find state and probability for the transition associated
                        #with actions
                        t = self.mdp.getTransitionStatesAndProbs(state, action)
                        value = 0
                        for stateAndProb in t:
                            value += stateAndProb[1] * (self.mdp.getReward(state, action, stateAndProb[1]) \
                            + self.discount * self.values[stateAndProb[0]])
                        maximumValue = max(value, maximumValue)
                    if maximumValue != -99999:
                        temp[state] = maximumValue
            self.values = temp


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        ## HECHO POR ELLA
        qValue = 0
        for stateAndProb in self.mdp.getTransitionStatesAndProbs(state, action):
            qValue += stateAndProb[1] * (self.mdp.getReward(state, action, stateAndProb[1]) \
            + self.discount * self.values[stateAndProb[0]])
        return qValue

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        ## HECHO POR ELLA

        #check for terminal
        if self.mdp.isTerminal(state):
            return None

        actions = self.mdp.getPossibleActions(state)
        #find all actions and the corresponding value and then return action
        #corresponding to the maximum value
        allActions = {}
        for action in actions:
            allActions[action] = self.computeQValueFromValues(state, action)

        return max(allActions, key=allActions.get)

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


