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
        """"
            Esta parte inicializa los valores
        """
        for it in range(self.iterations):
            valuestemporal=util.Counter()
            for estado in mdp.getStates():
                maximoValue=-99999999

                actionValues = []
                for action in self.mdp.getPossibleActions(estado):
                    actionValue = self.getQValue(estado, action)
                    maximovalue=max(maximoValue,actionValue)
                if maximoValue!=-99999999 :
                    valuestemporal[estado] = maximoValue
           for estado in self.mdp.getStates():
                self.values[estado] = valuestemporal[estado]



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    #Calcula el valor-Q para un (estado-accion), tomando en cuenta las recompensas futuras esperadas.
    def computeQValueFromValues(self, estado, accion):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
       
        qValue = 0
        #Se itera sobre todos los posibles estados a los que podemos ir y la probabilidad de llegar, dada una accion en el estado actual
        for prob in self.mdp.getTransitionStatesAndProbs(estado, accion): 
            #Se obtiene la recompensa correspondiente a la accion tomada en un estado especifico, terminando en un nuevo estado.
            recompensa = self.mdp.getReward(estado, accion, prob[0])
            #Actualiza el valor-Q, sumando el producto de la probabilidad de transición con la suma de la recompensa inmediata y el valor descontado del estado siguiente.
            qValue += prob[1] * (recompensa + self.discount * self.values[prob[0]])
        
        return qValue

       

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal():
            return None
        
        accionesPosibles = self.mdp.getPossibleActions(state)
        
        for accion in accionesPosibles:
            accion = accion.computeQValueFromValues(state, accion)
        
        return max(accionesPosibles)
   

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


