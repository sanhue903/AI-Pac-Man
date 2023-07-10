# Used code from
# DQN implementation by Tejas Kulkarni found at
# https://github.com/mrkulk/deepQN_tensorflow

import numpy as np
import random
import util
import time
import sys

# Pacman game
from pacman import Directions
from game import Agent
import game

# Replay memory
from collections import deque

# Neural nets
import tensorflow as tf
from DQN import *

params = {                   ## Parametros de la red neuronal
    # Model backups
    'load_file': None,            ##Si load_file es diferente de None, se carga el modelo que se especifica en load_file
    'save_file': 'smallClassic',            ##Si save_file es diferente de None, se guarda el modelo que se especifica en save_file
    'save_interval' : 1000,      ## es el intervalo de iteraciones en el que se guardara el modelo

    # Training parameters
    'train_start': 1000,    # Episodes before training starts ## Numero de episodios antes de que empiece el entrenamiento
    'batch_size': 32,       # Replay memory batch size  
    'mem_size': 100000,     # Replay memory size 

    'discount': 0.95,       # Discount rate (gamma value) ## Factor de descuento gamma
    'lr': 0.0002,            # Learning reate              ## Tasa de aprendizaje
    'rms_decay': 0.99,      # RMS Prop decay Solo optimizador RMS, comentar esta linea si se usa adams.
    'rms_eps': 1e-6,        # RMS Prop epsilon Solo optimizador RMS, comentar esta linea si se usa adams.

    # Epsilon value (epsilon-greedy)
    'eps': 1.0,             # Epsilon start value  
    'eps_final': 0.1,       # Epsilon end value 
    'eps_step': 10000       # Epsilon steps between start and end (linear) ## Numero de iteraciones en las que se disminuira epsilon
}                     



class PacmanDQN(game.Agent):
    def __init__(self, args):

        print("Initialise DQN Agent")

        ## Se cargan los parametros de la red neuronal
        self.params = params 
        self.params['width'] = args['width']
        self.params['height'] = args['height'] 
        self.params['num_training'] = args['numTraining']

        # Se define cuanta memoria de la GPU se utilizara y se crea una sesion de tensorflow
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1) 
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) 

        self.qnet = DQN(self.params) ## Se crea la red neuronal

        # time started 
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        # Q and cost
        self.Q_global = [] ## Lista que guarda el valor de Q(s,a) para cada accion
        self.cost_disp = 0 ## Costo de la red neuronal. Costo se refiere al error cuadratico medio. 

        # Stats
        self.cnt = self.qnet.sess.run(self.qnet.global_step) ## Se obtiene el numero de iteraciones que se han realizado
        self.local_cnt = 0 ## Numero de iteraciones que se han realizado en el episodio actual 

        self.numeps = 0 ## Numero de episodios que se han realizado
        self.last_score = 0 ## Puntaje del episodio anterior    
        self.s = time.time() ## Tiempo en el que se empezo el entrenamiento
        self.last_reward = 0. ## Recompensa del episodio anterior

        self.replay_mem = deque() ## Se crea la memoria de repeticion
        self.last_scores = deque() ## Se crea una cola que guarda los puntajes de los ultimos 100 episodios


    def getMove(self, state):
        ## Se decide si se va a explorar o explotar
        if np.random.rand() > self.params['eps']: ## Si el numero aleatorio es mayor que epsilon se explota
            ## Explotacion
            self.Q_pred = self.qnet.sess.run( ## se elije la accion que maximiza Q(s,a)
                self.qnet.y,
                feed_dict = {self.qnet.x: np.reshape(self.current_state,
                                                     (1, self.params['width'], self.params['height'], 6)), 
                             self.qnet.q_t: np.zeros(1), 
                             self.qnet.actions: np.zeros((1, 4)), 
                             self.qnet.terminals: np.zeros(1), 
                             self.qnet.rewards: np.zeros(1)})[0] 

            self.Q_global.append(max(self.Q_pred)) 
            a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred)) 

            if len(a_winner) > 1:
                move = self.get_direction(
                    a_winner[np.random.randint(0, len(a_winner))][0])
            else:
                move = self.get_direction(
                    a_winner[0][0])
        else: ## En cambio, si el numero aleatorio es menor que epsilon se explora
            # Random:
            move = self.get_direction(np.random.randint(0, 4))

        # Save last_action
        self.last_action = self.get_value(move)

        return move

    def get_value(self, direction): ## Se obtiene el valor de la accion que se va a realizar
        if direction == Directions.NORTH:
            return 0.
        elif direction == Directions.EAST:
            return 1.
        elif direction == Directions.SOUTH:
            return 2.
        else:
            return 3.

    def get_direction(self, value): ## Se obtiene la accion que se va a realizar
        if value == 0.:
            return Directions.NORTH
        elif value == 1.:
            return Directions.EAST
        elif value == 2.:
            return Directions.SOUTH
        else:
            return Directions.WEST
            
    def observation_step(self, state): ## Se realiza una observacion del estado actual
        if self.last_action is not None:
            # Se procesa el estado actual
            self.last_state = np.copy(self.current_state)
            self.current_state = self.getStateMatrices(state)

            # Se procesa la recompensa actual
            self.current_score = state.getScore() ## Se obtiene el puntaje actual
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            if reward > 20:
                self.last_reward = 50.    # Se come un fantasma 
            elif reward > 0:
                self.last_reward = 10.    # Come comida
            elif reward < -10:
                self.last_reward = -500.  # Es comido -500
                self.won = False
            elif reward < 0:
                self.last_reward = -1.    # Castigo por no comer

            
            if(self.terminal and self.won):
                self.last_reward = 100.
            self.ep_rew += self.last_reward

            # Se guarda la ultima experiencia en la memoria de repeticion
            experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            # Se guarda el modelo si se requiere
            if(params['save_file']):
                if self.local_cnt > self.params['train_start'] and self.numeps == self.params['save_interval'] : ## el modelo se guarda cada save_interval episodios
                    self.qnet.save_ckpt('saves/model-' + params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))
                    print('Model saved')

            ## Se entrena la red neuronal
            self.train()

        # Siguiente iteracion
        self.local_cnt += 1
        self.frame += 1
        self.params['eps'] = max(self.params['eps_final'],
                                 1.00 - float(self.cnt)/ float(self.params['eps_step']))


    def observationFunction(self, state): ## Se realiza una observacion del estado actual
        
        self.terminal = False
        self.observation_step(state)

        return state

    def final(self, state):## Se realiza una observacion del estado actual pero en el ultimo estado del episodio
        # Next 
        self.ep_rew += self.last_reward 

        # Do observation
        self.terminal = True
        self.observation_step(state)

        # Print stats
        log_file = open('./logs/'+str(self.general_record_time)+'-l-'+str(self.params['width'])+'-m-'+str(self.params['height'])+'-x-'+str(self.params['num_training'])+'.log','a')
        log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " % 
                         (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        log_file.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " % 
                         (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        sys.stdout.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.flush()

    def train(self): ## Funcion que entrena la red neuronal
        # Train
        if (self.local_cnt > self.params['train_start']):
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s = [] # States (s)
            batch_r = [] # Rewards (r)
            batch_a = [] # Actions (a)
            batch_n = [] # Next states (s')
            batch_t = [] # Terminal state (t)

            for i in batch:
                batch_s.append(i[0])
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_n.append(i[3])
                batch_t.append(i[4])
            batch_s = np.array(batch_s)
            batch_r = np.array(batch_r)
            batch_a = self.get_onehot(np.array(batch_a))
            batch_n = np.array(batch_n)
            batch_t = np.array(batch_t)

            self.cnt, self.cost_disp = self.qnet.train(batch_s, batch_a, batch_t, batch_n, batch_r)


    def get_onehot(self, actions): 
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):                                           
            actions_onehot[i][int(actions[i])] = 1      
        return actions_onehot   

    def mergeStateMatrices(self, stateMatrices): 
        """ Merge state matrices to one state tensor """
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        total = np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total

    def getStateMatrices(self, state):
        """ Return wall, ghosts, food, capsules matrices """ 
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.layout.walls
            matrix = np.zeros((height, width), dtype=np.int8)
            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell
            return matrix

        def getPacmanMatrix(state): ## Se obtiene el estado de pacman
            """ Return matrix with pacman coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates: 
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getGhostMatrix(state): ## Se obtiene la matriz de los fantasmas 
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getScaredGhostMatrix(state): ##  Se obtiene el estado de los fantasmas asustados
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getFoodMatrix(state): ## Se obtiene el estado de la comida
            """ Return matrix with food coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.food
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell

            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            capsules = state.data.layout.capsules
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1-i[1], i[0]] = 1

            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height 
        width, height = self.params['width'], self.params['height']
        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)

        observation = np.swapaxes(observation, 0, 2)

        return observation

    def registerInitialState(self, state): # inspects the starting state

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        self.current_state = self.getStateMatrices(state)

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.numeps += 1

    def getAction(self, state): ## Se obtiene la accion que se va a realizar
        move = self.getMove(state)

         ## Se detiene el movimiento cuando no es legal
        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP

        return move
