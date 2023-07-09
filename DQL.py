# Modified version of
# DQN implementation by Tejas Kulkarni found at
# https://github.com/mrkulk/deepQN_tensorflow

import numpy as np
import tensorflow as tf

class DQN:
    def __init__(self, params): ## Inicializacion red neuronal
        self.params = params 
        self.network_name = 'qnet' ## Nombre de la red
        self.sess = tf.Session() ## Se crea una sesion de tensorflow
        self.x = tf.placeholder('float', [None, params['width'],params['height'], 6],name=self.network_name + '_x') ## placeholder para la entrada de la red, el estado x del juego
        self.q_t = tf.placeholder('float', [None], name=self.network_name + '_q_t') ## placeholder para el valor de Q(s,a)
        self.actions = tf.placeholder("float", [None, 4], name=self.network_name + '_actions')## placeholder para las acciones
        self.rewards = tf.placeholder("float", [None], name=self.network_name + '_rewards')## placeholder para las recompensas
        self.terminals = tf.placeholder("float", [None], name=self.network_name + '_terminals')## placeholder para los terminales

        # Layer 1 (Convolutional) Capa se que encarga de extraer las caracteristicas de la imagen, se define el tamaño de la capa y se inicializan los pesos y bias
        layer_name = 'conv1' ; size = 3 ; channels = 6 ; filters = 16 ; stride = 1
        self.w1 = tf.Variable(tf.random_normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b1 = tf.Variable(tf.constant(0.1, shape=[filters]),name=self.network_name + '_'+layer_name+'_biases')
        self.c1 = tf.nn.conv2d(self.x, self.w1, strides=[1, stride, stride, 1], padding='SAME',name=self.network_name + '_'+layer_name+'_convs') ## Se realiza la convolucion
        self.o1 = tf.nn.relu(tf.add(self.c1,self.b1),name=self.network_name + '_'+layer_name+'_activations') ## Se aplica la funcion de activacion

        # Layer 2 (Convolutional) Capa que se encarga de extraer las caracteristicas de la imagen
        layer_name = 'conv2' ; size = 3 ; channels = 16 ; filters = 32 ; stride = 1 ## Se define el tamaño de la capa
        self.w2 = tf.Variable(tf.random_normal([size,size,channels,filters], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.c2 = tf.nn.conv2d(self.o1, self.w2, strides=[1, stride, stride, 1], padding='SAME',name=self.network_name + '_'+layer_name+'_convs') ## Se realiza la convolucion
        self.o2 = tf.nn.relu(tf.add(self.c2,self.b2),name=self.network_name + '_'+layer_name+'_activations') ## Se aplica la funcion de activacion
        
        o2_shape = self.o2.get_shape().as_list()         ## Se obtiene la forma de la salida de la capa 2

        # Layer 3 (Fully connected) Capa que se encarga de la toma de decisiones
        layer_name = 'fc3' ; hiddens = 256 ; dim = o2_shape[1]*o2_shape[2]*o2_shape[3] ## Se define el tamaño de la capa que es igual a la salida de la capa 2
        self.o2_flat = tf.reshape(self.o2, [-1,dim],name=self.network_name + '_'+layer_name+'_input_flat') ## Se aplana la salida de la capa 2, esto es necesario para poder realizar la multiplicacion de matrices
        self.w3 = tf.Variable(tf.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights') 
        self.b3 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
        self.ip3 = tf.add(tf.matmul(self.o2_flat,self.w3),self.b3,name=self.network_name + '_'+layer_name+'_ips') ## Se realiza la multiplicacion de matrices
        self.o3 = tf.nn.relu(self.ip3,name=self.network_name + '_'+layer_name+'_activations') ## Se aplica la funcion de activacion, y su entrada es la multiplicacion de matrices que se realizo en la linea anterior

        # Layer 4 Capa que se encarga de la toma de decisiones
        layer_name = 'fc4' ; hiddens = 4 ; dim = 256 ## Se define el tamaño de la capa que es igual a la salida de la capa 3
        self.w4 = tf.Variable(tf.random_normal([dim,hiddens], stddev=0.01),name=self.network_name + '_'+layer_name+'_weights')
        self.b4 = tf.Variable(tf.constant(0.1, shape=[hiddens]),name=self.network_name + '_'+layer_name+'_biases')
        self.y = tf.add(tf.matmul(self.o3,self.w4),self.b4,name=self.network_name + '_'+layer_name+'_outputs') 

        #Q,Cost,Optimizer Se define la funcion de costo la cual es el error cuadratico medio, y el optimizador
        self.discount = tf.constant(self.params['discount']) ## Se define el factor de descuento gamma
        self.yj = tf.add(self.rewards, tf.multiply(1.0-self.terminals, tf.multiply(self.discount, self.q_t))) ## Se calcula el valor de Q(s',a')
        self.Q_pred = tf.reduce_sum(tf.multiply(self.y,self.actions), reduction_indices=1) ## Se calcula el valor de Q(s,a), pero solo para las acciones que se tomaron, es decir el estado actual y la accion que se tomo
        self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.yj, self.Q_pred), 2)) ## Se calcula el error cuadratico medio para el valor de Q(s,a) y Q(s',a'), esto es lo que se busca minimizar en el entrenamiento
        
        if self.params['load_file'] is not None: ## Si existe un entrenamiento previo se carga el numero de iteraciones que se han realizado
            self.global_step = tf.Variable(int(self.params['load_file'].split('_')[-1]),name='global_step', trainable=False) 
        else:
            self.global_step = tf.Variable(0, name='global_step', trainable=False) ## sino se inicializan las iteraciones en 0
        
        # self.optim = tf.train.RMSPropOptimizer(self.params['lr'],self.params['rms_decay'],0.0,self.params['rms_eps']).minimize(self.cost,global_step=self.global_step) Si se descomenta esta linea se utiliza el optimizador RMSProp
        self.optim = tf.train.AdamOptimizer(self.params['lr']).minimize(self.cost, global_step=self.global_step) ## Se utiliza el optimizador Adam
        self.saver = tf.train.Saver(max_to_keep=0)   ## Se crea un saver para guardar el modelo, max_to_keep=0 indica que se guardara el ultimo modelo

        self.sess.run(tf.global_variables_initializer()) ## Se inicializan las variables de tensorflow las cuales son los pesos y bias de la red

        if self.params['load_file'] is not None: ##Si existe entrenamiento previo se cargan los pesos y bias de la red
            print('Loading checkpoint...') 
            self.saver.restore(self.sess,self.params['load_file']) ## Se cargan los pesos y bias de la red

         
    def train(self,bat_s,bat_a,bat_t,bat_n,bat_r): ## esta funcion lo que hace es entrenar la red, toma lotes de estados, acciones, terminales, estados siguientes y recompensas
        feed_dict={self.x: bat_n, self.q_t: np.zeros(bat_n.shape[0]), self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r}  ## Se calcula el valor de Q(s',a') para todos los estados siguientes
        q_t = self.sess.run(self.y,feed_dict=feed_dict) ## regresa un arreglo de 4 columnas, cada columna es el valor de Q(s',a') para cada accion
        q_t = np.amax(q_t, axis=1) ##regresa el valor maximo de cada fila, es decir el valor de Q(s',a') para la accion que maximiza Q(s',a')
        feed_dict={self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r} ## Se actualiza el valor de Q(s,a) con el valor de Q(s',a')
        _,cnt,cost = self.sess.run([self.optim, self.global_step,self.cost],feed_dict=feed_dict) ## Se realiza el entrenamiento 
        return cnt, cost ## Se regresa el numero de iteraciones y el costo del valor de Q(s,a) y Q(s',a')

    def save_ckpt(self,filename): ## Esta funcion guarda el modelo
        self.saver.save(self.sess, filename)
