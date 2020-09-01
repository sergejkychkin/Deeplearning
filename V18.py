import numpy 
import random
random.seed(101)
import gym 
import os 
import tensorflow as tf 
import queue
import math
class Fields(gym.Env):
    def __init__(self,size=10):
        #self.field = numpy.zeros(shape=(1,2))
        #self.catch = dict()
        
        #{(random.randint(0,10),random.randint(0,10)):random.randint(0,10) for i in range(times)}#[(random.randint(0,10),random.randint(0,10),10) for i in range(times)]
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(1)#(low=0.0,high=10.0,shape=(10),dtype=numpy.float32)
        self.done = False
        self.size = size-1
        self.posx = None       
        self.x = None
        self.v = None
    
      
       
    

        
            
            
    def position(self,action):
        if(action == 0):
            return -1
        elif(action == 1):
            return +1
        
    def reset(self):
        self.x = random.randint(0,self.size)
        self.v = random.randint(1,10)
        self.posx = random.randint(0,self.size)
        self.done = False
        return numpy.array([self.distances()])
        
    
    def distances(self):
        return self.x-self.posx
        
    
    
    def step(self,action):
        p = self.position(action)
        c = 1
        if ((self.posx+p<=-1 or self.posx+p > self.size)):
            r = -1
            #c = 1 
            #print(self.posx+p[0],self.posy+p[1])
        else:
            c = self.distances()
            self.posx+=p
            r = 0
        x = self.distances()
        if(x ==0 ):
            self.done = True
            #reward = self.v
        
        
        reward = abs(c)-abs(x)
        if(x == 0):
            reward = self.v
        
        
        return numpy.array([x]),reward+r,self.done
    
    def render(self):
        a = numpy.zeros(shape=(self.size+1))
        a[self.x] = self.v
        a[self.posx] = -1 
        return a
    
    
    
        
        
class Collector(object):
    def __init__(self,load=None):
       
        
        self.gamma = 0.95 
        self.epsilon = 1.0 
        self.epsilon_min = 0.01 
        self.epsilon_decay = 0.95
        self.learning_rate = 0.01
        
        #self.x = x 
        
        #self.y = y
        
        #self.lives =  lives
        self.memory = queue.deque(maxlen=200) 
       
        self.model = self._build_network(load)
    
    
        
    def _build_network(self,load):
        if(load):
            return tf.keras.models.load_model(load)
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(3,input_dim=(1),activation='relu'))
        #model.add(tf.keras.layers.Dense(25,activation='relu'))
        model.add(tf.keras.layers.Dense(2,activation='relu'))
        model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(self.learning_rate))
        return model
        
    
    
    
    
    def add_memory(self,state,action,reward,next_state,done):
        self.memory.append(tuple((state,action,reward,next_state,done)))
    
    
    def act(self,state):
        if numpy.random.rand() <= self.epsilon:
            
            return random.randrange(0,2),"r"
        else:
            #print(state)
            return numpy.argmax(self.model.predict(state)),"n"
            
           
        
        
    def work(self,state):
        return numpy.argmax(self.model.predict(numpy.array([state])))
    
    
    def replay(self,batch_size):
        if((len(self.memory)-batch_size) < 0):
            batch_size = len(self.memory)
            
        minibatch = random.sample(self.memory,batch_size)
        for state,action,reward,nextstate,done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma*numpy.amax(self.model.predict(nextstate))
                
            target_f = self.model.predict(state)
            
            target_f[0][action] = target
            
            
            self.model.fit(state,target_f,epochs=1,verbose=1)
                        
                
                                          
        if( self.epsilon > self.epsilon_min):
                           
                           self.epsilon *= self.epsilon_decay
            
        
            
        

    
    
    
    
    
    
#controll if all ship are destroyed / extraclass 
#
    
    
    
    
    
        
        
        
            
        
        
        
            
    
            
        
            
