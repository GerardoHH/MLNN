    
from keras import backend as K
from keras.engine.topology import Layer
from keras import activations


class DendralNeuron(Layer):
    
    def __init__(self, units, activation=None, **kwargs):
        self.Nd = units   #Number of dendrites
        self.activation = activations.get(activation)
        super(DendralNeuron, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Wmin = self.add_weight(name='Wmin', 
                                      shape=(self.Nd, input_shape[1]),
                                      initializer='uniform',
                                      trainable=True)
        
        self.Wmax = self.add_weight(name='Wmax', 
                                      shape=(self.Nd, input_shape[1]),
                                      initializer='uniform',
                                      trainable=True)
        
        super(DendralNeuron, self).build(input_shape) 

    def call(self, x):
        Q = K.int_shape(x)[0]
        if Q is None: Q = 1
        X = K.repeat(x,self.Nd)
        Wmin = K.permute_dimensions(K.repeat(self.Wmin, Q), (1,0,2))
        L1 = K.min(X - Wmin, axis=2)
        Wmax = K.permute_dimensions(K.repeat(self.Wmax, Q), (1,0,2))
        L2 = K.min(Wmax - X, axis=2)
        output = K.minimum(L1,L2)
        if self.activation is not None:
           output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.Nd) 

