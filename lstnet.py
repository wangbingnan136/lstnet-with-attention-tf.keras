import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, GRU, Dropout, Flatten, Activation,Dropout
from tensorflow.keras.layers import concatenate, add, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers,constraints,initializers




class Feed_Forward_Attention(tf.keras.layers.Layer):
    """
    Keras Layer that implements an Attention mechanism for temporal data.
    Supports Masking.
    Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    :param kwargs:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(Feed_Forward_Attention())
    """
    def __init__(self, step_dim,attention_dropout_rate,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        super(Feed_Forward_Attention, self).__init__()
        
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        self.dropout=Dropout(attention_dropout_rate)
        
        

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True
        

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a=self.dropout(a)
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
    
    
    

    
class LSTNet(object):
    def __init__(self, window, dims,hidRNN,hidCNN,hidSkip,CNN_kernel, \
                 skip,highway_window,dropout,output_fun,attention=True,attention_dropout_rate=0.5,task='one_step_forecasting',multi_step=None):
        super(LSTNet, self).__init__()
        self.P = window
        self.m = dims
        self.hidR = hidRNN
        self.hidC = hidCNN
        self.hidS = hidSkip
        self.Ck = CNN_kernel
        self.skip = skip
        self.attention=attention
        self.attention_dropout_rate=attention_dropout_rate
         
        if not self.attention:
            self.pt = int((self.P-self.Ck)/self.skip)
            
            
        self.hw = highway_window
        self.dropout = dropout
        self.output = output_fun
        self.task=task
        
        if self.task in ['one_step_forecasting','regression']:
            self.task_layer=Dense(1,activation='linear')
        else:
            assert multi_step==None,'if task is not regression of one step forecasting,you must set the multi_step>1'
            self.task_layer=Dense(multi_step,activation='linear')
            

    def make_model(self):
        
        x = Input(shape=(self.P, self.m))

        # CNN
        c = Conv1D(self.hidC, self.Ck, activation='relu',padding='same')(x)
        c = Dropout(self.dropout)(c)
        

        # skip-RNN
        if self.attention: ## 优先使用attention
            r = GRU(self.hidR,return_sequences=True)(c)
            #r = Lambda(lambda k: K.reshape(k, (-1, self.hidR)))(r)
            r = Feed_Forward_Attention(step_dim=self.P,attention_dropout_rate=self.attention_dropout_rate)(r)
            #r = Dropout(self.dropout)(r)
                
            
        elif self.skip > 0:
            # RNN
            r = GRU(self.hidR)(c)
            r = Lambda(lambda k: K.reshape(k, (-1, self.hidR)))(r)
            r = Dropout(self.dropout)(r)

            # c: batch_size*steps*filters, steps=P-Ck
            s = Lambda(lambda k: k[:, int(-self.pt*self.skip):, :])(c)
            s = Lambda(lambda k: K.reshape(k, (-1, self.pt, self.skip, self.hidC)))(s)
            s = Lambda(lambda k: K.permute_dimensions(k, (0,2,1,3)))(s)
            s = Lambda(lambda k: K.reshape(k, (-1, self.pt, self.hidC)))(s)

            s = GRU(self.hidS)(s)
            s = Lambda(lambda k: K.reshape(k, (-1, self.skip*self.hidS)))(s)
            s = Dropout(self.dropout)(s)
            r = concatenate([r,s])
        
        res = Dense(self.m)(r)

        # highway
        if self.hw > 0:
            z = Lambda(lambda k: k[:, -self.hw:, :])(x)
            z = Lambda(lambda k: K.permute_dimensions(k, (0,2,1)))(z)
            z = Lambda(lambda k: K.reshape(k, (-1, self.hw)))(z)
            z = Dense(1)(z)
            z = Lambda(lambda k: K.reshape(k, (-1, self.m)))(z)
            res = add([res, z])
       
        if self.output != 'no':
            res = Activation(self.output)(res)
        output=self.task_layer(res)

        model = Model(inputs=x, outputs=output)
        #model.compile(optimizer=Adam(lr=self.lr, clipnorm=self.clip), loss=self.loss)
        return model
    
    
    
