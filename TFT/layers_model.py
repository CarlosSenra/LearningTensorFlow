import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import warnings

sigmoid = tf.keras.activations.sigmoid
elu = tf.keras.activations.elu 


class GLU(layers.Layer):
    def __init__(self, 
                 hidden_layer_size, 
                 dropout_rate = None):
        super(GLU, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate

        if self.dropout_rate:
            self.dropout = layers.Dropout(self.dropout_rate)

        self.dense1 = layers.Dense(hidden_layer_size) 
        self.dense2 = layers.Dense(hidden_layer_size)

    def call(self, inputs):
        x = inputs
        if self.dropout_rate:
            x = self.dropout(inputs)

        dense1 = self.dense1(x)
        dense2 = self.dense2(x)
        sigmoid_dense = sigmoid(dense1)

        output = sigmoid_dense * dense2

        return output

class GateAddNorm(layers.Layer):
    def __init__(self, 
                 hidden_layer_size, 
                 dropout_rate = None):
        super(GateAddNorm,self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate

        self.layer_glu = GLU(hidden_layer_size=self.hidden_layer_size,
                             dropout_rate=self.dropout_rate)
        self.layer_norm = layers.LayerNormalization()

    def call(self,inputs):

        x, input_to_add = inputs
        
        glu_out = self.layer_glu(x)
        output = self.layer_norm(glu_out + input_to_add)

        return output


class GRN(layers.Layer):
    def __init__(self, 
                 hidden_layer_size, 
                 dropout_rate = None,
                 optional_context = None,
                 specific_output_size = None):
        
        super(GRN,self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.optional_context = optional_context
        self.specific_output_size = specific_output_size


        if self.specific_output_size:
            self.dense1 = layers.Dense(self.specific_output_size) 
            self.layer_add_norm = GateAddNorm(hidden_layer_size=self.specific_output_size,
                                              dropout_rate=self.dropout_rate)
        else:
            self.dense1 = layers.Dense(self.hidden_layer_size) 
            self.linear_transform = layers.Dense(self.hidden_layer_size)
            self.layer_add_norm = GateAddNorm(hidden_layer_size=self.hidden_layer_size,
                                              dropout_rate=self.dropout_rate)
            

        self.dense2 = layers.Dense(self.hidden_layer_size)

        if self.optional_context:
            self.dense3 = layers.Dense(self.hidden_layer_size, use_bias = False)

    def call(self,inputs):
        if self.optional_context:
            inputs, context = inputs
            n_2 = elu(self.dense2(inputs) + self.dense3(context))
        else:
            out_dense2 = self.dense2(inputs)
            n_2 = elu(out_dense2)



        if self.specific_output_size:
            n_1 = self.dense1(n_2)
            output = self.layer_add_norm([n_1,inputs])
        else:
            n_1 = self.dense1(n_2)
            out_linear = self.linear_transform(inputs)
            output = self.layer_add_norm([n_1,out_linear])

        return output












