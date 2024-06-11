import tensorflow as tf
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

class GLU(tf.keras.layers.Layer):
    def __init__(self,units):
        """Define a Gate Linear Unit

        Args:
            units (int): the units quantities in the dense layer on GLU
        """
        super(GLU,self).__init__()

        self.linear = tf.keras.layers.Dense(units)                      #creating a layer to calculate W_{5}x+b_{5}
        self.dense_sigmoid = tf.keras.layers.Dense(units,activation='sigmoid') #creating a layer to calculate sigmoid(W_{4}x+b_{4})

    def call(self,inputs):
        """Calculate the element wise multiplication between the layers

        Args:
            inputs (tf.Tensor,tf.Dataset or np.array): the input to calculate the GLU

        Returns:
            tf.Tensor,tf.Dataset or np.array: element_wise_product
        """
        print("My inputs:\n")
        print(inputs)
        sig = self.dense_sigmoid(inputs) 
        print("sigmoid:\n")
        print(sig)
        linear = self.linear(inputs)
        print("linear:\n")
        print(linear)
        element_wise_product = sig * linear
        print("product:")
        print(element_wise_product)
        return element_wise_product
    
class Drop_GLU_Add_Norm(tf.keras.layers.Layer):
    def __init__(self, units, drop_rate):
        """Define the process of apply a dropout, a GLU add the residuals to GLU output and normalize them

        Args:
            units (int): the units quantities in the dense layer on GLU
            drop_rate (float): a float number 0 <= drop_rate <= 1, where define te dropout rate
        """
        super(Drop_GLU_Add_Norm,self).__init__()
        self.units= units
        self.drop_rate = drop_rate

        self.dropout_layer = tf.keras.layers.Dropout(self.drop_rate)
        self.layer_GLU = GLU(self.units)
        self.norm_layer = tf.keras.layers.LayerNormalization()

    def call(self,inputs, residual):
        """Compute all process 

        Args:
            inputs (tf.Tensor,tf.Dataset or np.array): real input
            residual (tf.Tensor,tf.Dataset or np.array): the residuals input to add to normalize
        Returns:
            tf.Tensor,tf.Dataset or np.array : normalized_values
        """
        print("My input\n")
        print(inputs)
        print("My residual:\n")
        print(residual)
        input_droped = self.dropout_layer(inputs)
        print("Droping:\n")
        print(input_droped)
        glu_output = self.layer_GLU(input_droped)
        print("GLU:\n")
        print(glu_output)
        normalized_values = self.norm_layer(glu_output + residual)
        return normalized_values
    

class GRN(tf.keras.layers.Layer):
    def __init__(self, units, drop_rate, optional_context=False):
        super(GRN,self).__init__()

        self.units = units
        self.drop_rate = drop_rate
        self.optional_context = optional_context

        self.layer_ELU = tf.keras.layers.ELU()
        self.first_linear = tf.keras.layers.Dense(self.units)
        self.second_linear = tf.keras.layers.Dense(self.units)

        if self.optional_context:
            self.linear_optioal = tf.keras.layers.Dense(self.units,use_bias=False)

        
        self.add_norm = Drop_GLU_Add_Norm(units=self.units,
                                          drop_rate=self.drop_rate)
        
    def call(self,inputs):
        if self.optional_context:
            X, context = inputs
            dense_out = self.first_linear(X)
            context_out = self.linear_optioal(context)
            first_output = self.layer_ELU(dense_out + context_out)
        else:
            X = inputs
            dense_out = self.first_linear(X)
            first_output = self.layer_ELU(dense_out)

        second_output = self.second_linear(first_output)

        final_output = self.add_norm(second_output,X)

        return final_output


if __name__ == "__main__":
    d_model = 6
    y = tf.random.uniform((1, 8, d_model))

    #model = GLU(units=d_model)

    #model(y)

    #model = Drop_GLU_Add_Norm(units=8,drop_rate=0.1)

    #a = model(y,y)

    model = GRN(units=d_model,drop_rate=0.0)
    model(y)
    

  