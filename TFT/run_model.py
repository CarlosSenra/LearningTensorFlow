import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import warnings

import layers_model

d_model = 6
hidden_states = 32

layer_glu = layers_model.GLU(hidden_layer_size=hidden_states)
layer_addnorm = layers_model.GateAddNorm(hidden_states,dropout_rate=0.1)

layer_grn = layers_model.GRN(hidden_states,
                             dropout_rate=0.0,
                             optional_context=True)

layer_grn_1 = layers_model.GRN(hidden_states,
                             dropout_rate=0.0,
                             specific_output_size=d_model,
                             optional_context=True)
if __name__ == "__main__":
    y = tf.random.uniform((1, 8, d_model))
    context = tf.random.uniform((1, 8, hidden_states))
    print(y)
    print("----------")

    out = layer_glu(y)
    print(out)

    print("----------")

    out1 = layer_addnorm([y,context])
    print(out1)

    print("----------")

    out2 = layer_grn([y,context])
    print(out2)

    print("----------")

    out2 = layer_grn_1([y,context])
    print(out2)