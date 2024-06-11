import tensorflow as tf
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')



def scaled_dot_product_attention(q, k, v, mask):
  print("Applying SELF ATTENTION:")
  print("-------------------------------------------------")
  
  matmul_qk = tf.matmul(q, k, transpose_b=True) 
  print("\nmultiplying QUERIE and KEY:")
  print(matmul_qk)

  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  print("\nTaking the dk size:")
  print(dk)

  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
  print("\nStandarding the process:")
  print(scaled_attention_logits)

  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  
  print("\nApplying SOFTMAX in the Standarded vlaues, CALCULATING attention weights:")
  print(attention_weights)
  output = tf.matmul(attention_weights, v)
  print("\nCalculating the output, multipling the attention weights and the VALUE MATRIX:")
  print(output)
  print("-------------------------------------------------")
  return output, attention_weights


class InterpretableMultiHead(tf.keras.layers.Layer):
  def __init__(self, num_heads, d_model, use_bias=False, **kwargs):
    super(InterpretableMultiHead, self).__init__(**kwargs)
    self.num_heads = num_heads
    self.d_model = d_model
    self.d_k = self.d_q = self.d_v = d_model // num_heads

    self.v_layer = tf.keras.layers.Dense(self.d_v, use_bias=use_bias)
    self.q_layers = [tf.keras.layers.Dense(self.d_q, use_bias=use_bias) for _ in range(num_heads)]
    self.k_layers = [tf.keras.layers.Dense(self.d_k, use_bias=use_bias) for _ in range(num_heads)]
    self.w_h = tf.keras.layers.Dense(self.d_model, use_bias=use_bias)


  def call(self, queries, keys, values, mask=None):
    print("Query : \n")  
    print(queries)
    print("\n Key : \n")  
    print(keys)
    print("\n Value : \n")  
    print(values)

    heads, attns = [], []
    v = self.v_layer(values)
    print("\nGet inside of looping, for heads calculations !!\n")
    for i in range(self.num_heads):
      q = self.q_layers[i](queries)
      k = self.k_layers[i](keys)

      print(f"loop {i}")
      head, attn = scaled_dot_product_attention(q, k, v, mask)
      heads.append(head)
      attns.append(attn)

    print("\nMY HEADS lists : \n")
    print(heads)
    print("\nMY ATTENTION WEIGHT lists : \n")
    print(attns)
    
    #heads_concat = tf.concat(heads, axis=-1) if self.num_heads > 1 else heads[0]
    ##attns_concat = tf.concat(attns, axis=-1)

    ##print(heads_concat)
    ##print("\nCONCATENING ATTENTION lists:\n")
    ##print(attns_concat)

    outputs = tf.reduce_mean(heads, axis=0) if self.num_heads > 1 else head
    print("\nCALCULATING THE MEAN HEADS\n")
    print(outputs)
    outputs = self.w_h(outputs)
    print("\nCALCULATING the final results\n")
    print(outputs)

    return outputs, attn


if __name__ == "__main__":

  d_model = 6

  y = tf.random.uniform((1, 8, d_model))

  model = InterpretableMultiHead(d_model=d_model,num_heads=2)

  a, b = model(y,y,y,mask = None)