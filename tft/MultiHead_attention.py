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


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, use_bias=False):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.d_q = self.d_k = self.d_v = self.depth = d_model // num_heads

    self.wq = [tf.keras.layers.Dense(self.depth, use_bias=use_bias) for _ in range(num_heads)]
    self.wk = [tf.keras.layers.Dense(self.depth, use_bias=use_bias) for _ in range(num_heads)]
    self.wv = [tf.keras.layers.Dense(self.depth, use_bias=use_bias) for _ in range(num_heads)]

    self.w_h = tf.keras.layers.Dense(self.d_model,use_bias=use_bias)


  def call(self, v, k, q, mask):
    print("Query : \n")  
    print(q)
    print("\n Key : \n")  
    print(k)
    print("\n Value : \n")  
    print(v)

     

    heads, attns = [], []
    print("Get inside of looping, for heads calculations !!\n")
    for i in range(self.num_heads):
      q = self.wq[i](q)
      k = self.wk[i](k)  
      v = self.wv[i](v)

      print(f"loop {i}")
      scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

      heads.append(scaled_attention)
      attns.append(attention_weights)

    print("MY HEADS lists : \n")
    print(heads)
    print("MY ATTENTION WEIGHT lists : \n")
    print(attns)

    heads_concat = tf.concat(heads, axis=-1) if self.num_heads > 1 else heads[0]
    attns_concat = tf.concat(attns, axis=-1)

    print("CONCATENING HEADS lists:\n")
    print(heads_concat)
    print("\nCONCATENING ATTENTION lists:\n")
    print(attns_concat)

    outputs = self.w_h(heads_concat)
    print("returning the final results:")
    print(outputs)
    return outputs, attns_concat

if __name__ == "__main__":

  d_model = 6

  y = tf.random.uniform((1, 8, d_model))

  model = MultiHeadAttention(d_model=d_model,num_heads=2)

  a, b = model(y,y,y,mask = None)