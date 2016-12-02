import numpy as np
import tensorflow as tf

def mu_law(x, mu=255, x_max=1):
  return x_max*np.sign(x)*np.log(1+mu*np.abs(x)/x_max)/np.log(1+mu)

def inverse_mu_law(y, mu=255, x_max=1):
  return x_max*np.sign(y)*(np.exp(np.abs(y)*np.log(1+mu)/x_max)-1)/mu

def quantize(x, n_bins=256, x_max=1):
  bins = np.zeros(list(np.shape(x))+[n_bins])
  if np.any(np.abs(np.array(x) > x_max)):
    raise ValueError('|x| should not be greater than x_max (x_max=' + str(x_max) + ')')
  x_bin = np.round((np.array(x)/(2*x_max)+0.5)*(n_bins-1)).astype(int)
  flat_bins = np.reshape(bins, [-1, n_bins])
  flat_x_bin = np.reshape(x_bin, [-1])
  flat_bins[(np.arange(len(flat_x_bin)),flat_x_bin)] = 1
  return np.reshape(flat_bins, np.shape(bins))


def pick_max(X):
  n_bins=X.shape[-1]
  bins = np.zeros(list(np.shape(X)))
  flat_bins = np.reshape(bins, [-1, n_bins])
  flat_X = np.reshape(X, [-1,n_bins])
  flat_X_bin = np.argmax(flat_X, axis=1)
  flat_bins[(np.arange(len(flat_X_bin)),flat_X_bin)] = 1
  return np.reshape(flat_bins, np.shape(bins))

def unquantize(X, x_max=1):
  n_bins=X.shape[-1]
  indices = np.where(np.reshape(X, [-1,n_bins])==1)[1]
  try:
    indices = np.reshape(indices, X.shape[:-1])
  except ValueError:
    print(np.where(np.reshape(X, [-1,n_bins])==1))
    print(indices.shape)
    raise ValueError('Quantized array x should have shape (x.shape, n_bins), with one non-zero value per bin.')
  return (indices/(n_bins-1) - 0.5) * 2*x_max

def tf_inverse_mu_law(y, mu=255, x_max=1):
  return x_max*tf.sign(y)*(tf.exp(tf.abs(y)*np.log(1+mu)/x_max)-1)/mu

def tf_unquantize_pick_max(X, X_shape, x_max=1):
  n_bins=X_shape[-1]
  flat_X = tf.reshape(X, [-1,n_bins])
  flat_X_bin = tf.argmax(flat_X, axis=1)
  indices = tf.reshape(flat_X_bin, [-1]+X_shape[:-1])
  return (indices/(n_bins-1) - 0.5) * 2*x_max

if __name__ == "__main__":
  import matplotlib.pyplot as plt
  n_bins = 256
  x = np.linspace(0,4*np.pi,1000)
  y = np.sin(x)
  plt.figure('quantization_test')
  plt.step(x, y)
  plt.step(x, inverse_mu_law(unquantize(quantize(mu_law(y), n_bins))))
  plt.show()
