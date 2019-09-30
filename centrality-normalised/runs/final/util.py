import time, sys, os, random
import numpy as np
import tensorflow as tf

def timestamp():
  return time.strftime( "%Y%m%d%H%M%S", time.gmtime() )
#end timestamp

def memory_usage():
  pid=os.getpid()
  s = next( line for line in open( '/proc/{}/status'.format( pid ) ).read().splitlines() if line.startswith( 'VmSize' ) ).split()
  return "{} {}".format( s[-2], s[-1] )
#end memory_usage

def percent_error(labels,predictions,dtype=tf.float32):
  # Calculates percent error, avoiding division-by-zero
  # Replace any zeroes with ones
  labels_div = tf.where( tf.equal( labels, 0 ), tf.ones_like( labels, dtype=dtype ), labels )
  # Get % error
  return tf.reduce_mean( tf.divide( tf.abs( tf.subtract( labels, predictions ) ), labels_div ) )
#end percent_error

def precision_at_10(labels, predicted):
  precs = np.zeros(predicted.shape[0])
  for i in range(0, predicted.shape[0]):
    intersect = np.intersect1d( predicted[i], labels[i], assume_unique=True)
    precs[i] = intersect.size / predicted[i].size
  #end
  return np.mean( precs )
#end

def flatten( L, levels = 1 ):
  for i in range(levels):
    L_new = []
    for l in L:
      L_new = [*L_new, *l]
    #end for
    L = L_new
  #end for
  return L_new
#end flatten

def sparse_to_dense( M_sparse, default = 0.0 ):
  M_i, M_v, M_shape = M_sparse
  n, m = M_shape
  M = np.ones( (n, m), dtype = np.float32 ) * default
  for indexes, value in zip( M_i, M_v ):
    i,j = indexes
    M[i,j] = value
  #end for
  return M
#end sparse_to_dense

def dense_to_sparse( M, check = lambda x: x != 0, val = lambda x: x ):
  n, m = M.shape
  M_i = []
  M_v = []
  M_shape = (n,m)
  for i in range( n ):
    for j in range( m ):
      if check( M[i,j] ):
        M_i.append( (i,j ) )
        M_v.append( val( M[i,j] ) )
      #end if
    #end for
  #end for
  return (M_i,M_v,M_shape)
#end dense_to_sparse

def reindex_matrix( n, m, M ):
  new_index = []
  new_value = []
  for i, v in zip( M[0], M[1] ):
    s, t = i
    new_index.append( (n + s, m + t) )
    new_value.append( v )
  #end for
  return zip( new_index, new_value )
#end reindex_matrix

def load_weights(sess,path,scope=None):
    if os.path.exists(path):
      # Create model saver
      if scope is None:
        saver = tf.train.Saver()
      else:
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
      #end if
      saver.restore(sess, "%s/model.ckpt" % path)
    else:
        raise Exception('Checkpoint path does not exist!')
    #end if
#end save_weights

def save_weights(sess,path,scope=None):
  # Create /tmp/ directory to save weights
  if not os.path.exists(path):
    os.makedirs(path)
  #end if
  # Create model saver
  if scope is None:
    saver = tf.train.Saver()
  else:
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
  #end if
  saver.save(sess, "%s/model.ckpt" % path)
#end save_weights
