import tensorflow as tf
from mlp import Mlp

class GraphNN(object):
  def __init__(
    self,
    var,
    mat,
    msg,
    loop,
    MLP_depth = 3,
    MLP_weight_initializer = tf.contrib.layers.xavier_initializer,
    MLP_bias_initializer = tf.zeros_initializer,
    Cell_activation = tf.nn.relu,
    Msg_activation = tf.nn.relu,
    Msg_last_activation = None,
    float_dtype = tf.float32,
    name = 'GraphNN'
  ):
    """
    Receives four dictionaries: var, mat, msg and loop.

    ○ var is a dictionary from variable names to embedding sizes.
      That is: an entry var["V1"] = 10 means that the variable "V1" will have an embedding size of 10.
    
    ○ mat is a dictionary from matrix names to variable pairs.
      That is: an entry mat["M"] = ("V1","V2") means that the matrix "M" can be used to mask messages from "V1" to "V2".
    
    ○ msg is a dictionary from function names to variable pairs.
      That is: an entry msg["cast"] = ("V1","V2") means that one can apply "cast" to convert messages from "V1" to "V2".
    
    ○ loop is a dictionary from variable names to lists of dictionaries:
      {
        "mat": the matrix name which will be used,
        "transpose?": if true then the matrix M will be transposed,
        "fun": transfer function (python function built using tensorflow operations,
        "msg": message name,
        "var": variable name
      }
      If "mat" is None, it will be the identity matrix,
      If "transpose?" is None, it will default to false,
      if "fun" is None, no function will be applied,
      If "msg" is false, no message conversion function will be applied,
      If "var" is false, then [1] will be supplied as a surrogate.
      
      That is: an entry loop["V2"] = [ {"mat":None,"fun":f,"var":"V2"}, {"mat":"M","transpose?":true,"msg":"cast","var":"V1"} ] enforces the following update rule for every timestep:
        V2 ← tf.append( [ f(V2), Mᵀ × cast(V1) ] )
    
    You can also specify:
    
    ○ MLP_depth, which indicates how many layers before the output layer each message MLP will have, defaults to 3.
    ○ MLP_weight_initializer, which indicates which initializer is to be used on the MLP layers' kernels, defaults to tf.contrib.layers.xavier_initializer.
    ○ MLP_bias_initializer, which indicates which initializer is to be used on the MLP layers' biases, defaults to tf.zeros_initializer.
    ○ Cell_activation, which indicates which activation function should be used on the LSTM cell, defaults to tf.nn.relu.
    ○ Msg_activation, which indicates which activation function should be used on the hidden layers of the MLPs, defaults to tf.nn.relu.
    ○ Msg_last_activation, which indicates which acitvation function should be used on the output, defaults to None (linear activation).
    ○ float_dtype, which indicates which float type should be used (not tested with others than tf.float32), defaults to tf.float32.
    ○ name, which is the scope name that the GNN will use to declare its parameters and execution graph, defaults to 'GraphNN'.
    """
    self.var, self.mat, self.msg, self.loop, self.name = var, mat, msg, loop, name

    self.MLP_depth = MLP_depth
    self.MLP_weight_initializer = MLP_weight_initializer
    self.MLP_bias_initializer = MLP_bias_initializer
    self.Cell_activation = Cell_activation
    self.Msg_activation = Msg_activation
    self.Msg_last_activation  = Msg_last_activation 
    self.float_dtype = float_dtype

    # Check model for inconsistencies
    self.__check_model()

    # Build the network
    with tf.variable_scope(self.name):
      with tf.variable_scope('placeholders'):
        self._init_placeholders()
      #end placeholders variable scope
      with tf.variable_scope('parameters'):
        self._init_parameters()
      #end parameters variable scope
      with tf.variable_scope('utilities'):
        self._init_utilities()
      #end utilities variable scope
      with tf.variable_scope('run'):
        self._run()
      #end run variable scope
    #end GNN variable scope
  #end __init__

  def __check_model(self):
    # Procedure to check model for inconsistencies before building the execution graph
    for v in self.var:
      if v not in self.loop:
        raise Warning('Variable {v} is not updated anywhere! Consider removing it from the model'.format(v=v))
      #end if
    #end for

    for v in self.loop:
      if v not in self.var:
        raise Exception('Updating variable {v}, which has not been declared!'.format(v=v))
      #end if
    #end for

    for mat, (v1,v2) in self.mat.items():
      if v1 not in self.var:
        raise Exception('Matrix {mat} definition depends on undeclared variable {v}'.format(mat=mat, v=v1))
      #end if
      if v2 not in self.var and type(v2) is not int:
        raise Exception('Matrix {mat} definition depends on undeclared variable {v}'.format(mat=mat, v=v2))
      #end if
    #end for

    for msg, (v1,v2) in self.msg.items():
      if v1 not in self.var:
        raise Exception('Message {msg} maps from undeclared variable {v}'.format(msg=msg, v=v1))
      #end if
      if v2 not in self.var:
        raise Exception('Message {msg} maps to undeclared variable {v}'.format(msg=msg, v=v2))
      #end if
    #end for
  #end __check_model

  def _init_placeholders(self):
    # Initialize tensorflow placeholders for feeding matrices to the model.
    self.matrix_placeholders = {}
    for m in self.mat:
      if type(self.mat[m][1]) == int:
        self.matrix_placeholders[m] = tf.placeholder(self.float_dtype, shape=(None,self.mat[m][1]), name=m)
      else:
        self.matrix_placeholders[m] = tf.placeholder(self.float_dtype, shape=(None,None), name=m)
      #end if
      self.time_steps = tf.placeholder(tf.int32, shape=(), name='time_steps')
    #end for
  #end _init_placeholders

  def _init_parameters(self):
    # Init embeddings
    self._initial_embeddings = {
      v: tf.get_variable(
        initializer = tf.random_normal( (1,d) ),
        dtype = self.float_dtype,
        name = '{v}_init'.format( v = v )
      ) for (v,d) in self.var.items()
    }
    # Init LSTM cells
    self._RNN_cells = {
      v: tf.contrib.rnn.LayerNormBasicLSTMCell(
        d,
        activation = self.Cell_activation
      ) for (v,d) in self.var.items()
    }
    # Init message-computing MLPs
    self._msg_MLPs = {
      msg: Mlp(
        layer_sizes = [ self.var[vin] ] * self.MLP_depth + [ self.var[vout] ],
        activations = [ self.Msg_activation ] * self.MLP_depth + [ self.Msg_last_activation ],
        kernel_initializer = self.MLP_weight_initializer(),
        bias_initializer = self.MLP_weight_initializer(),
        name = msg,
        name_internal_layers = True
      ) for msg, (vin,vout) in self.msg.items()
    }
  #end _init_parameters

  def _init_utilities(self):
    # Generates utility variables to be called inside the processing graph
    self._num_vars = {}
    for m, (v1,v2) in self.mat.items():
      if v1 not in self._num_vars:
        self._num_vars[v1] = tf.shape(self.matrix_placeholders[m])[0]
      #end if
      if v2 not in self._num_vars:
        self._num_vars[v2] = tf.shape(self.matrix_placeholders[m])[1]
      #end if
    #end for
  #end _init_utilities

  def _run(self):
    states = {}
    for v, init in self._initial_embeddings.items():
      denom = tf.sqrt( tf.cast( self.var[v], self.float_dtype ) )
      h0 = tf.tile( tf.div( init, denom ), ( self._num_vars[v], 1 ) )
      c0 = tf.zeros_like( h0, dtype = self.float_dtype )
      states[v] = tf.contrib.rnn.LSTMStateTuple( h = h0, c = c0 )
    #end for
    
    _, self.last_states = tf.while_loop(
      self.__while_cond,
      self.__while_body,
      [ 0, states ]
    )
  #end _run
  
  def __while_body( self, t, states ):
    new_states = {}
    for v in self.var:
      inputs = []
      for update in self.loop[v]:
        if 'var' in update:
          y = states[update['var']].h
          if 'fun' in update:
            y = update['fun']( y )
          #end if
          if 'msg' in update:
            y = self._msg_MLPs[ update['msg'] ]( y )
          #end if
          if 'mat' in update:
            y = tf.matmul(
              self.matrix_placeholders[ update['mat'] ],
              y,
              adjoint_a = (
                'transpose?' in update
                and update['transpose?']
              )
            )
          #end if
          inputs.append( y )
        else:
          inputs.append( self.matrix_placeholders[ update['mat'] ] )
        #end if
      #end for
      inputs = tf.concat( inputs, axis = 1 )
      with tf.variable_scope( '{v}_cell'.format( v = v ) ):
        _, new_states[v] = self._RNN_cells[v]( inputs = inputs, state = states[v] )
      #end {v}_Cell variable scope
    #end for
    return ( t + 1 ), new_states
  #end while_body
  
  def __while_cond( self, t, states ):
    return tf.less( t, self.time_steps )
  #end while_cond
  
#end class GraphNN
