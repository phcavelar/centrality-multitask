import numpy as np
import tensorflow as tf
from graphnn import GraphNN
from mlp import Mlp
from scipy.stats import rankdata

def model_builder(
  d,
  centralities,
  learning_rate = 2e-5,
  parameter_l2norm_scaling = 1e-10,
  global_norm_gradient_clipping_ratio = 0.65
):
  # Build the model
  
  # Build a dictionary to index the values
  centrality_dict = { centrality:i for i, centrality in enumerate( centralities ) }
  
  # Define placeholder for result values (one per problem)
  labels = [
    tf.placeholder(
      tf.float32,
      [ None ],
      name = "labels_{}".format( centrality )
    )
    for centrality in centralities
  ]
  
  cmp_labels = [
    tf.placeholder(
      tf.float32,
      [ None, None ],
      name = "labels_{}".format( centrality )
    )
    for centrality in centralities
  ]
  
  nodes_n = tf.placeholder( tf.int32, [ None ], name = "nodes_n" )

  # Define Graph neural network
  gnn = GraphNN(
    {
      "N": d
    },
    {
      "M": ("N","N")
    },
    {
      "Nsource": ("N","N"),
      "Ntarget": ("N","N")
    },
    {
      "N": [
        {
          "mat": "M",
          "var": "N",
          "msg": "Nsource"
        },
        {
          "mat": "M",
          "transpose?": True,
          "var": "N",
          "msg": "Ntarget"
        }
      ]
    },
    name = "centrality",
  )
  
  # Define votes
  MLPs = [
    Mlp(
      layer_sizes          = [ d for _ in range(2) ],
      activations          = [ tf.nn.relu for _ in range(2) ],
      output_size          = 1,
      name_internal_layers = True,
      kernel_initializer   = tf.contrib.layers.xavier_initializer(),
      bias_initializer     = tf.zeros_initializer(),
      name                 = "{}_MLP".format( centrality )
    )
    for centrality in centralities
  ]
  
  # Compute the number of variables
  n = tf.shape( gnn.matrix_placeholders["M"] )[0]
  # Compute number of problems
  p = tf.shape( nodes_n )[0]

  # Get the last embeddings
  N_n = gnn.last_states["N"].h
  
  Cs = [
    tf.squeeze( 
      MLPs[ centrality_dict[ centrality ] ](
        N_n
      )
    )
    for centrality in centralities
  ]

  # Reorganize votes' result to obtain a prediction for each problem instance
  def _vote_while_cond(i, n_acc, array_accuracies, array_costs, cmp_matrices):
    return tf.less( i, p )
  #end _vote_while_cond

  def _vote_while_body(i, n_acc, array_accuracies, array_costs, cmp_matrices):
    # Gather the embeddings for that problem
    p_Cs = [
      tf.slice(
        Cs[ centrality_dict[centrality] ],
        [n_acc],
        [nodes_n[i]]
      )
      for centrality in centralities
    ]
    
    problem_labels = [
      tf.slice(
        labels[ centrality_dict[ centrality ] ],
        [n_acc],
        [nodes_n[i]]
      )
      for centrality in centralities
    ]
    
    #Calculate cost for this problem
    problem_costs = [
      tf.losses.mean_squared_error(
        labels = problem_labels[ centrality_dict[ centrality ] ],
        predictions = p_Cs[ centrality_dict[ centrality ] ]
      )
      for centrality in centralities
    ]
  
    # Build comparison matrix
    p_Cs_expanded = [
      tf.expand_dims(
        p_Cs[ centrality_dict[ centrality ] ],
        0
      )
      for centrality in centralities
    ]
    N1_Cs = [
      tf.tile(
        p_Cs_expanded[ centrality_dict[ centrality ] ],
        (nodes_n[i],1)
      )
      for centrality in centralities
    ]
    
    p_Cs_expanded_transposed = [
      tf.transpose(
        p_Cs_expanded[ centrality_dict[ centrality ] ],
        (1,0)
      )
      for centrality in centralities
    ]
    N2_Cs = [
      tf.tile(
        p_Cs_expanded_transposed[ centrality_dict[ centrality ] ],
        (1,nodes_n[i])
      )
      for centrality in centralities
    ]
    
    pred_matrices = [
      tf.cast(
        tf.greater(
          N2_Cs[ centrality_dict[ centrality ] ],
          N1_Cs[ centrality_dict[ centrality ] ]
        ),
        tf.float32
      )
      for centrality in centralities
    ]
    
    problem_cmp_labels = [
      tf.slice(
        cmp_labels[ centrality_dict[ centrality ] ],
        [n_acc, n_acc],
        [nodes_n[i], nodes_n[i]]
      )
      for centrality in centralities
    ]
    
    #Compare labels to predicted values
    problem_accuracies = [
      tf.reduce_mean(
        tf.cast(
          tf.equal(
            tf.round(pred_matrices[ centrality_dict[ centrality ] ]), problem_cmp_labels[ centrality_dict[ centrality ] ]
          ),
          tf.float32
        )
      ) for centrality in centralities
    ]
        
    # Update TensorArrays
    array_accuracies = array_accuracies.write( i, problem_accuracies )
    array_costs = array_costs.write(i, problem_costs )

    cmp_matrices = [
        cmp_matrices[centrality_dict[centrality]].write(
            i,
            tf.reshape( pred_matrices[centrality_dict[centrality]], [-1] )
        )
        for centrality in centralities
    ]
    
    return tf.add( i, tf.constant( 1 ) ), tf.add( n_acc, nodes_n[i] ), array_accuracies, array_costs, cmp_matrices
  #end _vote_while_body
        
  array_accuracies = tf.TensorArray( size = p, dtype = tf.float32 )
  array_costs = tf.TensorArray( size = p, dtype = tf.float32 )
  cmp_matrices = [
    tf.TensorArray( size = p, dtype = tf.float32, infer_shape = False, element_shape = [None] )
    for centrality in centralities
  ]

  _, _, array_accuracies, array_costs, cmp_matrices = tf.while_loop(
    _vote_while_cond,
    _vote_while_body,
    [ tf.constant( 0, dtype = tf.int32 ), tf.constant( 0, dtype = tf.int32 ), array_accuracies, array_costs, cmp_matrices ]
  )
  
  array_accuracies = array_accuracies.stack()
  array_costs = array_costs.stack()
  
  array_accuracies = [
    tf.reduce_mean(
      array_accuracies[:, centrality_dict[ centrality ] ]
    )
    for centrality in centralities
  ]
  array_costs = [
    tf.reduce_mean(
      array_costs[:, centrality_dict[ centrality ] ]
    )
    for centrality in centralities
  ]
  
  vars_cost = tf.zeros([])
  tvars = tf.trainable_variables()
  for var in tvars:
    vars_cost = tf.add( vars_cost, tf.nn.l2_loss( var ) )
  #end for
  loss = tf.add_n(
    array_costs + [
      tf.multiply(
        vars_cost,
        parameter_l2norm_scaling
      )
    ]
  )
  
  optimizer = tf.train.AdamOptimizer( name = "Adam", learning_rate = learning_rate )
  grads, _ = tf.clip_by_global_norm( tf.gradients( loss, tvars ), global_norm_gradient_clipping_ratio )
  train_step = optimizer.apply_gradients( zip( grads, tvars ) )
  overall_accuracy = tf.reduce_mean( array_accuracies )
  
  model_dict                                          = dict()
  model_dict["no_op"]                                 = tf.no_op()
  model_dict["gnn"]                                   = gnn
  model_dict["accuracy"]                              = overall_accuracy
  model_dict["loss"]                                  = loss
  model_dict["nodes_n"]                               = nodes_n
  model_dict["train_step"]                            = train_step
  for centrality in centralities:
    model_dict["{}_labels".format(centrality)]        = labels[ centrality_dict[centrality] ]
    model_dict["{}_cmp_labels".format(centrality)]    = cmp_labels[ centrality_dict[centrality] ]
    model_dict["{}_cost".format(centrality)]          = array_costs[ centrality_dict[centrality] ]
    model_dict["{}_accuracy".format(centrality)]      = array_accuracies[ centrality_dict[centrality] ]
    model_dict["{}_predictions".format(centrality)]   = Cs[ centrality_dict[centrality] ]
    model_dict["{}_cmp".format(centrality)]           = cmp_matrices[ centrality_dict[centrality] ].concat()
  #end for
  # Return model dictionary
  return model_dict
#end

def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))
#end sigmoid

def gather_matrices( flattened_values, problem_n ):
  logits_list = list()
  i_acc = 0
  for n in problem_n:
    logits = np.zeros( [n,n] )
    for i in range(n):
      for j in range(n):
        logits[i,j] = flattened_values[ i_acc + n * i + j ]
      #end for
    #end for
    logits_list.append( logits )
    i_acc += n*n
  #end for
  return logits_list
#end gather_matrices

def separate_batch( batch_matrix, problem_n ):
  problem_list = list()
  n_acc = 0
  for n in problem_n:
    problem = batch_matrix[ n_acc : n_acc + n, n_acc : n_acc + n ].copy()
    problem_list.append( problem )
    n_acc += n
  #end for
  return problem_list
#end separate_batch

if __name__ == "__main__":
  import instance_loader
  #import pprint
  import util
  
  time_steps = 3
  batch_size = 2
  k = 10
  
  batch = instance_loader.InstanceLoader("instances").get_batch( batch_size )

  with tf.variable_scope( "degree_only" ):
    d = model_builder( 2, [ "degree" ] )
  #end
  with tf.variable_scope( "transfer" ):
    t = model_builder( 2, [ "degree", "closeness", "betweenness", "eigenvector" ] )
  #end
  
  pprint.pprint( d )
  print()
  pprint.pprint( t )
  
  with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )
    
    matrix = util.sparse_to_dense( batch["matrix"] )
    degree_compare = util.sparse_to_dense( batch["degree_compare"] )
    closeness_compare = util.sparse_to_dense( batch["closeness_compare"] )
    betweenness_compare = util.sparse_to_dense( batch["betweenness_compare"] )
    eigenvector_compare = util.sparse_to_dense( batch["eigenvector_compare"] )
    
    dda,ddc,ddl,tda,tdc,tdl,tba,tbc,tbl,tca,tcc,tcl,tea,tec,tel = sess.run(
      [
        d["degree_accuracy"],
        d["degree_cost"],
        d["degree_logits"],
        t["degree_accuracy"],
        t["degree_cost"],
        t["degree_logits"],
        t["betweenness_accuracy"],
        t["betweenness_cost"],
        t["betweenness_logits"],
        t["closeness_accuracy"],
        t["closeness_cost"],
        t["closeness_logits"],
        t["eigenvector_accuracy"],
        t["eigenvector_cost"],
        t["eigenvector_logits"]
      ],
      feed_dict = {
        d["gnn"].matrix_placeholders["M"] : matrix,
        d["gnn"].time_steps               : time_steps,
        d["nodes_n"]                      : batch["problem_n"],
        d["degree_labels"]                : degree_compare,
        t["gnn"].matrix_placeholders["M"] : matrix,
        t["gnn"].time_steps               : time_steps,
        t["nodes_n"]                      : batch["problem_n"],
        t["degree_labels"]                : degree_compare,
        t["closeness_labels"]             : closeness_compare,
        t["betweenness_labels"]           : betweenness_compare,
        t["eigenvector_labels"]           : eigenvector_compare
      }
    )
    
    pprint.pprint(
      [
        batch["problem_n"],
        dda,
        ddc,
        list(
          zip(
            map( lambda x: rankdata(np.sum(np.around(sigmoid(x)),axis=0),"min"), gather_matrices( ddl, batch["problem_n"] ) ),
            map( lambda x: rankdata(np.sum(x,axis=0),"min"), separate_batch( degree_compare, batch["problem_n"] ) )
          )
        ),
        list(
          zip(
            map( lambda x: np.argpartition(np.sum(np.around(sigmoid(x)),axis=0),-k)[-k:], gather_matrices( ddl, batch["problem_n"] ) ),
            map( lambda x: np.argpartition(np.sum(x,axis=0),-k)[-k:], separate_batch( degree_compare, batch["problem_n"] ) )
          )
        ),
        tda,
        tdc,
        list( map( np.shape, gather_matrices( tdl, batch["problem_n"] ) ) ),
        tba,
        tbc,
        list( map( np.shape, gather_matrices( tbl, batch["problem_n"] ) ) ),
        tca,
        tcc,
        list( map( np.shape, gather_matrices( tcl, batch["problem_n"] ) ) ),
        tea,
        tec,
        list( map( np.shape, gather_matrices( tel, batch["problem_n"] ) ) )
      ]
    )
  #end with
