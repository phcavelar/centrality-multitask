import tensorflow as tf
import numpy as np
from instance_loader import InstanceLoader
import pprint, os, sys, model, random, util, time
import training as tr

if __name__ == "__main__":
  embedding_size = 64
  batch_size = 32
  time_steps = 32
  centralities = sorted( sys.argv[1:] )
  
  if len( centralities ) <= 0:
    raise ValueError( "No centrality passed" )
  #end if
  for centrality in centralities:
    if centrality not in [ "betweenness","closeness","degree","eigenvector" ]:
      raise ValueError( "Centrality {c} not one of the accepted ones.".format( c=centrality ) )
    #end if
  #end for
  
  fname = "centrality-" + "-".join( centralities )

  # Build model
  print( "Building model ..." )
  GNN = model.model_builder( embedding_size, centralities )
  # Load instances with a predefined seed and separate random generator for reproducibility
  random_seed = time.time()
  test_instance_loader_random_generator = random.Random( random_seed )
  test_instance_loader = InstanceLoader("./test-instances", rng = test_instance_loader_random_generator )

  test_logging_file = open( "./test-instances-{fname}.log".format( fname = fname ), "w" )
  model_checkpoint_filename = fname

  # Disallow GPU use
  config = tf.ConfigProto(
  #  device_count =  {"GPU": 0 },
  #  inter_op_parallelism_threads = 1,
  #  intra_op_parallelism_threads = 1
  )
  with tf.Session(config=config) as sess:
    print( "Initializing global variables ... " )
    sess.run( tf.global_variables_initializer() )
    print( "Loading weights ..." )
    util.load_weights(sess,fname)
    print( "Loading instances ..." )
    total_test_metrics_dict = tr.build_metrics_dict( centralities )
    number_of_test_instances = 0
    tr.log_batch( "epoch_id", "batch_id", tr.build_metrics_dict( centralities, header = True ), centralities, test_logging_file )
    
    for b, batch in enumerate( test_instance_loader.get_batches( batch_size ) ):
      with open(os.devnull, 'w') as nullfile:
        batch_test_metrics_dict = tr.run_batch(
          "test", 
          b, 
          sess,
          GNN,
          time_steps,
          centralities,
          batch,
          nullfile,
          train=False,
          log_to_stdout=False
        )
        pprint.pprint((1+b)*batch_size)
        tr.log_batch(
          "test",
          b,
          batch_test_metrics_dict,
          centralities,
          test_logging_file
        )
        for metric in total_test_metrics_dict:
          total_test_metrics_dict[metric] += batch_test_metrics_dict[metric]
        #end for
        number_of_test_instances += 1
     #end for
    # Normalize the metrics by the number of test instances
    for metric in total_test_metrics_dict:
      total_test_metrics_dict[metric] /= number_of_test_instances
    #end for
    tr.log_batch(
          "test",
          "avg",
          total_test_metrics_dict,
          centralities,
          test_logging_file
        )

  #end Session
