import tensorflow as tf
import numpy as np
from instance_loader import InstanceLoader
import pprint, sys, model, random, util, time, metrics, os
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
  #training_instance_loader_random_generator = random.Random( random_seed )

  #training_instance_loader = InstanceLoader("./instances", rng = training_instance_loader_random_generator )
  test_instance_loader = InstanceLoader("./test-realgraphs", rng = test_instance_loader_random_generator )

  test_logging_file = open( "./test-realgraphs/testreal-{fname}.log".format( fname = fname ), "w" )
  model_checkpoint_filename = fname

  # Disallow GPU use
  config = tf.ConfigProto(
    device_count =  {"GPU": 0 },
    inter_op_parallelism_threads = 1,
    intra_op_parallelism_threads = 1
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
    
    for b, batch in enumerate( test_instance_loader.get_batches( 1 ) ):
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
        pprint.pprint(batch["fnames"][0])
        tr.log_batch(
          "test",
          batch["fnames"][0],
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
      
#    batch = test_instance_loader.get_batch( 1 )
#    print( "Building feed dict ..." )
#    feed_dict = {}
#    feed_dict[ GNN["gnn"].matrix_placeholders["M"] ] = util.sparse_to_dense( batch["matrix"] )
#    feed_dict[ GNN["gnn"].time_steps ]               = time_steps
#    feed_dict[ GNN["nodes_n"] ]                      = batch["problem_n"]
#    for centrality in centralities:
#      clabels  = "{c}_labels" .format( c = centrality )
#      ccompare = "{c}_compare".format( c = centrality )
#      feed_dict[ GNN[clabels] ]                      = util.sparse_to_dense( batch[ccompare] )
#    #end for
#    print( "Gathering accuracy and loss ..." )
#    oacc, loss = sess.run(
#      [ GNN["accuracy"], GNN["loss"] ],
#      feed_dict = feed_dict
#    )
#    print( "Accuracy {acc:.4f}, loss {loss:.4f}".format( acc = oacc, loss = loss ) )
#    print( "Gathering probabilities ..." )
#    centrality_probabilities = sess.run(
#      [ GNN["{c}_probabilities".format( c = c )] for c in centralities ],
#      feed_dict = feed_dict
#    )
#    print( "Gather per-centrality accuracy, calculated inside tensorflow ..." )
#    tfaccs = sess.run(
#      [ GNN["{c}_accuracy".format( c = c )] for c in centralities ],
#      feed_dict = feed_dict
#    )
#    print( "For each centrality, plase compare the tf accuracy with the one calculated with the probabilities:" )
#    for probabilities, tfacc, c in zip( centrality_probabilities, tfaccs, centralities ):
#      ccompare = "{c}_compare".format( c = c )
#      bACC = 0
#      bnpeqacc = 0
#      for prd, lbl in zip(
#        model.gather_matrices( np.around( probabilities ), batch["problem_n"] ),
#        model.separate_batch( util.sparse_to_dense( batch[ccompare] ), batch["problem_n"] )
#      ):
#        TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1, MCC, BM, MK, AUC = metrics.confusion_matrix(
#          prd,
#          lbl
#        )
#        bACC += ACC
#      #end for
#      bACC /= len( batch["problem_n"] )
#      
#      
#      pprint.pprint(
#        list(
#          zip(
#            ["c", "tfacc", "bACC", "ACC" ],
#            [ c ,  tfacc ,  bACC ,  ACC   ]
#          )
#        )
#      )
    #end for
      
    #end Session
