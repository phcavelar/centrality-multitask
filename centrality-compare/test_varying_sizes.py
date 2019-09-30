import tensorflow as tf
import numpy as np
from instance_loader import InstanceLoader
import pprint, os, sys, model, random, util, time, itertools
import training as tr

from matplotlib import pyplot as plt

if __name__ == '__main__':
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

  model_checkpoint_filename = fname

  # Init list of problem sizes
  problem_sizes = np.arange(32,240+1,16)

  # Disallow GPU use
  config = tf.ConfigProto(
#    device_count =  {"GPU": 0 },
#    inter_op_parallelism_threads = 1,
#    intra_op_parallelism_threads = 1
  )
  with tf.Session(config=config) as sess:
    print( "Initializing global variables ... " )
    sess.run( tf.global_variables_initializer() )
    print( "Loading weights ..." )
    util.load_weights(sess,fname)
    
    # Init accuracies vector
    accuracies = np.zeros(len(problem_sizes))

    for i,n in enumerate(problem_sizes):
      test_logging_file = open( "./test-instances-sizes-{size}-{fname}.log".format( fname = fname, size = n ), "w" )
      # Init instance loader for this problem size
      test_instance_loader = InstanceLoader('test-varying-sizes/n={}'.format(n), rng=random.Random(0))
    
      total_test_metrics_dict = tr.build_metrics_dict( centralities )
      number_of_test_instances = 0
      tr.log_batch( "epoch_id", "batch_id", tr.build_metrics_dict( centralities, header = True ), centralities, test_logging_file )
    
      for b, batch in enumerate( test_instance_loader.get_batches( batch_size ) ):
      #for b, batch in itertools.islice( enumerate( test_instance_loader.get_batches( batch_size ) ), 1 ):
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
          print( "{}\t{}".format(n, (b+1)*batch_size) )
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
      #end with
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
      accuracies[i] = np.mean( [ total_test_metrics_dict["{c}_ACC".format(c=c)] for c in centralities ] )
      #
    #end for
    # Get the ylim
    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    ax.set_xlabel('Problem size')
    ax.set_ylabel('Accuracy (%)')
    #ax.set_ylim(75,95)
    plt.axvline(x=32, c='black', linestyle='--')
    plt.axvline(x=128, c='black', linestyle='--')
    plt.xticks(problem_sizes)
    locs, labels = plt.yticks()

    ax.plot(problem_sizes, 100*accuracies, linestyle='-', marker='o', color = "#8A4F7D")
    ymin, ymax = ax.get_ylim()
    plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    ax.set_xlabel('Problem size')
    ax.set_ylabel('Accuracy (%)')
    #ax.set_ylim(75,95)
    plt.axvline(x=32, c='black', linestyle='--')
    plt.axvline(x=128, c='black', linestyle='--')
    plt.xticks(problem_sizes[::2])
    ax.set_ylim(0,ymax)
    ax.fill_betweenx([0,ymax], 32, 128, alpha=0.1, facecolor='#F7EF81')

    ax.plot(problem_sizes, 100*accuracies, linestyle='-', marker='o', color = "#8A4F7D")
    plt.savefig('test-varying-sizes-{fname}.eps'.format( fname = fname ), format='eps', dpi=200)
  #end
#end
