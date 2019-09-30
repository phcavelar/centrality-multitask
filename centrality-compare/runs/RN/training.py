# Import libraries and set random seeds for reproducibility
random_seed = 1237
import random
random.seed( random_seed )
import numpy as np
np.random.seed( random_seed )
import tensorflow as tf
tf.set_random_seed( random_seed )
# Import model and instance loader
import model
from instance_loader import InstanceLoader
import os, sys, itertools, util, metrics

METRICS_LIST = [ "TPR", "TNR", "PPV", "NPV", "FNR", "FPR", "FDR", "FOR", "ACC", "F1", "MCC", "BM", "MK", "P@10min", "P@10max", "AUC" ]

def get_metrics_from_batch( predictions_list, labels_list ):
  """
  Gets all the metrics from a batch for a specific centrality
  """
  bTPR, bTNR, bPPV, bNPV, bFNR, bFPR, bFDR, bFOR, bACC, bF1, bMCC, bBM, bMK, bP10min, bP10max, bAUC = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  for predictions, labels in zip( predictions_list, labels_list ):
    TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1, MCC, BM, MK, P10min, P10max, AUC = get_metrics( predictions, labels )
    bTPR, bTNR, bPPV, bNPV, bFNR, bFPR, bFDR, bFOR, bACC, bF1, bMCC, bBM, bMK, bP10min, bP10max, bAUC = map(
      lambda pair: pair[0] + pair[1],
      zip(
        (bTPR, bTNR, bPPV, bNPV, bFNR, bFPR, bFDR, bFOR, bACC, bF1, bMCC, bBM, bMK, bP10min, bP10max, bAUC),
        (TPR,  TNR,  PPV,  NPV,  FNR,  FPR,  FDR,  FOR,  ACC,  F1,  MCC,  BM,  MK,  P10min,  P10max, AUC)
      )
    )
  #end for
  b = len( labels_list )
  bTPR, bTNR, bPPV, bNPV, bFNR, bFPR, bFDR, bFOR, bACC, bF1, bMCC, bBM, bMK, bP10min, bP10max, bAUC = map(
    lambda x: x / b,
    (bTPR, bTNR, bPPV, bNPV, bFNR, bFPR, bFDR, bFOR, bACC, bF1, bMCC, bBM, bMK, bP10min, bP10max, bAUC)
  )
  return bTPR, bTNR, bPPV, bNPV, bFNR, bFPR, bFDR, bFOR, bACC, bF1, bMCC, bBM, bMK, bP10min, bP10max, bAUC
#end get_metrics_batch

def get_metrics( predictions, labels ):
  """
  Gets all the metrics for a specific centrality
  """
  TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1, MCC, BM, MK, AUC = metrics.confusion_matrix( predictions, labels )
  P10min = metrics.compute_precision_at_k( predictions, labels, 10, "min" )
  P10max = metrics.compute_precision_at_k( predictions, labels, 10, "max" )
  return TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR, ACC, F1, MCC, BM, MK, P10min, P10max, AUC
#end

def build_metrics_dict( centralities, header = False, header_prepend = "" ):
  """
  Builds the dictionary used to log the values or the dictionary containing the headers
  """
  metrics_dict                                                      = dict()
  for metric in METRICS_LIST:
    metrics_dict[metric]                                            = header_prepend + metric if header else 0
    for centrality in centralities:
      centrality_metric = "{c}_{m}".format(c=centrality,m=metric)
      metrics_dict[centrality_metric]                               = header_prepend + centrality_metric if header else 0
    #end for
  #end for
  metrics_dict["loss"]                                              = header_prepend + "loss" if header else 0
  for centrality in centralities:
    centrality_cost = "{c}_cost".format(c=centrality)
    metrics_dict[centrality_cost]                                   = header_prepend + centrality_cost if header else 0
  #end for
  return metrics_dict
#end build_metrics_dict

def log_metrics_dict( metrics_dict, centralities, log_file ):
  """
    Log a dictionary to a file.
    Note that it assumes one will write at least some value before the values being logged in the file and it also doesn't end a line.
  """
  print(
    "\t{val}".format(
      val = metrics_dict["loss"]
    ),
    end = "",
    file = log_file
  )
  for centrality in centralities:
    print(
      "\t{val}".format(
        val = metrics_dict["{c}_cost".format(c=centrality)]
      ),
      end = "",
      file = log_file
    )
  #end for
  for metric in METRICS_LIST:
    print(
      "\t{val}".format(
        val = metrics_dict[metric]
      ),
      end = "",
      file = log_file
    )
  #end for
  for centrality in centralities:
    for metric in METRICS_LIST:
      print(
        "\t{val}".format(
          val = metrics_dict["{c}_{m}".format(c=centrality,m=metric)]
        ),
        end = "",
        file = log_file
      )
    #end for
  #end for
#end log_metrics_dict

def train(
  session,
  model_dict,
  time_steps,
  centralities,
  epochs_to_run,
  train_instance_loader,
  batch_size,
  test_batch_size,
  batches_per_epoch,
  test_instance_loader,
  epoch_logging_file,
  batch_logging_file,
  model_checkpoint_filename,
  log_to_stdout = False
):
  """
  Runs the training procedure, logs the metrics and saves the model's weights to a checkpoint after every epoch.
  """
  log_epoch( "epoch_id", build_metrics_dict( centralities, header = True, header_prepend = "train_" ), build_metrics_dict( centralities, header = True, header_prepend = "test_" ), centralities, epoch_logging_file )
  log_batch( "epoch_id", "batch_id", build_metrics_dict( centralities, header = True ), centralities, batch_logging_file )
  
  print( "Starting training for {} epochs".format( epochs_to_run ) )
  for epoch_id in range( epochs_to_run ):
    if log_to_stdout:
      print( "Epoch\t{}".format( epoch_id ), end = "", file = sys.stdout )
      log_metrics_dict( build_metrics_dict( centralities, header = True ), centralities, sys.stdout )
      print( "", flush = True, file = sys.stdout )
    #end if
    run_epoch(
      epoch_id,
      session,
      model_dict,
      time_steps,
      centralities,
      train_instance_loader,
      batch_size,
      test_batch_size if epoch_id != epochs_to_run - 1 else 1,
      batches_per_epoch,
      test_instance_loader,
      epoch_logging_file,
      batch_logging_file,
      log_to_stdout = log_to_stdout
    )
    print( "SAVING MODEL WEIGHTS TO {}".format( model_checkpoint_filename ) )
    util.save_weights( session, model_checkpoint_filename )
  #end for
#end train

def log_epoch(
  epoch_id,
  epoch_train_metrics_dict,
  epoch_test_metrics_dict,
  centralities,
  epoch_logging_file
):
  # Log the training part of the epoch
  print( epoch_id, end = "", file = epoch_logging_file )
  log_metrics_dict( epoch_train_metrics_dict, centralities, epoch_logging_file )
  # Log the testing part of the epoch and flush the line
  log_metrics_dict( epoch_test_metrics_dict, centralities, epoch_logging_file )
  print( "", flush = True, file = epoch_logging_file )
#end log_epoch

def run_epoch(
  epoch_id,
  session,
  model_dict,
  time_steps,
  centralities,
  train_instance_loader,
  batch_size,
  test_batch_size,
  batches_per_epoch,
  test_instance_loader,
  epoch_logging_file,
  batch_logging_file,
  log_to_stdout = False
):
  """
    Runs and logs a training/testing epoch
  """
  # Build the metrics dictionary for logging
  epoch_train_metrics_dict = build_metrics_dict( centralities )
  # Reset instance loader
  train_instance_loader.reset()
  for batch_id, batch in itertools.islice( enumerate( train_instance_loader.get_batches( batch_size ) ), batches_per_epoch ):
    # Run and log every training batch, accumulating the metrics
    batch_metrics_dict = run_batch(
      epoch_id,
      batch_id,
      session,
      model_dict,
      time_steps,
      centralities,
      batch,
      batch_logging_file,
      train = True,
      log_to_stdout = log_to_stdout
    )
    for metric in epoch_train_metrics_dict:
      epoch_train_metrics_dict[metric] += batch_metrics_dict[metric]
    #end for
  #end for
  # Normalize the metrics by the number of batches
  for metric in epoch_train_metrics_dict:
    epoch_train_metrics_dict[metric] /= batches_per_epoch
  #end for
  
  # Test
  # Build the metrics dictionary for logging
  epoch_test_metrics_dict = build_metrics_dict( centralities )
  # Reset instance loader
  test_instance_loader.reset()
  # Counter for the number of instances
  number_of_test_batches = 0
  for cbat, batch in enumerate( test_instance_loader.get_batches( test_batch_size ) ):
    # Open a null file so that we don't log every test instance being ran as a separate batch
    with open(os.devnull, 'w') as nullfile:
      # Run and log every test instance, accumulating the metrics
      batch_metrics_dict = run_batch(
        epoch_id,
        "test",
        session,
        model_dict,
        time_steps,
        centralities,
        batch,
        nullfile,
        train = False,
        log_to_stdout = log_to_stdout
      )
    #end with
    for metric in epoch_test_metrics_dict:
      epoch_test_metrics_dict[metric] += batch_metrics_dict[metric]
    #end for
    number_of_test_batches += 1
  #end for
  # Normalize the metrics by the number of test instances
  for metric in epoch_test_metrics_dict:
    epoch_test_metrics_dict[metric] /= number_of_test_batches
  #end for
  
  log_epoch( epoch_id, epoch_train_metrics_dict, epoch_test_metrics_dict, centralities, epoch_logging_file )
  if log_to_stdout:
    print( "EPOCH\t", end = "" )
    log_epoch( "summary", epoch_train_metrics_dict, epoch_test_metrics_dict, centralities, sys.stdout )
  #end if
#end run_epoch

def log_batch(
  epoch_id,
  batch_id,
  batch_metrics_dict,
  centralities,
  batch_logging_file
):
  print( 
    "{eid}\t{bid}".format(
      eid = epoch_id,
      bid = batch_id
    ),
    end = "",
    file = batch_logging_file
  )
  log_metrics_dict( batch_metrics_dict, centralities, batch_logging_file )
  print( "", flush = True, file = batch_logging_file )
#end

def run_batch(
  epoch_id,
  batch_id,
  session,
  model_dict,
  time_steps,
  centralities,
  batch,
  batch_logging_file,
  train = False,
  log_to_stdout = False
):
  """
  Runs and logs a batch
  """
  # Build metrics dictionary for logging
  batch_metrics_dict = build_metrics_dict( centralities )
  # Transform sparse batch labels to dense
  labels = {
    centrality: util.sparse_to_dense( batch["{c}_compare".format(c=centrality)] )
    for centrality in centralities
  }
  # Build the feed_dict
  feed_dict = {
    model_dict["{c}_labels".format(c=centrality)]: labels[centrality]
    for centrality in centralities
  }
  feed_dict[ model_dict["gnn"].matrix_placeholders["M"] ] = util.sparse_to_dense( batch["matrix"] )
  feed_dict[ model_dict["gnn"].time_steps ] = time_steps
  feed_dict[ model_dict["nodes_n"] ] = batch["problem_n"]
  # Train if required
  if train:
    returned_values = session.run(
      model_dict["train_step"],
      feed_dict = feed_dict
    )
  #end if
  # Get logits for batch
  returned_probabilities = session.run(
    [
      model_dict["{c}_probabilities".format( c = centrality ) ]
      for centrality in centralities
    ],
    feed_dict = feed_dict
  )
  # Get losses for batch
  returned_losses = session.run(
    [
      model_dict["loss"]
    ] + [
      model_dict["{c}_cost".format( c = centrality ) ]
      for centrality in centralities
    ],
    feed_dict = feed_dict
  )
  # Update the overall loss
  batch_metrics_dict["loss"] = returned_losses[0]
  # Update each centrality's value
  for centrality, predictions, cost in zip( centralities, returned_probabilities, returned_losses[1:] ):
    metric_values = get_metrics_from_batch(
      model.gather_matrices(
        predictions,
        batch["problem_n"]
      ),
      model.separate_batch(
        labels[centrality],
        batch["problem_n"]
      )
    )
    # Update loss for the centrality
    batch_metrics_dict["{c}_cost".format(c=centrality)] = cost
    # Update every other metric for the centrality
    for metric, value in zip( METRICS_LIST, metric_values ):
      batch_metrics_dict["{c}_{m}".format(c=centrality,m=metric)] = value
    #end for
  #end for
  
  # For every metric, comput the average over the centralities
  for metric in METRICS_LIST:
    for centrality in centralities:
      batch_metrics_dict[metric] += batch_metrics_dict["{c}_{m}".format(c=centrality,m=metric)]
    #end for
    batch_metrics_dict[metric] /= len( centralities )
  #end for
  
  # Log the batch
  log_batch( epoch_id, batch_id, batch_metrics_dict, centralities, batch_logging_file )
  if log_to_stdout:
    log_batch( "batch", batch_id, batch_metrics_dict, centralities, sys.stdout )
  #end if
  
  return batch_metrics_dict
#end run_batch

if __name__ == "__main__":
  embedding_size = 64
  epochs_to_run = 32
  batches_per_epoch = 32
  batch_size = 32
  test_batch_size = 32
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
  training_instance_loader_random_generator = random.Random( random_seed )
  test_instance_loader_random_generator = random.Random( random_seed )

  training_instance_loader = InstanceLoader("./instances", rng = training_instance_loader_random_generator )
  test_instance_loader = InstanceLoader("./test-instances", rng = test_instance_loader_random_generator )

  epoch_logging_file = open( "{fname}.epoch.log".format( fname = fname ), "w" )
  batch_logging_file = open( "{fname}.batch.log".format( fname = fname ), "w" )
  model_checkpoint_filename = fname

  # Disallow GPU use
  config = tf.ConfigProto(
  #  device_count =  {"GPU": 0 },
  #  inter_op_parallelism_threads = 1,
  #  intra_op_parallelism_threads = 1
  #  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1/3.1)
  )
  with tf.Session(config=config) as sess:
    # Initialize global variables
    print( "Initializing global variables ... " )
    sess.run( tf.global_variables_initializer() )
    
    train(
      sess,
      GNN,
      time_steps,
      centralities,
      epochs_to_run,
      training_instance_loader,
      batch_size,
      test_batch_size,
      batches_per_epoch,
      test_instance_loader,
      epoch_logging_file,
      batch_logging_file,
      model_checkpoint_filename,
      log_to_stdout = True
    )
  #end Session
