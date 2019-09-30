
import os, sys, itertools, argparse, random
import tensorflow as tf
import numpy as np
import model, metrics, util
from instance_loader import InstanceLoader, create_batch
from training import get_metrics_from_batch, get_metrics, build_metrics_dict, log_batch

METRICS_LIST = [ "TPR", "TNR", "PPV", "NPV", "FNR", "FPR", "FDR", "FOR", "ACC", "F1", "MCC", "BM", "MK", "P@10min", "P@10max", "AUC" ]

def build_feed_dict(model_dict, batch, centralities, time_steps):
    feed_dict = {}
    for centrality in centralities:
        feed_dict[ model_dict['{c}_labels'.format(c=centrality)] ] = util.sparse_to_dense(batch['{c}_compare'.format(c=centrality)])
    #end
    feed_dict[ model_dict['gnn'].matrix_placeholders['M'] ] = util.sparse_to_dense(batch['matrix'])
    feed_dict[ model_dict['gnn'].time_steps ] = time_steps
    feed_dict[ model_dict['nodes_n'] ] = batch['problem_n']
    return feed_dict
#end

def test(
  session,
  model_dict,
  time_steps,
  centralities,
  instance_loader,
  logging_file,
  model_checkpoint_filename,
  log_to_stdout = False,
  log_progress_to_stdout = False,
  log_individual_entries = False
  ):
#end

    log_batch('epoch_id', 'batch_id', build_metrics_dict(centralities, header=True), centralities, logging_file)

    # Build accumulated metrics dict to compute averages at the end of the test
    acc_metrics_dict = build_metrics_dict(centralities)
    
    # Reset instance loader
    instance_loader.reset()
    n_problems = len(instance_loader.filenames)
    # Iterate over all instances
    for i, (filename, instance) in enumerate(zip(instance_loader.filenames, instance_loader.get_instances(n_problems))):

        # Create (size 1) batch from instance
        batch = create_batch([instance])

        # Build the feed_dict
        feed_dict = build_feed_dict(model_dict,batch,centralities,time_steps)

        # Build metrics dictionary for logging
        metrics_dict = build_metrics_dict(centralities)

        # Register training loss
        metrics_dict['loss'] = session.run(model_dict['loss'], feed_dict=feed_dict)
        acc_metrics_dict['loss'] += metrics_dict['loss']

        # Iterate over centralities
        for centrality in centralities:
            
            # Fetch predictions and cost
            predictions, cost = session.run(
                [
                    model_dict['{c}_probabilities'.format(c=centrality)],
                    model_dict['{c}_cost'.format(c=centrality)]
                ],
                feed_dict=feed_dict
            )

            # Register the cost relative to this centrality
            metrics_dict['{c}_cost'.format(c=centrality)] = cost
            acc_metrics_dict['{c}_cost'.format(c=centrality)] += cost

            # Compute metrics relative to this centrality
            metric_values = get_metrics_from_batch(
                model.gather_matrices(
                    predictions,
                    batch["problem_n"]
                ),
                model.separate_batch(
                    feed_dict[model_dict['{c}_labels'.format(c=centrality)]],
                    batch["problem_n"]
                )
            )

            for metric, value in zip(METRICS_LIST, metric_values):
                metrics_dict["{c}_{m}".format(c=centrality,m=metric)] = value
                acc_metrics_dict["{c}_{m}".format(c=centrality,m=metric)] += value
            #end
        #end

        # For every metric, comput the average over the centralities
        for metric in METRICS_LIST:
            metrics_dict[metric] = np.mean([ metrics_dict['{c}_{m}'.format(c=centrality,m=metric)] for centrality in centralities ])
        #end for

        # Log the batch
        if log_individual_entries:
            log_batch(0, i, metrics_dict, centralities, logging_file)
        #end
        if log_to_stdout:
            log_batch('batch', i, metrics_dict, centralities, sys.stdout )
        #end
        if log_progress_to_stdout:
            if i % int(np.round(n_problems/10)) == 0:
                print('\t{}% Complete'.format(int(np.floor(100*i/n_problems))))
            #end
        #end
    #end

    # Compute averages
    avg_metrics_dict = build_metrics_dict(centralities)
    avg_metrics_dict['loss'] = acc_metrics_dict['loss'] / n_problems
    for centrality in centralities:
        avg_metrics_dict['{c}_cost'.format(c=centrality)] = acc_metrics_dict['{c}_cost'.format(c=centrality)] / n_problems
        for metric in METRICS_LIST:
            avg_metrics_dict['{c}_{m}'.format(c=centrality,m=metric)] = acc_metrics_dict['{c}_{m}'.format(c=centrality,m=metric)] / n_problems
        #end
    #end
    for metric in METRICS_LIST:
        avg_metrics_dict[metric] = np.mean([ avg_metrics_dict['{c}_{m}'.format(c=centrality,m=metric)] for centrality in centralities ])
    #end

    # Log the averages
    log_batch(0, 'AVERAGE', avg_metrics_dict, centralities, logging_file)
    if log_to_stdout:
        log_batch('batch', 'AVERAGE', avg_metrics_dict, centralities, sys.stdout )
    #end
#end

if __name__ == '__main__':

    # Add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', default='centrality-betweenness-closeness-degree-eigenvector', help='Which trained model to load')
    parser.add_argument('-savepath', default='test.log', help='Where to save the test results')
    parser.add_argument('-instances', default='test-instances', help='Test instances folder path')
    parser.add_argument('-timesteps', default=32, help='Number of timesteps to run the model')
    parser.add_argument('--log_entries', const=True, default=False, action='store_const', help='Log individual entries?')
    parser.add_argument('--betweenness', const=True, default=False, action='store_const', help='Test Betweennes?')
    parser.add_argument('--closeness', const=True, default=False, action='store_const', help='Test Closeness?')
    parser.add_argument('--degree', const=True, default=False, action='store_const', help='Test Degree?')
    parser.add_argument('--eigenvector', const=True, default=False, action='store_const', help='Test Eigenvector?')
    parser.add_argument('--all', const=True, default=False, action='store_const', help='Test all centralities?')

    # Parse arguments from command line
    args = vars(parser.parse_args())

    centralities = ['betweenness','closeness','degree','eigenvector']
    if not args['all']:
        centralities = [ c for c in centralities if args[c] ]
    #end

    embedding_size = 64
    time_steps = args['timesteps']

    # Build model
    print( "Building model ..." )
    GNN = model.model_builder(embedding_size,centralities)
    
    # Load instances with a predefined seed and separate random generator for reproducibility
    instance_loader_random_generator = random.Random(0)
    instance_loader = InstanceLoader(args['instances'], rng=instance_loader_random_generator)

    logging_file = open(args['savepath'], 'w')
    model_checkpoint_filename = args['model']

    # Disallow GPU use
    config = tf.ConfigProto(
        device_count =  {"GPU": 0 },
        inter_op_parallelism_threads = 1,
        intra_op_parallelism_threads = 1
    )
    with tf.Session(config=config) as sess:
        
        # Initialize global variables
        print( "Initializing global variables ... " )
        sess.run( tf.global_variables_initializer() )

        print( "Loading weights ..." )
        util.load_weights(sess, model_checkpoint_filename)
        
        test(
            sess,
            GNN,
            time_steps,
            centralities,
            instance_loader,
            logging_file,
            model_checkpoint_filename,
            log_to_stdout = False,
            log_progress_to_stdout = True,
            log_individual_entries = args['log_entries']
        )
    #end
#end
