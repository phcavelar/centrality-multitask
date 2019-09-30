
import os, sys, itertools, argparse, random
import tensorflow as tf
import numpy as np
import model, metrics, util
from instance_loader import InstanceLoader, create_batch
from training import get_metrics_from_batch, get_metrics, build_metrics_dict, log_batch
from dataset import create_dataset
from test import build_feed_dict, METRICS_LIST

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
import seaborn

def test(
  session,
  model_dict,
  time_steps,
  centralities,
  instance_loader,
  model_checkpoint_filename
  ):
#end    
    # Reset instance loader
    instance_loader.reset()
    n_problems = len(instance_loader.filenames)
    # Init accuracies vector
    accs = np.zeros(n_problems)
    # Iterate over all instances
    for i, (filename, instance) in enumerate(zip(instance_loader.filenames, instance_loader.get_instances(n_problems))):

        # Create (size 1) batch from instance
        batch = create_batch([instance])

        # Build the feed_dict
        feed_dict = build_feed_dict(model_dict,batch,centralities,time_steps)

        # Build metrics dictionary for logging
        metrics_dict = build_metrics_dict(centralities)

        # Fetch overall accuracy
        accs[i] = session.run(model_dict['accuracy'], feed_dict=feed_dict)
    #end

    # Return average accuracy
    return np.mean(accs)
#end

if __name__ == '__main__':

    # Add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', default='centrality-betweenness-closeness-degree-eigenvector', help='Which trained model to load')
    parser.add_argument('-savepath', default='stats/test-varying-sizes.dat.log', help='Where to save the test results')
    parser.add_argument('-dataset_size', default=256, help='Number of instances to generate for each problem size for each graph distribution')
    parser.add_argument('-timesteps', default=32, help='Number of timesteps to run the model')
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

    logging_file = open(args['savepath'], 'w')
    model_checkpoint_filename = args['model']

    # Init list of problem sizes
    problem_sizes = np.arange(32,240+1,16)

    # Create datasets
    for n in problem_sizes:

        # Test datasets
        if not os.path.isdir('test-varying-sizes/n={}/erdos'.format(n)):
            # Create random Erdos Renyi test dataset
            os.makedirs('test-varying-sizes/n={}/erdos'.format(n), exist_ok=True)
            create_dataset(
              instances     = args['dataset_size'],
              min_n        = n,
              max_n        = n,
              path         = 'test-varying-sizes/n={}/erdos'.format(n),
              graph_type   = 'erdos_renyi',
              graph_args   = [0.4],
              graph_kwargs = dict()
            )
        #end
        if not os.path.isdir('test-varying-sizes/n={}/pltree'.format(n)):
            # Create random Powerlaw Tree test dataset
            os.makedirs('test-varying-sizes/n={}/pltree'.format(n), exist_ok=True)
            create_dataset(
              instances     = args['dataset_size'],
              min_n        = n,
              max_n        = n,
              path         = 'test-varying-sizes/n={}/pltree'.format(n),
              graph_type   = 'powerlaw_tree',
              graph_args   = [3],
              graph_kwargs = dict()
            )
        #end
        if not os.path.isdir('test-varying-sizes/n={}/smallworld'.format(n)):
            # Create random Watts Strogatz Smallworld test dataset
            os.makedirs('test-varying-sizes/n={}/smallworld'.format(n), exist_ok=True)
            create_dataset(
              instances     = args['dataset_size'],
              min_n        = n,
              max_n        = n,
              path         = 'test-varying-sizes/n={}/smallworld'.format(n),
              graph_type   = 'watts_strogatz',
              graph_args   = [4,0.15],
              graph_kwargs = dict()
            )
        #end
        if not os.path.isdir('test-varying-sizes/n={}/plcluster'.format(n)):
            # Create random Powerlaw Tree test dataset
            os.makedirs('test-varying-sizes/n={}/plcluster'.format(n), exist_ok=True)
            create_dataset(
              instances     = args['dataset_size'],
              min_n        = n,
              max_n        = n,
              path         = 'test-varying-sizes/n={}/plcluster'.format(n),
              graph_type   = 'powerlaw_cluster',
              graph_args   = [3,0.15],
              graph_kwargs = dict()
            )
        #end
    #end

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

        # Init accuracies vector
        accuracies = np.zeros(len(problem_sizes))

        for i,n in enumerate(problem_sizes):
            # Init instance loader for this problem size
            loader = InstanceLoader('test-varying-sizes/n={}'.format(n), rng=random.Random(0))
            # Fetch accuracy
            accuracies[i] = test(
                sess,
                GNN,
                time_steps,
                centralities,
                loader,
                model_checkpoint_filename
                )
            print('\t(n,t)=({n},{t})\t|\tacc={acc}'.format(n=n,t=time_steps,acc=accuracies[i]))
        #end

        #accuracies = np.array([0.8961069583892822,0.899834846495650,0.898047089576721,0.894547195232007,0.890805192117113,0.878895857778843,0.853585287928581,0.824788305151742,0.808806534216273,0.793779184401501,0.7836012911866419,0.7785228241700679])

        fig, ax = plt.subplots(1, 1, figsize=(5,4))
        ax.set_xlabel('Problem size')
        ax.set_ylabel('Accuracy (%)')
        #ax.set_ylim(75,95)
        plt.axvline(x=32, c='black', linestyle='--')
        plt.axvline(x=128, c='black', linestyle='--')
        plt.xticks(problem_sizes)
        #ax.fill_betweenx(problem_sizes, 32, 128, alpha=0.1, facecolor='#cccccc')

        ax.plot(problem_sizes, 100*accuracies, linestyle='-', marker='o')
        plt.savefig('figures/test-varying-sizes.eps', format='eps', dpi=200)
    #end
#end
