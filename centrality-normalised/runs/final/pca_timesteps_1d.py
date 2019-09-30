import random
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from instance_loader import InstanceLoader
from sklearn.decomposition import PCA
import itertools, sys, os, model, util

if __name__ == "__main__":
  embedding_size = 64
  time_steps = 32
  max_time_steps = 128
  centralities = sorted( sys.argv[1:] )

  if len( centralities ) <= 0:
    raise ValueError( "No centrality passed" )
  #end if
  for centrality in centralities:
    if centrality not in [ "betweenness", "closeness", "degree", "eigenvector" ]:
      raise ValueError( "Centrality {c} not one of the accepted ones".format( c = centrality ) )
    #end if
  #end for
  
  fname = "centrality-" + "-".join( centralities )

  # Build model
  print( "Building model" )
  GNN = model.model_builder( embedding_size, centralities )
  # Load instances
  random_seed = 1477
  instance_loader_random_generator = random.Random( random_seed )

  instance_loader = InstanceLoader( "test-instances", rng = instance_loader_random_generator )

  # Disallow GPU and multicore
  config = tf.ConfigProto(
    device_count = {"GPU": 0},
    intra_op_parallelism_threads = 1,
    inter_op_parallelism_threads = 1
  )
  with tf.Session(config=config) as sess:
    print( "Initializing global variables ... " )
    sess.run( tf.global_variables_initializer() )
    print( "Loading weights ... " )
    util.load_weights( sess, fname )
    print( "Loading an instance ..." )
    instance = instance_loader.get_batch( 1 )
    print( "Building feed dict ..." )
    feed_dict = {}
    feed_dict[ GNN["gnn"].matrix_placeholders["M"] ] = util.sparse_to_dense( instance["matrix"] )
    feed_dict[ GNN["nodes_n"] ]                      = instance["problem_n"]
    for centrality in centralities:
      clabels  = "{c}_labels" .format( c = centrality )
      ccompare = "{c}_compare".format( c = centrality )
      feed_dict[ GNN[clabels] ]                      = util.sparse_to_dense( instance[ccompare] )
    #end for

    print( "Visualizing the PCA of the embeddings, compared to the real centrality values,\n" +
           "varying the number of time_steps the networks runs from 1 to {tmax}".format(
             tmax = max_time_steps
           )
    )
    nc = len( centralities )
    ncols = 1 if nc == 1 else 2
    nrows = nc // ncols
    figure, axes = plt.subplots( nrows = nrows, ncols = ncols, sharex = "all", sharey = "none" )
    flat_axes = list( axes.flat ) if nc > 1 else [axes]
    
    for centrality, axis, i in zip( centralities, flat_axes, range( 1, 1 + nc ) ):
      cmin = min( filter( lambda x: x>0, instance[centrality][0] ) ) # Filter zeros to avoid problems with log-scale
      cmax = max( filter( lambda x: x>0,  instance[centrality][0] ) )
      if i >= nc - 1:
       axis.set_xlabel( "Normalized PCA value".format( c = centrality ) )
      #end if
      axis.set_xlim( 0, 1 )
      axis.set_xticks( np.linspace( 0, 1, 11 ) )
      axis.set_ylabel( "log {c} value".format( c = centrality ) )
      axis.set_yscale( "log" )
      axis.set_ylim( cmin, cmax )
      if i % 2 == 0:
        axis.yaxis.tick_right()
        axis.yaxis.set_label_position( "right" )
      #end if
    #end for

    scatterplots = [
      axis.scatter(
        np.zeros( instance["problem_n"][0] ),
        np.zeros( instance["problem_n"][0] ),
        c = 'w',
        edgecolor='b'
      ) for axis in flat_axes
    ]
    
    best_cacc = { c:0.0 for c in centralities }
    best_acc = [0.0]

    def pca_animation_function(t):
      feed_dict[ GNN["gnn"].time_steps ] = t

      print( "Timestep {t:03d}, ".format( t = t ), end = "" )
      embeddings, acc = sess.run(
        [ GNN["gnn"].last_states["N"].h, GNN["accuracy"] ],
        feed_dict = feed_dict
      )
      best_acc[0] = max( best_acc[0], acc )
      print( "accuracy {acc:04.2f}{star}".format( acc = acc * 100, star = "*" if acc == best_acc[0] else " " ) )
      centrality_accs = sess.run(
        [ GNN["{c}_accuracy".format(c=c)] for c in centralities ],
        feed_dict = feed_dict
      )

      pca = PCA( n_components = 1 ).fit_transform( embeddings )
      pca[:,0] = (pca[:,0]-min(pca[:,0])) / (max(pca[:,0])-min(pca[:,0]))

      for centrality, scatterplot, axis, cacc in zip( centralities, scatterplots, flat_axes, centrality_accs ):
        best_cacc[centrality] = max( best_cacc[centrality], cacc )
        scatterplot.set_offsets( [ (x,y) for x,y in zip( pca, instance[centrality][0] ) ] )
        axis.set_title(
          "{c}, t {t:03d}, acc {a:04.2f}{star}".format(
            c = centrality,
            t = t,
            a = cacc * 100,
            star = "*" if cacc == best_cacc[centrality] else " "
          )
        )
      #end for
      
      drawables = [s for s in scatterplots] + [a for a in flat_axes]
      return drawables #scatterplots, axes
    #end pca_animation_function

    animation = FuncAnimation( figure, pca_animation_function, frames = np.arange(0,max_time_steps), interval = 200 )
    animation.save( "{}-1d.gif".format( fname ), writer="imagemagick" )
  #end session
#EOF
