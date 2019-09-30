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
  time_steps = [32,64,96,128]#[2,12,22,32]
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

  instance_loader = InstanceLoader( "instances-pca", rng = instance_loader_random_generator )

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
    for instance in instance_loader.get_batches( 1 ):
      print( "Building feed dict ..." )
      feed_dict = {}
      feed_dict[ GNN["gnn"].matrix_placeholders["M"] ] = util.sparse_to_dense( instance["matrix"] )
      feed_dict[ GNN["nodes_n"] ]                      = instance["problem_n"]
      for centrality in centralities:
        clabels  = "{c}_labels" .format( c = centrality )
        ccompare = "{c}_compare".format( c = centrality )
        feed_dict[ GNN[clabels] ]                      = util.sparse_to_dense( instance[ccompare] )
      #end for

      print( instance["fnames"][0].split("/")[-1].split(".")[0] )
      
      ncols =  len( time_steps )
      for centrality in centralities:
        print( centrality )
        fig, ax = plt.subplots( nrows = 1, ncols = ncols, sharey = "all", figsize=(ncols*2, 2.5) )
        axes = list( ax.flat )
        cmin = min( filter( lambda x: x>0, instance[centrality][0] ) ) # Filter zeros to avoid problems with log scale
        cmax = max( filter( lambda x: x>0,  instance[centrality][0] ) )
        for i, axis in enumerate( axes ):
          if i+1 == ncols//2:
           axis.set_xlabel( "Normalized PCA value".format( c = centrality ) )
          #end if
          axis.set_xlim( 0, 1 )
          num_xticks = 5
          axis.set_xticks( np.linspace( 0, 1, num_xticks ) )  
          axis.set_xticklabels( "{:.1f}".format(x) if i==0 or i==2 or i==4 else None for i, x in enumerate( list( np.linspace( 0, 1, num_xticks ) ) ) )
          if i == 0:
            axis.set_ylabel( "log scale {c} value".format( c = centrality ) )
          else:
            axis.set_yticklabels( [] )
          #end if
          axis.set_yscale( "log" )
          axis.set_ylim( cmin, cmax )
          #end if
        #end for
        if centrality == "betweenness":
          color = "#E85D75"
        elif centrality == "closeness":
          color = "#729EA1"
        elif centrality == "degree":
          color = "#B5BD89"
        elif centrality == "eigenvector":
          color = "#F7EF81"
        else:
          color = "#8A4F7D"
        scatterplots = [
          axis.scatter(
            np.zeros( instance["problem_n"][0] ),
            np.zeros( instance["problem_n"][0] ),
            c = color,
            edgecolor= "#000000"#"#8A4F7D"
          ) for axis in list( axes )
        ]
        
        for i, t, scatterplot, axis in zip( range(len(time_steps)), time_steps, scatterplots, axes ):
        
          feed_dict[ GNN["gnn"].time_steps ] = t

          print( "Timestep {t:02d}, ".format( t = t ), end = "" )
          embeddings, acc = sess.run(
            [ GNN["gnn"].last_states["N"].h, GNN["{c}_accuracy".format(c=centrality)] ],
            feed_dict = feed_dict
          )
          print( "accuracy {acc:04.2f}".format( acc = acc * 100 ) )

          pca = PCA( n_components = 1 ).fit_transform( embeddings )
          pca[:,0] = (pca[:,0]-min(pca[:,0])) / (max(pca[:,0])-min(pca[:,0]))

          scatterplot.set_offsets( [ (x,y) for x,y in zip( pca, instance[centrality][0] ) ] )
          if True:#i%2==1:
            axis.set_title(
              "t {t:02d}, acc {a:04.2f}".format(
                t = t,
                a = acc * 100
              )
            )
          #end if
        
          fig.savefig(
            "pca-1d-{m}-{c}-{g}.eps".format(
              m = "".join( map( lambda x: x[0], centralities ) ),
              c = centrality,
              g = instance["fnames"][0].split("/")[-1].split(".")[0]
            ),
            bbox_inches = "tight"
          )
          plt.close(fig)
        #end for
    #end for instances
  #end session
#EOF
