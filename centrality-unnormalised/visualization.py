
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import random
import itertools
import os
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.animation import FuncAnimation

from model import model_builder
from util import load_weights, sparse_to_dense
from instance_loader import InstanceLoader, create_batch

def get_embeddings(sess, GNN, batch, time_steps):
    """
        Givan an input batch, return a list of lists of n-dimensional vertex
        embeddings produced by the GNN in 'time_steps' timesteps
    """

    # Build the feed_dict
    feed_dict = {}
    feed_dict[ GNN["gnn"].matrix_placeholders["M"] ] = sparse_to_dense( batch["matrix"] )
    feed_dict[ GNN["gnn"].time_steps ] = time_steps
    feed_dict[ GNN["nodes_n"] ] = batch["problem_n"]

    # Fetch all embeddings in the batch
    all_embeddings = sess.run(GNN['gnn'].last_states["N"].h, feed_dict=feed_dict)

    # Organize embeddings into a list of lists, each corresponding to a problem in the batch
    embeddings = [ all_embeddings[sum(batch['problem_n'][0:i]):sum(batch['problem_n'][0:i]+batch['problem_n'])] for i in range(len(batch['problem_n'])) ]

    return embeddings
#end

def get_acc(sess, GNN, batch, time_steps):
    """
        Givan an input batch, return the accuracy of each instance
    """
    labels = {
      centrality: sparse_to_dense( batch["{c}_compare".format(c=centrality)] )
      for centrality in sorted([ "betweenness","closeness","degree","eigenvector" ])
    }

    # Build the feed_dict
    feed_dict = {
      GNN["{c}_labels".format(c=centrality)]: labels[centrality]
      for centrality in sorted([ "betweenness","closeness","degree","eigenvector" ])
    }
    feed_dict[ GNN["gnn"].matrix_placeholders["M"] ] = sparse_to_dense( batch["matrix"] )
    feed_dict[ GNN["gnn"].time_steps ] = time_steps
    feed_dict[ GNN["nodes_n"] ] = batch["problem_n"]

    # Get accuracy
    acc = sess.run(GNN['accuracy'], feed_dict=feed_dict)

    return acc
#end

def get_accs(sess, GNN, batch, time_steps, centralities):
    """
        Givan an input batch, return the accuracy of each instance
    """
    labels = {
      centrality: sparse_to_dense( batch["{c}_compare".format(c=centrality)] )
      for centrality in centralities
    }

    # Build the feed_dict
    feed_dict = {
      GNN["{c}_labels".format(c=centrality)]: labels[centrality]
      for centrality in sorted([ "betweenness","closeness","degree","eigenvector" ])
    }
    feed_dict[ GNN["gnn"].matrix_placeholders["M"] ] = sparse_to_dense( batch["matrix"] )
    feed_dict[ GNN["gnn"].time_steps ] = time_steps
    feed_dict[ GNN["nodes_n"] ] = batch["problem_n"]

    # Get accuracies
    accs = sess.run([ GNN['{c}_accuracy'.format(c=c)] for c in centralities ], feed_dict=feed_dict)

    return { centrality:acc for centrality,acc in zip(centralities,accs) }
#end

def get_projections(embeddings, k):
    # Given a list of n-dimensional vertex embeddings, project them into k-dimensional space

    # Standarize dataset onto the unit scale (mean = 0, variance = 1)
    embeddings = StandardScaler().fit_transform(embeddings)

    # Get principal components
    principal_components = PCA(n_components=k).fit_transform(embeddings)

    return principal_components
#end

def get_rank(centrality):
    """
        Compute a rank given a list of centrality values ('centrality')
    """
    return [ i for (i,x) in sorted(enumerate(np.array(centrality).argsort()[::-1]), key=lambda x: x[1])]
#end

def show_projections(embeddings, centrality):
    """
        Given a batch of embeddings, plot their projections, coloring each
        datapoint according to its centrality (as given by 'centrality')
    """

    # Obtain projections
    projections = get_projections(embeddings, 2)

    # Normalize centralities
    #centrality = np.array(centrality)
    #centrality = (centrality-min(centrality))/(max(centrality)-min(centrality))
    #print(centrality)

    plt.scatter(projections[:,0], projections[:,1], edgecolors='black', c=get_rank(centrality), cmap='plasma')

    plt.show()
#end

def plot_1D_projections_through_time(savepath, sess, GNN, instance, tmin, tmax, step, centrality='eigenvector'):

    # Obtain centralities
    _,degree,_,betweenness,_,closeness,_,eigenvector,_,_ = instance

    centralities = {
        'degree': degree,
        'betweenness': betweenness,
        'closeness': closeness,
        'eigenvector': eigenvector
    }

    # Create a batch (size=1) from instance
    batch = create_batch([instance])

    timesteps = list(range(tmin,tmax+1,step))

    f, axes = plt.subplots(1, len(timesteps), dpi=200, sharex=True, sharey=True, figsize=(4*2, 2.5))

    # Iterate over timesteps
    for i,(t,ax) in enumerate(zip(timesteps,axes)):
        
        # Fetch embeddings
        embeddings = get_embeddings(sess, GNN, batch, t)[0]

        # Fetch accuracy
        acc = 100*get_accs(sess,GNN,batch,t,centralities)[centrality]

        # Obtain 2D projections
        projections = get_projections(embeddings, 1)
        
        # Write x and y axis labels
        if i == (len(timesteps)-1)//2:
            ax.set_xlabel('1D PCA')
        #end
        if i==0:
            ax.set_ylabel('{centrality} Centrality'.format(centrality=centrality.capitalize()))
        #end

        # Set subplot title
        ax.set_title('{t} steps\n{acc:.0f}% acc'.format(t=t,acc=acc))

        # Plot subplot
        ax.scatter(projections[:,0], centralities[centrality], edgecolors='black', c=get_rank(centralities[centrality]), cmap='jet', alpha=0.5, linewidths=0.1)#, s=10)
    #end

    # Set aspect ratio to yield square subplots
    x0,x1 = axes[0].get_xlim()
    y0,y1 = axes[0].get_ylim()
    for ax in axes:
        ax.set_aspect((x1-x0)/(y1-y0))
    #end

    plt.tight_layout()
    plt.savefig(savepath, format='eps', dpi=200)
#end

def plot_1D_pair_projections_through_time(savepath, sess, GNN, instance, tmin, tmax, step, centrality='eigenvector'):

    # Obtain centralities
    _,degree,_,betweenness,_,closeness,_,eigenvector,_,_ = instance

    centralities = {
        'degree': degree,
        'betweenness': betweenness,
        'closeness': closeness,
        'eigenvector': eigenvector
    }

    # Create a batch (size=1) from instance
    batch = create_batch([instance])

    timesteps = list(range(tmin,tmax+1,step))

    f, axes = plt.subplots(1, len(timesteps), dpi=200, sharex=True, sharey=True, figsize=(4*2, 2.5))

    # Iterate over timesteps
    for i,(t,ax) in enumerate(zip(timesteps,axes)):
        
        # Fetch embeddings
        embeddings = get_embeddings(sess, GNN, batch, t)[0]

        # Fetch accuracy
        acc = 100*get_accs(sess,GNN,batch,t,centralities)[centrality]

        # Compute rank
        rank = get_rank(centralities[centrality])

        # Compute 'pair embeddings', 'diff_centrality' and 'diff_rank'
        n,d = embeddings.shape
        pair_embeddings = np.zeros((n**2,2*d))
        diff_centrality = np.zeros(n**2)
        diff_rank       = np.zeros(n**2)
        for v1 in range(n):
            for v2 in range(n):
                pair_embeddings[n*v1+v2, :d] = embeddings[v1,:]
                pair_embeddings[n*v1+v2, d:] = embeddings[v2,:]
                diff_centrality[n*v1+v2]     = centralities[centrality][v1]-centralities[centrality][v2]
                diff_rank[n*v1+v2]           = rank[v1]-rank[v2]
            #end
        #end
        pair_embeddings = pair_embeddings[:len(pair_embeddings)//2]
        diff_centrality = diff_centrality[:len(diff_centrality)//2]
        diff_rank       = diff_rank[:len(diff_rank)//2]

        # Obtain 2D projections
        projections = get_projections(pair_embeddings, 1)
        # Normalize projections
        #projections[:,0] = (projections[:,0]-min(projections[:,0]))/(max(projections[:,0])-min(projections[:,0]))
        
        # Write x and y axis labels
        if i == (len(timesteps)-1)//2:
            ax.set_xlabel('1D PCA')
        #end
        if i==0:
            ax.set_ylabel('{centrality} Centrality delta'.format(centrality=centrality.capitalize()))
        #end

        # Set subplot title
        ax.set_title('{t} steps\n{acc:.0f}% acc'.format(t=t,acc=acc))

        # Plot subplot
        ax.scatter(projections[:,0], diff_centrality, edgecolors='black', c=diff_rank, cmap='jet', alpha=0.5, linewidths=0.1)#, s=10)
    #end

    # Set aspect ratio to yield square subplots
    x0,x1 = axes[0].get_xlim()
    y0,y1 = axes[0].get_ylim()
    for ax in axes:
        ax.set_aspect((x1-x0)/(y1-y0))
    #end

    plt.tight_layout()
    plt.savefig(savepath, format='eps', dpi=200)
#end

def plot_2D_projections_through_time(savepath, sess, GNN, instance, tmin, tmax, step):

    # Obtain centralities
    M,degree,_,betweenness,_,closeness,_,eigenvector,_,_ = instance
    edges = [ (i,j) for (i,j) in M[0] if i < j ]

    centrality = eigenvector

    # Create a batch (size=1) from instance
    batch = create_batch([instance])

    timesteps = list(range(tmin,tmax+1,step))

    f, axes = plt.subplots(1, len(timesteps), sharex=True, sharey=True, dpi=200, figsize=(4*2, 2.5))

    # Iterate over timesteps
    for i,(t,ax) in enumerate(zip(timesteps,axes)):
        
        # Fetch embeddings
        embeddings = get_embeddings(sess, GNN, batch, t)[0]

        # Fetch accuracy
        acc = 100*get_acc(sess,GNN,batch,t)

        # Obtain 2D projections
        projections = get_projections(embeddings, 2)

        # Set plot parameters
        ax.set_title('{t} steps\n{acc:.0f}% acc'.format(t=t,acc=acc))
        ax.set_aspect(adjustable='box-forced', aspect='equal')
        ax.tick_params(axis='both', which='both', left=False, bottom=False, labelbottom=False, labelleft=False)

        # Plot edges
        for i,j in edges:
            ax.plot(projections[[i,j],0], projections[[i,j],1], '-', c='black', linewidth=0.25, zorder=1)
        #end
        
        # Plot vertices
        ax.scatter(projections[:,0], projections[:,1], edgecolors='black', c=get_rank(centrality), cmap='jet', alpha=0.5, linewidths=0.1, zorder=2)#, s=10)
    #end

    plt.savefig(savepath, format='eps', dpi=200)
#end

def make_gif_1D(savepath, sess, GNN, instance, tmin=1, tmax=32, step=1):

    # Obtain centralities
    _,degree,_,betweenness,_,closeness,_,eigenvector,_,_ = instance

    # Create a batch (size=1) from instance
    batch = create_batch([instance])

    timesteps = list(range(tmin,tmax+1,step))

    # Get number of centralities
    nc = len(centralities)
    # Compute number of rows and columns in the mosaic of PCA animations (one
    # per centrality)
    ncols = 1 if nc == 1 else 2
    nrows = nc // ncols
    # Create figure
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex='all', sharey='none')
    flat_axes = list(axes.flat) if nc > 1 else [axes]

    # Iterate over centralities
    for centrality, axis, i in zip(centralities, flat_axes, range(1,1+nc)):
        # Compute minimum and maximum centrality values, filtering out zeros
        # to avoid problems with log-scale
        cmin = min([ x for x in batch[centrality][0] if x>0 ])
        cmax = min([ x for x in batch[centrality][0] if x>0 ])

        if i >= nc-1: axis.set_xlabel('Normalized PCA value');

        axis.set_xlim(0,1)
        axis.set_xticks( np.linspace(0,1,11) )
        axis.set_ylabel('log {c} value'.format(c=centrality))
        axis.set_yscale('log')
        axis.set_ylim(cmin,cmax)
        if i % 2 == 0:
            axis.yaxis.tick_right()
            axis.yaxis.set_label_position('right')
        #end
    #end

    scatterplots = [
        axis.scatter(
            np.zeros(batch['problem_n'][0]),
            np.zeros(batch['problem_n'][0]),
            c = 'w',
            edgecolors = 'b'
        ) for axis in flat_axes
    ]

    best_cacc = { c:0.0 for c in centralities }
    best_acc = [0.0]

    def update(i):

        # Get number of timesteps
        t = timesteps[i]

        # Fetch embeddings
        embeddings = get_embeddings(sess, GNN, batch, t)[0]

        # Fetch accuracy
        accs = 100*get_accs(sess,GNN,batch,t,centralities)

        # Obtain 1D PCA projections
        projections = PCA(n_components=1).fit_transform(embeddings)
        projections[:,0] = (projections[:,0]-min(projections[:,0]))/(max(projections[:,0]))-min(projections[:,0])
        
        # Iterate over centralities to update mosaic
        for centrality, scatterplot, axis, cacc in zip(centralities, scatterplots, flat_axes, accs):
            best_cacc[centrality] = max(best_cacc[centrality], cacc)
            scatterplot.set_offsets([ (x,y) for x,y in zip(projections, batch[centrality][0])])
            axis.set_title(
                '{c}, t {t:03d}, acc {a:04.2f}{star}'.format(
                    c = centrality,
                    t = t,
                    a = cacc * 100,
                    star = '*' if cacc == best_cacc[centrality] else ' '
                    )
                )
        #end

        drawables = [s for s in scatterplots] + [a for a in flat_axes]
        return drawables
    #end

    animation = FuncAnimation(figure, update, frames=range(len(timesteps)), interval=200)
    animation.save(savepath, writer="imagemagick" )
#end

def make_gif_2D(savepath, sess, GNN, instance, tmin=1, tmax=32, step=1):

    # Obtain centralities
    _,degree,_,betweenness,_,closeness,_,eigenvector,_,_ = instance

    centrality = eigenvector

    # Create a batch (size=1) from instance
    batch = create_batch([instance])

    timesteps = list(range(tmin,tmax+1,step))

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    
    # Plot a scatter that persists (isn't redrawn) and the initial line.
    n = batch['problem_n'][0]
    
    scatter = ax.scatter(np.zeros(n),np.zeros(n), edgecolors='black', c=get_rank(centrality), cmap='jet', linewidths=0.5, zorder=2)
    
    ax.set_xlim(-0.1,1.1), ax.set_xticks( [] )
    ax.set_ylim(-0.1,1.1), ax.set_yticks( [] )

    def update(i):

        # Get number of timesteps
        t = timesteps[i]

        # Fetch embeddings
        embeddings = get_embeddings(sess, GNN, batch, t)[0]

        # Fetch accuracy
        acc = 100*get_acc(sess,GNN,batch,t)

        # Obtain 2D projections
        projections = get_projections(embeddings, 2)
        projections[:,0] = (projections[:,0] - min(projections[:,0])) / (max(projections[:,0]) - min(projections[:,0]))
        projections[:,1] = (projections[:,1] - min(projections[:,1])) / (max(projections[:,1]) - min(projections[:,1]))

        ax.set_xlabel('Timestep {t}'.format(t=t))

        scatter.set_offsets( list(zip(projections[:,0],projections[:,1])) )
        return ax, scatter
    #end

    anim = FuncAnimation(fig, update, frames=np.arange(len(timesteps)), interval=200)
    anim.save(savepath, writer='imagemagick')
#end

if __name__ == '__main__':

    # Define parameters
    embedding_size = 64
    centralities = sorted([ "betweenness","closeness","degree","eigenvector" ])
    time_steps = 32
    batch_size = 32
    batches_per_epoch = 32
    random_seed = 412

    plt.rcParams.update({'figure.max_open_warning': 0})

    # Instantiate model
    GNN = model_builder( embedding_size, centralities )

    # Define test instance loader
    test_instance_loader_random_generator = random.Random( random_seed )
    test_instance_loader = InstanceLoader("./test-instances-small", rng = test_instance_loader_random_generator )

    # Disallow GPU use
    config = tf.ConfigProto(
      device_count =  {"GPU": 0 },
      inter_op_parallelism_threads = 1,
      intra_op_parallelism_threads = 1
    )
    with tf.Session(config=config) as sess:
        
        # Initialize global variables
        print('Initializing global variables ...')
        sess.run( tf.global_variables_initializer() )

        # Restore saved weights
        print('Restoring saved weights ...')
        load_weights(sess,'centrality-betweenness-closeness-degree-eigenvector')

        # Reset loader
        test_instance_loader.reset()
        n_problems = len(test_instance_loader.filenames)
        # Iterate over all test problems
        for i, (filename, instance) in enumerate(zip(test_instance_loader.filenames, test_instance_loader.get_instances(n_problems))):

            # Get number of vertices for this instance
            n = instance[0][2][0]

            # Get graph distribution name and instance index
            graphtype = filename.split('/')[2]
            index = filename.split('/')[3].split('.')[0]
            
            # Create directories, if inexistent
            for figure_type in ['figures-1D','figures-1D-pair','figures-2D','gifs-1D','gifs-2D']:
                for centrality in centralities:
                    if figure_type == 'figures-1D' or figure_type == 'figures-1D-pair':
                        folder_path = 'figures/{figure_type}/{centrality}/{graphtype}'.format(figure_type=figure_type,centrality=centrality,graphtype=graphtype)
                    else:
                        folder_path = 'figures/{figure_type}/{graphtype}'.format(figure_type=figure_type,centrality=centrality,graphtype=graphtype)
                    #end
                    
                    if not os.path.exists(folder_path): os.makedirs(folder_path);
                #end
            #end

            for centrality in centralities:
                
                # Plot 1D PCA projections through time
                plot_1D_projections_through_time(
                    'figures/figures-1D/{centrality}/{graphtype}/{index}.eps'.format(
                        centrality=centrality,
                        graphtype=graphtype,
                        index=index
                    ),
                    sess,GNN,instance,2,32,10,
                    centrality=centrality
                )

                # Plot 1D PCA pair projections through time
                plot_1D_pair_projections_through_time(
                    'figures/figures-1D-pair/{centrality}/{graphtype}/{index}.eps'.format(
                        centrality=centrality,
                        graphtype=graphtype,
                        index=index
                    ),
                    sess,GNN,instance,2,32,10,
                    centrality=centrality
                )
            #end

            # Plot 2D PCA projections through time
            plot_2D_projections_through_time(
                'figures/figures-2D/{graphtype}/{index}.eps'.format(
                    centrality=centrality,
                    graphtype=graphtype,
                    index=index
                ),
                sess,GNN,instance,2,32,10,
            )

            if i % (n_problems//10) == 0:
                print('{comp}% complete'.format(comp=np.round(100*i/n_problems)))
            #end
        #end
    
    #end
#end