import pandas
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import sys

centralities = [ "betweenness","closeness","degree","eigenvector" ]

fname = "centrality-" + "-".join( centralities )
sepfnames = [ "centrality-" + centrality for centrality in centralities ]

dataframe_transfer = pandas.read_csv( fname + ".batch.log", sep = "\t" )
dataframe_transfer["training_step"] = dataframe_transfer["epoch_id"] * 32 + dataframe_transfer["batch_id"]
dataframes_separated = [ pandas.read_csv( fname + ".batch.log", sep = "\t" ) for fname in sepfnames ]
for i in range( len(dataframes_separated ) ):
  dataframes_separated[i]["training_step"] = dataframes_separated[i]["epoch_id"] * 32 + dataframes_separated[i]["batch_id"]
#end for

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


nrows = 2
ncols = 2
if len( sys.argv ) >= 3:
  nrows = int( sys.argv[1] )
  ncols = int( sys.argv[2] )
elif len( sys.argv ) >= 2:
  nrows = int( sys.argv[1] )
  ncols = 1
#end if
assert ( ( nrows * ncols ) // len( centralities ) ) <= 1

sx = ncols*4 + (1 if ncols == 1 else 0)
sy = nrows*2

figure, axes = plt.subplots(nrows=nrows,ncols=ncols,sharex="all",sharey="all")
figure.set_size_inches( sx, sy )
taxes = np.ndarray( axes.shape, dtype = np.object )

flat_axes = axes.flat if ncols > 1 else axes
flat_taxes = taxes.flat if ncols > 1 else taxes

for i in range(len(flat_axes)):
  flat_taxes[i] = flat_axes[i].twinx()
#end for

for c, df, axis, taxis in zip( centralities, dataframes_separated, flat_axes, flat_taxes ):
  taxis.plot(
    df["training_step"],
    df[c + "_cost"],
    c = colors[0]
  )
  taxis.plot(
    dataframe_transfer["training_step"],
    dataframe_transfer[c + "_cost"],
    c = colors[1]
  )
  taxis.set_ylim( [0.1,0.8] )
  
  axis.plot(
    df["training_step"],
    df[c + "_ACC"],
    c = colors[2]
  )
  axis.plot(
    dataframe_transfer["training_step"],
    dataframe_transfer[c + "_ACC"],
    c = colors[3]
  )
  axis.set_ylim( [0,1] )
  axis.set_title( c )
#end

if ncols == 1:
  for i in range( nrows ):
    flat_axes[i].set_ylabel( "Accuracy" )
    flat_taxes[i].set_ylabel( "Loss" )
  #end for
  flat_axes[nrows-1].set_xlabel( "Training steps" )
elif nrows == 1:
  flat_axes[0].set_ylabel( "Accuracy" )
  flat_taxes[ncols-1].set_ylabel( "Loss" )
  for i in range( ncols-1 ):
    flat_taxes[i].set_yticklabels( [] )
    flat_axes[i].set_xlabel( "Training steps" )
  #end for
  flat_axes[ncols-1].set_xlabel( "Training steps" )
else:
  for j in range(ncols):
    print( nrows-1, j )
    axes[nrows-1,j].set_xlabel( "Training steps" )
    for i in range(nrows-1):
      taxes[i,j].set_yticklabels( [] )
    #end for
  #end for
  for i in range(nrows):
    axes[i,0].set_ylabel( "Accuracy" )
    taxes[i,ncols-1].set_ylabel( "Loss" )
  #end
#end if
figure.savefig(
  "training_{nrows}x{ncols}_{sx}x{sy}.eps".format(
    nrows = nrows,
    ncols = ncols,
    sx = sx,
    sy = sy
  )
)
plt.show()
