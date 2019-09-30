# Set random seeds for reproducibility
random_seed = 42#1237
import random
random.seed( random_seed )
import numpy as np
np.random.seed( random_seed )

import sys, os, json
import networkx as nx

def calc_degree( G, normalized = True ):
  # Calculates the degree centrality, non-normalized
  degree = nx.degree_centrality( G )
  if not normalized:
    degree = { node:value * ( G.number_of_nodes() - 1 ) for node, value in degree.items() }
  #end
  return degree
#end calc_degree

def calc_betweenness( G, normalized = True ):
  # Calculates the betweenness centrality, non-normalized
  return nx.betweenness_centrality( G, normalized = normalized )
#end calc_betweenness

def calc_closeness( G, normalized = True ):
  # Calculates the closeness centrality
  closeness = { node:nx.closeness_centrality( G, node ) for node in G }
  if not normalized:
    g_n = len( G )
    closeness = { node: value * ( G.number_of_nodes() - 1 ) / ( len( nx.node_connected_component( G, node ) ) - 1 ) if 0 < len( nx.node_connected_component( G, node ) ) else 0 for node,value in closeness.items() }
  #end if
  return closeness
#end calc_closeness

def calc_eigenvector( G ):
  # Calculates the eigenvector centrality
  return nx.eigenvector_centrality( G )
#end calc_eigenvector

def circular_shell_list(N, pin, pout):
  get_n = lambda r: int( np.round( 2 * np.pi * r ) )
  get_m = lambda last_n, n: int( np.round( n * n * pin ) + np.round( ns[-1] * n * pout ) )
  ns = [0]
  ms = [0]
  r = 0
  cum_n = sum(ns)
  while cum_n < N:
    r += 1
    n = get_n( r )
    if N < cum_n + n:
      break
    #end if
    m = get_m( ns[-1], n )
    cum_n += n
    ns.append( n )
    ms.append( m )
  #end while
  ns[-1] -= cum_n - N
  ms[-1] = get_m( ns[-2], ns[-1] )
  assert sum( ns ) == N
  return [ (n, m, pin/(pin+pout)) for n, m in zip( ns[1:], ms[1:] ) ]
#end circular_shell_list

def save_graph(
  G,
  name,
  path = "instances/erdos"
):
  """ Calculates the centrality values of every node of a graph and save it to disk """
  degree_normalized = calc_degree( G, normalized = True )
  degree_not_normalized = calc_degree( G, normalized = False )
  betweenness_normalized = calc_betweenness( G, normalized = True )
  betweenness_not_normalized = calc_betweenness( G, normalized = False )
  closeness_normalized = calc_closeness( G, normalized = True )
  closeness_not_normalized = calc_closeness( G, normalized = False )
  eigenvector = calc_eigenvector( G ) # May raise nx.exception.PowerIterationFailedConvergence
  
  for node in G:
    G.nodes[node]["d_n"] = degree_normalized[ node ]
    G.nodes[node]["d_p"] = degree_not_normalized[ node ]
    G.nodes[node]["b_n"] = betweenness_normalized[ node ]
    G.nodes[node]["b_p"] = betweenness_not_normalized[ node ]
    G.nodes[node]["c_n"] = closeness_normalized[ node ]
    G.nodes[node]["c_p"] = closeness_not_normalized[ node ]
    G.nodes[node]["e_n"] = eigenvector[ node ]
  #end for
  
  data = nx.readwrite.json_graph.node_link_data( G )
  s = json.dumps( data )
  with open(
    "{path}/{name}.g".format(
      path = path,
      name = name
    ),
    mode = "w"
    
  ) as f:
    f.write(s)
  #end with open f
#end save_graph

def create_dataset(
  instances,
  min_n        = 32,
  max_n        = 128,
  path         = "instances/erdos",
  graph_type   = "erdos_renyi",
  graph_args   = list(),
  graph_kwargs = dict()
):
  """ Creates a dataset with a certain number of instances, with the number of nodes sampled linearly from min_n to max_n (inclusive), from a certain distribution and save it to path. """
  i = 0
  while i < instances:
    g_n = np.random.randint( min_n, max_n + 1 )
    instance_graph_args = [ arg() if callable( arg ) else arg for arg in graph_args ]
    instance_graph_kwargs = { key: arg() if callable( arg ) else arg for key, arg in graph_kwargs.items() }
    
    if graph_type == "erdos_renyi":
      G = nx.fast_gnp_random_graph( g_n, *instance_graph_args, **instance_graph_kwargs )
    elif graph_type == "powerlaw_tree":
      try:
        G = nx.random_graphs.random_powerlaw_tree( g_n, *instance_graph_args, **instance_graph_kwargs )
      except nx.NetworkXError as e:
        print( e, file = sys.stderr, flush = True )
        continue
      #end try
    elif graph_type == "watts_strogatz":
      try:
        G = nx.random_graphs.connected_watts_strogatz_graph( g_n, *instance_graph_args, **instance_graph_kwargs )
      except nx.NetworkXError as e:
        print( e, file = sys.stderr, flush = True )
        continue
      #end try
    elif graph_type == "powerlaw_cluster":
      try:
        G = nx.random_graphs.powerlaw_cluster_graph( g_n, *instance_graph_args, **instance_graph_kwargs )
      except nx.NetworkXError as e:
        print( e, file = sys.stderr, flush = True )
        continue
      #end try
    elif graph_type == "circular_shell":
      G = nx.random_graphs.random_shell_graph( circular_shell_list( g_n, *instance_graph_args[:2] ), *instance_graph_args[2:], **instance_graph_kwargs )
    elif graph_type == "barabasi_albert":
      G = nx.generators.random_graphs.barabasi_albert_graph( g_n, *instance_graph_args, **instance_graph_kwargs )
    else:
      raise InvalidArgumentError( "Graph type not supported" )
    #end if
    
    if len( nx.node_connected_component( G, 0 ) ) != g_n:
      # No disjoint subgraphs allowed
      continue
    #end if
    
    try:
      save_graph( G, i, path = path )
    except nx.exception.PowerIterationFailedConvergence as e:
      # Failed to calculate eigenvector centrality, dump the graph and try again.
      continue
    #end try
    print( "{i}-th instance at \"{path}\" created.".format( i = i, path = path ), file = sys.stdout, flush = True )
    i += 1
  #end while
#end create_dataset

def parse( path ):
  #Parse graphs in edge-list format to our json format
  for filename in os.listdir( path ):
    name = filename[:-2]
    with open(
      "{path}/{name}.g".format(
        path = path,
        name = name
      ),
    mode = "rb"
    ) as f:
      G = nx.read_edgelist(f, comments="#", nodetype=int )
      subgraphs = list(nx.connected_component_subgraphs( G ))
      #print(len(subgraphs))
      #G.remove_nodes_from(nx.isolates(G))
      for i, g in enumerate( subgraphs ):
        #subgraphs smaller than 50 are discarded
        if( len(g.nodes()) > 50 ):
          #relabel the nodes so it remains consecutive
          g = nx.convert_node_labels_to_integers(g)
          save_graph( g, name+"_{i}".format(i = i), "test-realgraphs" ) 

#end parse

if __name__ == "__main__":

  n_instances_training = 4096
  min_n_training = 32
  max_n_training = 128

  n_instances_test = 4096
  min_n_test = min_n_training
  max_n_test = max_n_training
  
  n_instances_test_extreme = 64
  min_n_test_extreme = max_n_training
  max_n_test_extreme = 4*max_n_training
  
  n_instances_test_different = 256
  min_n_test_different = min_n_test
  max_n_test_different = max_n_test
  
  if (len(sys.argv) == 3 and sys.argv[1] == "parse"):
    path = sys.argv[2]
    assert os.path.isdir( path ), "Path is not a directory. Path {}".format( path )
    parse(path)
  else:

    graph_types = ['erdos_renyi','powerlaw_tree','watts_strogatz','powerlaw_cluster','circular_shell','barabasi_albert']
    graph_abbrv = { name:abbrv for name,abbrv in zip(graph_types,['erdos','pltree','smallworld','plcluster','cshell','barabasi']) }
    graph_args = { name:args for name,args in zip(graph_types, [ [0.25],[3],[4,0.25],[4,0.1],[0.25,0.1],[lambda:random.randint(2,5)] ]) }
    
    create_train = False
    create_test = False
    create_test_extreme = False
    create_test_different = True

    if create_train:
        # Create training datasets
        random_seed = 2312; random.seed(random_seed); np.random.seed(random_seed)
        for graph_type in graph_types[:-2]:
            os.makedirs('instances/{abbrv}'.format(abbrv=graph_abbrv[graph_type]), exist_ok=True)
            create_dataset(
                instances = n_instances_training,
                min_n = min_n_training,
                max_n = max_n_training,
                path = 'instances/{abbrv}'.format(abbrv=graph_abbrv[graph_type]),
                graph_type = graph_type,
                graph_args = graph_args[graph_type],
                graph_kwargs = dict()
            )
        #end
    #end

    if create_test:
        # Create test datasets
        random_seed = 32132; random.seed(random_seed); np.random.seed(random_seed)
        for graph_type in graph_types[:-2]:
            os.makedirs('test-instances/{abbrv}'.format(abbrv=graph_abbrv[graph_type]), exist_ok=True)
            create_dataset(
                instances = n_instances_test,
                min_n = min_n_test,
                max_n = max_n_test,
                path = 'test-instances/{abbrv}'.format(abbrv=graph_abbrv[graph_type]),
                graph_type = graph_type,
                graph_args = graph_args[graph_type],
                graph_kwargs = dict()
            )
        #end
    #end

    if create_test_extreme:
        # Create extreme test datasets
        random_seed = 4132; random.seed(random_seed); np.random.seed(random_seed)
        for graph_type in graph_types[:-2]:
            os.makedirs('test-instances-extreme/{abbrv}'.format(abbrv=graph_abbrv[graph_type]), exist_ok=True)
            create_dataset(
                instances = n_instances_test_extreme,
                min_n = min_n_test_extreme,
                max_n = max_n_test_extreme,
                path = 'test-instances-extreme/{abbrv}'.format(abbrv=graph_abbrv[graph_type]),
                graph_type = graph_type,
                graph_args = graph_args[graph_type],
                graph_kwargs = dict()
            )
        #end
    #end

    if create_test_different:
        # Create datasets with unseen distributions
        random_seed = 57681; random.seed(random_seed); np.random.seed(random_seed)
        for graph_type in graph_types[-2:]:
            os.makedirs('test-instances-different/{abbrv}'.format(abbrv=graph_abbrv[graph_type]), exist_ok=True)
            create_dataset(
                instances = n_instances_test_different,
                min_n = min_n_test_different,
                max_n = max_n_test_different,
                path = 'test-instances-different/{abbrv}'.format(abbrv=graph_abbrv[graph_type]),
                graph_type = graph_type,
                graph_args = graph_args[graph_type],
                graph_kwargs = dict()
            )
        #end
    #end
