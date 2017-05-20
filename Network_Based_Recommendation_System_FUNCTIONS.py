import csv
import pprint as pp
import networkx as nx
import itertools as it
import math
import scipy.sparse
import random




def pagerank(M, N, nodelist, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-6, dangling=None):
	if N == 0:
		return {}
	S = scipy.array(M.sum(axis=1)).flatten()
	S[S != 0] = 1.0 / S[S != 0]
	Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
	M = Q * M
	
	# initial vector
	x = scipy.repeat(1.0 / N, N)
	
	# Personalization vector
	if personalization is None:
		p = scipy.repeat(1.0 / N, N)
	else:
		missing = set(nodelist) - set(personalization)
		if missing:
			#raise NetworkXError('Personalization vector dictionary must have a value for every node. Missing nodes %s' % missing)
			print
			print 'Error: personalization vector dictionary must have a value for every node'
			print
			exit(-1)
		p = scipy.array([personalization[n] for n in nodelist], dtype=float)
		#p = p / p.sum()
		sum_of_all_components = p.sum()
		if sum_of_all_components > 1.001 or sum_of_all_components < 0.999:
			print
			print "Error: the personalization vector does not represent a probability distribution :("
			print
			exit(-1)
	
	# Dangling nodes
	if dangling is None:
		dangling_weights = p
	else:
		missing = set(nodelist) - set(dangling)
		if missing:
			#raise NetworkXError('Dangling node dictionary must have a value for every node. Missing nodes %s' % missing)
			print
			print 'Error: dangling node dictionary must have a value for every node.'
			print
			exit(-1)
		# Convert the dangling dictionary into an array in nodelist order
		dangling_weights = scipy.array([dangling[n] for n in nodelist], dtype=float)
		dangling_weights /= dangling_weights.sum()
	is_dangling = scipy.where(S == 0)[0]

	# power iteration: make up to max_iter iterations
	for _ in range(max_iter):
		xlast = x
		x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p
		# check convergence, l1 norm
		err = scipy.absolute(x - xlast).sum()
		if err < N * tol:
			return dict(zip(nodelist, map(float, x)))
	#raise NetworkXError('power iteration failed to converge in %d iterations.' % max_iter)
	print
	print 'Error: power iteration failed to converge in '+str(max_iter)+' iterations.'
	print
	exit(-1)




def create_graph_set_of_users_set_of_items(user_item_ranking_file):
	graph_users_items = {}
	all_users_id = set()
	all_items_id = set()
	g = nx.DiGraph()
	input_file = open(user_item_ranking_file, 'r')
	input_file_csv_reader = csv.reader(input_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
	for line in input_file_csv_reader:
		user_id = int(line[0])
		item_id = int(line[1])
		rating = int(line[2])
		g.add_edge(user_id, item_id, weight=rating)
		all_users_id.add(user_id)
		all_items_id.add(item_id)
	input_file.close()
	graph_users_items['graph'] = g
	graph_users_items['users'] = all_users_id
	graph_users_items['items'] = all_items_id
	return graph_users_items
	












def create_item_item_graph(graph_users_items):
	g = nx.Graph()
	# Your code here ;)
	# Separate into the bivariate graph, users and items
	bg = graph_users_items["graph"]
	users = graph_users_items["users"]
	items = graph_users_items["items"]

	# Create a dict where the key is a user and
	# the value is a list of the watched movies by the user
	saw = {user: [] for user in users}
	for edge in bg.edges():
		saw[edge[0]].append(edge[1])

	# Create the tuples that represent the edges of the nodes and a weight 0:
	t = []
	for user in saw:
		for node1 in saw[user]:
			for node2 in saw[user]:
				if node1 < node2:
					t.append((node1,node2))

	import collections
	# Get the weight
	counter = collections.Counter(t)
	# Create tuples like (node1, node2, weight)
	tw = []
	for edge in counter.keys():
		tw.append((edge[0], edge[1], counter[edge]))

	# Create the graph
	g.add_weighted_edges_from(tw)
	return g




def create_preference_vector_for_teleporting(user_id, graph_users_items):
	preference_vector = {}
	# Your code here ;)

	# Get the total weight of the user aka denominator of the formula
	total_weight = 0
	for edge in graph_users_items["graph"].edges(user_id, data=True):
		total_weight += edge[2]["weight"]

	# Create the preference vector with the non zero edges
	for edge in graph_users_items["graph"].edges(user_id, data=True):
		preference_vector[edge[1]] = float(edge[2]["weight"])/float(total_weight)

	# Fill with 0 the items that are not conected to the user
	for item in graph_users_items["items"]:
		if item not in preference_vector:
			preference_vector[item] = float(0.)
	return preference_vector
	



def create_ranked_list_of_recommended_items(page_rank_vector_of_items, user_id, training_graph_users_items):
	# This is a list of 'item_id' sorted in descending order of score.
	sorted_list_of_recommended_items = []
	# You can obtain this list from a list of [item, score] couples sorted in descending order of score.
	# Your code here ;)

	# Items rated by the user
	items = []
	for edge in training_graph_users_items["graph"].edges():
		if edge[0] == user_id:
			items.append(edge[1])

	# Put the items that are not rated by the user in the list without sorting
	for item in page_rank_vector_of_items:
		if item not in items:
			sorted_list_of_recommended_items.append([item, page_rank_vector_of_items[item]])

	# Order it by score descending
	sorted_list_of_recommended_items = sorted(sorted_list_of_recommended_items, key=lambda k: k[1], reverse=True)
	
	return sorted_list_of_recommended_items




def discounted_cumulative_gain(user_id, sorted_list_of_recommended_items, test_graph_users_items):
	dcg = 0.
	# Your code here ;)
	
	
	return dcg
	



def maximum_discounted_cumulative_gain(user_id, test_graph_users_items):
	dcg = 0.
	# Your code here ;)
	
	
	return dcg













