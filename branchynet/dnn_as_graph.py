import networkx as nx, sys
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import Counter
import config


class ModelDnnGraph(object):

	def __init__(self, processing_time_edge, processing_time_cloud, output_size, avg_bandwidth, nr_samples_early_exits, cloud_edge_factor=None):
	""" This class is used to initialize DNNs and BranchyNet (DNNs with early exits) as a graphs
	"""
		
		self.processing_time_cloud = processing_time_cloud.mean()
		self.cloud_edge_factor = cloud_edge_factor
		self.output_size = output_size
		self.avg_bandwidth = avg_bandwidth
		self.nr_samples_early_exits = nr_samples_early_exits

		if(cloud_edge_factor is None):
			self.processing_time_edge = processing_time_edge.mean()

		else:
			self.processing_time_edge = cloud_edge_factor*(processing_time_cloud.mean())

		_, self.nr_layers_backbone = self.count_layers()

		self.nr_early_exits = len(config.BRANCHES_POSITIONS)
		self.branches_positions = config.BRANCHES_POSITIONS

		self.branchynet = self.__generate_graph()

	def count_layers(self):
		"""
		Counts the number of early exits and the backbone layers.
		"""
		count_layers = Counter(layer[0] for layer in list(self.processing_time_edge.index))
		nr_early_exits, nr_layers_backbone = count_layers["b"], count_layers["l"]
		return nr_early_exits, nr_layers_backbone

	def __generate_graph(self):
		"""
		This converts DNN architecture into a line graph.
		"""
		vertices = []
		branchynet = nx.DiGraph()

		for backbone_layer in range(1, self.nr_layers_backbone+1):
			branchynet.add_node("l_%s(e)"%(backbone_layer), group="backbone", device="edge")
			branchynet.add_node("l_%s(aux)"%(backbone_layer), group="aux", device="edge")
			branchynet.add_node("l_%s(c)"%(backbone_layer), group="backbone", device="cloud")

		#for early_exit in range(1, self.nr_early_exits+1):
		#	branchynet.add_node("b_%s"%(early_exit), group="side_branch", device="edge")

		for early_exit in config.BRANCHES_POSITIONS:
			branchynet.add_node("b_%s"%(early_exit), group="side_branch", device="edge")
		
		branchynet_edges = []
		for backbone_layer in range(1, self.nr_layers_backbone):
			branchynet_edges.append(("l_%s(c)"%(backbone_layer), "l_%s(c)"%(backbone_layer+1)))
			branchynet_edges.append(("l_%s(e)"%(backbone_layer), "l_%s(aux)"%(backbone_layer)))
			branchynet_edges.append(("l_%s(aux)"%(backbone_layer), "l_%s(e)"%(backbone_layer+1)))
			branchynet_edges.append(("l_%s(aux)"%(backbone_layer), "l_%s(c)"%(backbone_layer+1)))

		#for early_exit in range(1, self.nr_early_exits+1):
		#	branchynet_edges.append(("l_%s(aux)"%(early_exit), "b_%s"%(early_exit)))
		#	branchynet_edges.append(("l_%s(aux)"%(early_exit), "l_%s(c)"%(early_exit+1)))
		#	branchynet_edges.append(("b_%s"%(early_exit), "l_%s(e)"%(early_exit+1)))


		for early_exit in config.BRANCHES_POSITIONS:
			branchynet_edges.append(("l_%s(aux)"%(early_exit), "b_%s"%(early_exit)))
			branchynet_edges.append(("l_%s(aux)"%(early_exit), "l_%s(c)"%(early_exit+1)))
			branchynet_edges.append(("b_%s"%(early_exit), "l_%s(e)"%(early_exit+1)))



		branchynet.add_edges_from(branchynet_edges)

		return branchynet


	def model_dnn_as_shortest_path_graph(self, branchynet_graph, bandwidth):

		"""
		This model the line graph into a shortest path graph in order applying dijkstra algorithm.
		"""

		prob_dict = self.compute_probabilities(branchynet_graph)		

		self.communication_time = self.output_size/bandwidth


		dt = 10**(-9)

		branchynet_graph.add_nodes_from(["input", "output"], group='s-t', device='both')

		branchynet_graph.add_edges_from([("input", "l_1(e)", {"weight":0}),
			("input", "l_1(c)", {"weight": self.communication_time["input"].values[0]})])

		branchynet_graph.add_edges_from([("l_%s(c)"%(self.nr_layers_backbone), "output", 
			{"weight": dt + prob_dict["l_%s(c)"%(self.nr_layers_backbone)]*self.processing_time_cloud["l_%s"%(self.nr_layers_backbone)]}),
			("l_%s(e)"%(self.nr_layers_backbone), "output", 
				{"weight": prob_dict["l_%s(c)"%(self.nr_layers_backbone)]*self.processing_time_edge["l_%s"%(self.nr_layers_backbone)]})])

		groups = nx.get_node_attributes(branchynet_graph, "group")
		devices = nx.get_node_attributes(branchynet_graph, "device")


		for node_id, node_att in dict(branchynet_graph.nodes.data()).items():

			layer_id = node_id.split("(")[0] if node_id[0]=="l" else node_id 
			group_origin = node_att['group']
			device_origin = node_att['device']

			node_neightboor_dict = dict(branchynet_graph[node_id])

			for node_neightboor_id, weight in node_neightboor_dict.items():
				group_dest = groups[node_neightboor_id]
				device_dest = devices[node_neightboor_id]
				
				if ((device_origin == "cloud") and (device_dest == "cloud")):
					branchynet_graph[node_id][node_neightboor_id]['weight'] = prob_dict[node_id]*self.processing_time_cloud[layer_id]
					
				if ((device_origin == "edge" and group_origin=="backbone") and (group_dest == "aux")):
					branchynet_graph[node_id][node_neightboor_id]['weight'] = prob_dict[node_id]*self.processing_time_edge[layer_id]

				if ((group_origin=="aux") and (device_dest=="side_branch")):
					branchynet_graph[node_id][node_neightboor_id]['weight'] = 0

				if ((group_origin=="aux") and (group_dest=="backbone" and device_dest=="edge")):
					branchynet_graph[node_id][node_neightboor_id]['weight'] = 0


				if ((group_origin=="side_branch") and (device_dest == "edge" and group_dest=="backbone")):
					branchynet_graph[node_id][node_neightboor_id]['weight'] = prob_dict[node_id]*self.processing_time_edge[layer_id]

				if((group_origin=="aux") and (device_dest == "cloud")):
					branchynet_graph[node_id][node_neightboor_id]['weight'] = prob_dict[node_id]*self.communication_time[layer_id].values[0]


		return branchynet_graph

	
	def partitioningDecision(self):

		shortest_path_graph = self.model_dnn_as_shortest_path_graph(self.branchynet, self.avg_bandwidth)

		inference_time, shortest_path = nx.single_source_dijkstra(shortest_path_graph,
			"input", "output", weight='weight')

		partitioning_layer = self.find_partitioning_layer(shortest_path, shortest_path_graph)

		return partitioning_layer


	def find_partitioning_layer(self, shortest_path, shortest_path_graph):

		devices = nx.get_node_attributes(shortest_path_graph, "device")

		edge_path = []
		for i, node in enumerate(shortest_path):
			dev_layer = devices[node]

			if (dev_layer == "cloud"):
				break
			else:
				edge_path.append(node)

		last_layer_edge = edge_path[-1]
		if (last_layer_edge == "input" or last_layer_edge == "output"):
			partitioning_layer = last_layer_edge

		else:
			partitioning_layer = int(last_layer_edge.split("(aux)")[0].split("l_")[-1])
	
		return partitioning_layer



	def compute_probabilities(self, branchynet_graph):

		pop = np.sum(self.nr_samples_early_exits)
		prob_branches = []
		prob_0 = 1
		cont = 0
		prob_dict = {}
		branches_positions = self.branches_positions
		branches_positions.append(self.nr_layers_backbone)

		for nr_exit in self.nr_samples_early_exits:
			prob_branches.append(nr_exit/pop)
			pop = pop - nr_exit

		for lid in range(1, self.nr_layers_backbone + 1):
			if (lid <= branches_positions[0]):
				prob_dict.update({"l_%s(e)"%(lid): prob_0, "l_%s(c)"%(lid): prob_0, "l_%s(aux)"%(lid):prob_0})

				if (lid == branches_positions[0]):
					prob_dict.update({"b_%s"%(lid):prob_0})
					prob_0 *= 1-prob_branches[cont]
					cont +=1
					del branches_positions[0]

		return prob_dict

	#def compute_probabilities(self, branchynet_graph):
	#	pop = np.sum(self.nr_samples_early_exits)
	#	prob_branches = []
	#	cont = 0
	#	prob_dict = {layer: 1.0 for layer in list(branchynet_graph.nodes)}
	#	branches_positions = self.branches_positions

	#	for nr_exit in self.nr_samples_early_exits:
	#		prob_branches.append(nr_exit/pop)
	#		pop = pop - nr_exit


	#	for i, (branch_position, p) in enumerate(zip(branches_positions, prob_branches), 1):

	#		for l_id in range(1, self.nr_layers_backbone+1):
	#			if (l_id <= branches_position):
	#				prob_dict["l_%s(e)"] *= 1
	#				prob_dict["l_%s(c)"] *= 1
	#				prob_dict["l_%s(aux)"] *= 1

				