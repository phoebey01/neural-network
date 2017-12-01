import math 
import sys
import random

class Network:
	def __init__(self, input_layer_size=0, depth=0, width=0):
		self.node_count = input_layer_size
		self.node_scores = {}
		self.input_edges = {}
		self.output_edges = {}
		self.output_layer_idx = input_layer_size + depth * width

	def score(self, index):
		return self.node_scores[index]

	def get_input_edges(self, index):
		return self.input_edges[index]

	def get_output_edges(self, index):
		return self.output_edges[index]

	@property
	def output_layer_idx(self):
		return self.output_layer_idx

	def set_score(self, index, score):
		self.node_scores[index] = score

	def set_input_edges(self, index, edges):
		self.input_edges[index] = edges

	def set_output_edges(self, index, edges):
		self.output_edges[index] = edges

	def add_input_edge(self, index, from_index, edge):
		self.input_edges[index][from_index] = edge

	def add_output_edge(self, index, to_index, edge):
		self.output_edges[index][to_index] = edge


	# remove later
	def print_all(self):
		print self.node_scores
		print self.input_edges
		print self.output_edges

class NeuralNetwork:
	def __init__(self, depth, width, arr):

		# network 
		self.depth = depth
		self.width = width
		self.input = arr

		# network
		self.node_count = 2
		self.network = Network(2, self.depth, self.width)


	def learn(self):
		sig = self._sigmoid(3)
		print sig, sig*(1-sig)
		# self._networkInit()
		# self._initWeight()
		# self._scoreInit(self.input)
		# self._forward()
		# print self.network.node_scores
		# theta = self._backward(0)
		# self._update(theta)

		# self.network.print_all()
		# print theta



	# update weights with theata
	def _update(self, theta):
		n = 0.1
		# input layer
		for i in range(self.node_count):
			self._updatefml(theta, i)

		# inner layers
		idx = self.node_count
		for i in range(self.depth):
			for idx in range(idx, idx+self.width):
				self._updatefml(theta, idx)
			idx += 1


	def _updatefml(self, theta, idx):
		n = 0.1
		node_val = self.network.score(idx)
		output_edges = self.network.get_output_edges(idx)

		for e in output_edges.keys():
			new_weight = output_edges[e] - theta[e]*n*node_val
			self.network.add_output_edge(idx, e, new_weight)
			self.network.add_input_edge(e, idx, new_weight)


	# backpropogation
	def _backward(self, example_index):
		L = 0
		idx = self.network.output_layer_idx
		theta = {}
		# output layer
		for idx in range(idx, idx+self.node_count):
			score = self.network.score(idx)
			sig_p = score*(1-score)
			theta[idx] = -sig_p*(L-score)

		# inner layer
		idx = self.network.output_layer_idx - self.width
		for i in range(self.depth):
			for idx in range(idx, idx+self.width):
				theta[idx] = self._backpropfml(idx, theta)			
			idx -= 2 * self.width - 1

		return theta

	def _backpropfml(self, idx, theta):
		score = self.network.score(idx)
		sig_p = score*(1-score)
		output_edges = self.network.get_output_edges(idx)
		th = sig_p * sum([output_edges[e] * theta[e] for e in output_edges.keys()])
		return th


	# calculate scores with current weight
	def _forward(self):
		# inner layers
		idx = self.node_count
		for i in range(self.depth):
			for j in range(self.width):
				input_edges = self.network.get_input_edges(idx)
				score = sum([self.network.score(node) * input_edges[node] for node in input_edges.keys()])
				self.network.set_score(idx, self._sigmoid(score))
				# print idx, score
				idx += 1

		# output layer
		for i in range(self.node_count):
			input_edges = self.network.get_input_edges(idx)
			score = sum([self.network.score(node) * input_edges[node] for node in input_edges.keys()])
			self.network.set_score(idx, self._sigmoid(score))
			# print idx, score
			idx += 1


	def _initWeight(self):
		input_edges = {2: {0:1, 1:1}, 3:{0:1, 1:1}, 4: {2:.6, 3:.6}, 5:{2:.6, 3:.6}, 6:{4:1,5:1},7:{4:1,5:1}}
		self.network.input_edges = input_edges
		output_edges = {0: {2:1, 3:1}, 1:{2:1,3:1}, 2:{4:.6,5:.6},3:{4:.6,5:.6}, 4:{6:1,7:1}, 5:{6:1,7:1}}
		self.network.output_edges = output_edges

	# set score with the current example
	def _scoreInit(self, example):
		for i in range(self.node_count):
			self.network.set_score(i, example[i])

	def _sigmoid(self, a):
		return 1.0/(1+ math.pow(math.e, -a))


if __name__ == '__main__':
	NN = NeuralNetwork(2, 2, [2,3])
	NN.learn()
	