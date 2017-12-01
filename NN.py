import math 
import sys
import random
import re

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


class NeuralNetwork:
	def __init__(self, depth, width, trainf, testf):
		self.trials = 10 # make custom param later
		self.lnRate = 0.1

		# network 
		self.depth = depth
		self.width = width

		# reading file input
		self.train_input = []
		self.correct_train_output = []
		self.test_input = []
		self.correct_test_output = []

		# network
		self.node_count = 0
		self.output_nodes = 0
		self._initialize(trainf, testf)
		self.network = Network(self.node_count, self.depth, self.width)


	def learn(self):
		self._networkInit()
		for it in range(self.trials):
			for i in range(len(self.train_input)):
				self._scoreInit(self.train_input[i])
				self._forward()
				theta = self._backward(i)
				self._update(theta)

			train_err = self._trainErrRate()
			print train_err

			for i in range(len(self.test_input)):
				err = self._errRate(self.test_input[i], self.correct_test_output[i])
				# if err:
				# 	print "miss"
				# else:
				# 	print "hit"

			test_err = self._testErrRate()
			print test_err

		# print self.node_count, self.output_nodes, len(self.network.node_scores)

	def _errRate(self, e, label):
		self._scoreInit(e)
		self._forward()

		idx = self.network.output_layer_idx
		scores = [self.network.score(j) for j in range(idx, idx+self.output_nodes)]

		if self._decode(label) != scores.index(max(scores))+1:
			return 1
		return 0


	def _trainErrRate(self):
		miss = 0.0
		for i in range(len(self.train_input)):
			L = self.correct_train_output[i]
			miss += self._errRate(self.train_input[i], L)
		return miss/len(self.train_input)

	def _testErrRate(self):
		miss = 0.0
		for i in range(len(self.test_input)):
			L = self.correct_train_output[i]
			miss += self._errRate(self.test_input[i], L)
		return miss/len(self.test_input)


	# update weights with theata
	def _update(self, theta):
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
		n = self.lnRate
		node_val = self.network.score(idx)
		output_edges = self.network.get_output_edges(idx)

		for e in output_edges.keys():
			new_weight = output_edges[e] - theta[e]*n*node_val
			self.network.add_output_edge(idx, e, new_weight)
			self.network.add_input_edge(e, idx, new_weight)


	# backpropogation
	# fix this 
	def _backward(self, example_index):
		L = self.correct_train_output[example_index]
		idx = self.network.output_layer_idx
		i = 0
		theta = {}

		# output layer
		for idx in range(idx, idx+self.output_nodes):
			score = self.network.score(idx)
			sig_p = score*(1-score)
			theta[idx] = -sig_p*(L[i]-score)
			i+=1

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
				idx += 1

		# output layer
		for i in range(self.output_nodes):
			input_edges = self.network.get_input_edges(idx)
			score = sum([self.network.score(node) * input_edges[node] for node in input_edges.keys()])
			self.network.set_score(idx, self._sigmoid(score))
			idx += 1


	# set score with the current example
	def _scoreInit(self, example):
		for i in range(self.node_count):
			self.network.set_score(i, example[i])

	# init and network and weights of the edges
	def _networkInit(self):
		random.seed(0)
		# set input layer scores
		idx = 0
		for idx in range(self.node_count):
			self.network.set_output_edges(idx, {})

		# set inner layers score and edges weight
		for d in range(self.depth):
			curr_layer = idx + 1
			if d == 0:
				for w in range(self.width):
					idx += 1
					edges = {}
					for n in range(self.node_count):
						weight = random.uniform(-.1,.1)
						self.network.add_output_edge(n, idx, weight)
						edges[n] = weight
					self.network.set_input_edges(idx, edges)
					self.network.set_score(idx, 0)
					self.network.set_output_edges(idx, {})

			else:
				for w in range(self.width):
					idx += 1
					edges = {}
					for n in range(self.width):
						weight = random.uniform(-.1, .1)
						self.network.add_output_edge(curr_layer - self.width + n, idx, weight)
						edges[curr_layer - self.width + n] = weight
					self.network.set_input_edges(idx, edges)
					self.network.set_score(idx, 0)
					self.network.set_output_edges(idx, {})

		# output layer
		idx += 1
		curr_layer = idx
		for idx in range(idx, idx+self.output_nodes):
			edges = {}
			for n in range(self.width):
				weight = random.uniform(-.1, .1)
				self.network.add_output_edge(curr_layer - self.width + n, idx, weight)
				edges[curr_layer - self.width + n] = weight
			self.network.set_input_edges(idx, edges)
			self.network.set_score(idx, 0)



	def _sigmoid(self, a):
		return 1.0/(1+ math.pow(math.e, -a))
	

	# read input and store train examples to train_input []
	# store test examples to test_input []
	def _initialize(self, trainf, testf):
		with open(trainf, 'r') as train_data:
			for line in train_data:
				if '@' not in line and len(line)>1:
					line = map(int, line.split(','))
					self.correct_train_output.append(self._encode(line[-1]))
					self.train_input.append(line[:-1])
				elif '@' in line and 'class' in line:
					self.output_nodes = len([int(x) for x in re.findall(r'\d+', line)])
			self.node_count = len(self.train_input[0])

		if testf:
			with open(testf, 'r') as test_data:
				for line in test_data:
					if '@' not in line and len(line)>1:
						line = map(int, line.split(','))
						self.correct_test_output.append(self._encode(line[-1]))
						self.test_input.append(line[:-1])

	def _encode(self, num):
		idx = num - 1
		arr = []
		for i in range(self.output_nodes):
			if i == idx:
				arr.append(1)
			else:
				arr.append(0)
		return arr

	def _decode(self, arr):
		for i in range(len(arr)):
			if arr[i] == 1:
				return i+1

if __name__ == '__main__':
	if sys.argv[1] == '-d' and sys.argv[3] == '-w' and len(sys.argv) >= 6:
		try:
			depth = int(sys.argv[2])
			width = int(sys.argv[4])
			trainf = sys.argv[5]
			if len(sys.argv) == 7:
				testf = sys.argv[6]
			elif len(sys.argv) == 6:
				testf = None

		except:
			print "Error: Invalid input!"

		finally:
			NN = NeuralNetwork(depth, width, trainf, testf)
			NN.learn()	
	
	else:
		print "Usage: python model.py -d [depth] -w [width] [train_file] [test_file]"
	