"""
IO bound data recovery model for arbitrary data schemes
"""

import sys, random
from collections import defaultdict
from multiprocessing import Pool, cpu_count

class RecoveryGraph:
	"""
	The class representing cluster with N nodes, C chunks with n blocks each.
	Before running disk faults simulation the cluster should be populated with
	chunk states and the failure transitions between them.
	"""
	def __init__(self, N, C, n):
		assert n < N
		self.N = N
		self.C = C
		self.n = n
		self.states = defaultdict(list)
		self.chunks = dict()
		self.faults = defaultdict(list)
		self.degraded_states = []
		self.failed_states = []
		self.max_level = -1

	def __str__(self):
		return '%d disks, %d chunks, n=%d' % (self.N, self.C, self.n)

	def add_state(self, id, lvl = None, recovery_grp = 0):
		"""
		Add chunk state to the graph given the degradation level. The level 0 is for the root,
		level None is for the failed state. The recovery_grp is the number of blocks necessary
		to recover from this state (not used for the root and failed state)
		"""
		self.states[lvl].append((id, recovery_grp))
		self.chunks[id] = float(self.C) if lvl == 0 else 0.
		if lvl > 0:
			self.degraded_states.append(id)
		if lvl > self.max_level:
			self.max_level = lvl
		if lvl is None:
			self.failed_states.append(id)

	def add_fault(self, from_state_id, to_state_id, weight = 1.):
		"""Add disk fault transition from one chunk state to another"""
		self.faults[from_state_id].append((to_state_id, weight))

	def has_degraded_chunks(self):
		return sum(map(lambda id: self.chunks[id], self.degraded_states))

	def recover_chunks(self, time):
		"""
		Recover degraded chunks assuming the unit chunk recovery rate
		given the amount of time to spend
		"""
		root_id, _ = self.states[0][0]
		for lvl in range(self.max_level, 0, -1):
			for id, recovery_grp in self.states[lvl]:
				rate = self.N / (recovery_grp + 1.)
				chunks = self.chunks[id]
				if not chunks:
					continue
				recoved_chunks = time * rate
				if recoved_chunks > chunks:
					recoved_chunks = chunks
				self.chunks[id] -= recoved_chunks
				self.chunks[root_id] += recoved_chunks
				if self.chunks[id] > 0:
					return
				time -= recoved_chunks / rate
				if time <= 0:
					return

	def disk_fault(self):
		"""
		Simulate disk fault. Note for simplicity we always assume that
		the number of failed disks << N
		"""
		for lvl in range(self.max_level, -1, -1):
			for id, _ in self.states[lvl]:
				chunks = self.chunks[id]
				if not chunks:
					continue
				for to_state_id, weight in self.faults[id]:
					faulted_chunks = chunks * weight * (self.n - lvl) / self.N
					self.chunks[to_state_id] += faulted_chunks
					self.chunks[id] -= faulted_chunks

	def is_data_lost(self):
		return sum(map(lambda id: self.chunks[id], self.failed_states))

	def verify(self):
		"""Debug checks"""
		assert len(self.states[0]) == 1
		assert self.failed_states
		lvl_map = dict()
		for lvl in range(self.max_level, 0, -1):
			for id, recovery_grp in self.states[lvl]:
				assert 0 < recovery_grp <= self.n
				assert self.faults[id]
				lvl_map[id] = lvl
				total_weight = 1.
				for to_state_id, weight in self.faults[id]:
					assert to_state_id in self.failed_states or lvl_map[to_state_id] == lvl + 1
					assert weight > 0
					total_weight -= weight
				assert abs(total_weight) < 1e-6

	def simulate_disk_faults(self, fault_rate):
		"""Run simulation till data lost"""
		self.verify()
		simulation_time = 0.
		while not self.is_data_lost():
			elapsed_time = random.expovariate(fault_rate)
			if self.has_degraded_chunks():
				self.recover_chunks(elapsed_time)
			self.disk_fault()
			simulation_time += elapsed_time
		return simulation_time

def nk_recovery_graph(N, C, n, k):
	"""Build recovery graph for (n,k) scheme"""
	assert 0 < k <= n
	g = RecoveryGraph(N, C, n)
	for l in range (n - k + 1):
		g.add_state(l, l, k)
	g.add_state(n - k + 1) # failed state
	for l in range (n - k + 1):
		g.add_fault(l, l + 1)
	return g

def simulate_once((cluster, fault_rate)):
	"""Run simulation given the cluster and fault rate"""
	time = cluster.simulate_disk_faults(fault_rate)
	print >> sys.stderr, time
	return time

def simulate_disk_faults(rounds, cluster, fault_rate):
	"""Run simulation in all available CPU cores"""
	p = Pool(max(1, cpu_count()-1))
	times = p.imap_unordered(simulate_once, [(cluster, fault_rate)]*rounds)
	p.close()
	p.join()
	return sum(times)/float(rounds)

if __name__ == '__main__':
	N, C, n = 50, 2500, 5
	c = nk_recovery_graph(N, C, n, 3)
	failure_rate = 0.003
	print c, failure_rate, simulate_disk_faults(100, c, failure_rate)

