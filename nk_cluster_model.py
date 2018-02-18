"""
The fast and simple cluster model with (n,k) storage scheme.
"""

import sys, random
from multiprocessing import Pool, cpu_count

class NKCluster:
	"""The class representing cluster with N nodes, C chunks using (n,k) scheme"""
	def __init__(self, N, C, n, k):
		assert n < N
		assert 0 < k <= n
		self.N = N
		self.C = C
		self.n = n
		self.k = k
		# keep one list per every degradation level including the fault level
		self.chunks = [0.]*(self.n-self.k+2)
		self.chunks[0] = float(self.C)

	def __str__(self):
		return '%d disks, %d chunks, (%d, %d)' % (self.N, self.C, self.n, self.k)

	def chunks_degraded(self, lvl, cnt):
		"""Called when chunks at the particular degradation level becomes corrupted due to the disk fault"""
		self.chunks[lvl+1] += cnt
		self.chunks[lvl] -= cnt
	
	def disk_fault(self):
		"""
		Simulate disk fault. Note for simplicity we always assume that
		the number of failed disks << N
		"""
		redundancy = self.n - self.k
		# move chunks between fault levels
		for i in range(redundancy, -1, -1):
			chunks = self.chunks[i]
			if not chunks:
				continue
			self.chunks_degraded(i, chunks * (self.n - i) / self.N)

	def has_degraded_chunks(self):
		return sum(self.chunks[1:]) > 0

	def recover_chunks(self, time):
		"""
		Recover degraded chunks assuming the unit chunk recovery rate
		given the amount of time to spend
		"""
		rate = self.N / (self.k + 1.)
		for i in range(self.n-self.k+1, 0, -1):
			chunks = self.chunks[i]
			if not chunks:
				continue
			recoved_chunks = time * rate
			if recoved_chunks > chunks:
				recoved_chunks = chunks
			self.chunks[i] -= recoved_chunks
			self.chunks[i-1] += recoved_chunks
			if self.chunks[i] > 0:
				return
			time -= recoved_chunks / rate
			if time <= 0:
				return

	def is_data_lost(self):
		"""Checks if we've already lost the data"""
		return self.chunks[-1]

	def time_elapsed(self, elapsed_time):
		"""Simulation time elapsed handler"""
		if self.has_degraded_chunks():
			self.recover_chunks(elapsed_time)

	def simulate_first_lost(self, fault_rate):
		"""Run simulation till data lost"""
		simulation_time = 0.
		while not self.is_data_lost():
			elapsed_time = random.expovariate(fault_rate)
			self.time_elapsed(elapsed_time)
			self.disk_fault()
			simulation_time += elapsed_time
		return simulation_time

	def simulate_multiple_lost(self, fault_rate, cycles=1):
		"""Run simulation. """
		self.simulate_first_lost(fault_rate)
		cycle, simulation_time = 0, 0.
		while cycle < cycles:
			healphy = False
			while True:
				elapsed_time = random.expovariate(fault_rate)
				self.time_elapsed(elapsed_time)
				self.disk_fault()
				if not self.is_data_lost():
					healphy = True
				elif healphy:
					break
				simulation_time += elapsed_time
			cycle += 1
		return simulation_time / cycles

def simulate_once((cluster, fault_rate, first_lost)):
	"""Run simulation given the cluster and fault rate"""
	if first_lost:
		time = cluster.simulate_first_lost(fault_rate)
	else:
		time = cluster.simulate_multiple_lost(fault_rate)
	print >> sys.stderr, time
	return time

def simulate_disk_faults(rounds, cluster, fault_rate, first_lost=True):
	"""Run simulation in all available CPU cores"""
	p = Pool(max(1, cpu_count()-1))
	times = p.imap_unordered(simulate_once, [(cluster, fault_rate, first_lost)]*rounds)
	p.close()
	p.join()
	return sum(times)/float(rounds)

if __name__ == '__main__':
	from mttdl import *
	N, C, n = 50, 2500, 5
	c = NKCluster(N, C, n, 3)
	failure_rate = 0.002
	rounds = 100
	print c, failure_rate, nk2mttdl(N, C, n, failure_rate),\
		simulate_disk_faults(rounds, c, failure_rate, first_lost=True),\
		simulate_disk_faults(rounds, c, failure_rate, first_lost=False)




