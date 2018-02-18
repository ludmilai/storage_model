"""
Storage cluster simulator for testing
different allocation and replication strategies.
"""

import random
import heapq
from multiprocessing import Pool, cpu_count
from collections import namedtuple
import sys

min_replication_coverage = .7
max_replication_coverage = .9
max_replication_attempts = 50

def bitmap(iterable):
	"""Make bitmap from bit index list"""
	return reduce(lambda v, d: v|(1<<d), iterable, 0)

def hi_bit(bmap):
	"""Returns the most significant bit of bitmap"""
	return 1 << (bmap.bit_length() - 1)

def hi_bit_index(bmap):
	"""Get index of the most significant bit"""
	return bmap.bit_length() - 1

def lo_bit_index(bmap):
	"""Get index of the least significant bit"""
	b = (bmap - 1) & bmap
	if not b: # there are only one bit
		return bmap.bit_length() - 1
	# b contains all bits but the least significant one
	low_bit = bmap - b
	return low_bit.bit_length() - 1

def bit_count(bmap):
	"""Count non-zero bits in the bitmap"""
	cnt = 0
	while bmap:
		bmap -= 1 << (bmap.bit_length()-1)
		cnt += 1
	return cnt

def clear_hi_bits(bmap, bit_cnt):
	"""Clear given number of bits in the bitmap (starting from highest order bits)"""
	for _ in range(bit_cnt):
		bmap -= 1 << (bmap.bit_length()-1)
	return bmap

def select_hi_bits(bmap, bit_cnt):
	"""Select given number of bits in the bitmap (starting from highest order bits)"""
	cnt, bits = 0, 0
	for _ in range(bit_cnt):
		if not bmap:
			break
		bit = 1 << (bmap.bit_length()-1)
		bmap, bits, cnt = bmap - bit, bits | bit, cnt + 1
	return bits, cnt

class Encoding:
	"""The class representing the particular data encoding scheme"""
	def __init__(self, n):
		"""The only parameter here is the number of disk blocks per chunk"""
		self.n = n

	def is_data_lost(self, chunk_disk_bmap, failed_disks):
		"""Returns true if the data can not be recovered"""
		raise NotImplementedError

	def get_recovery_bitmap(self, chunk_disk_bmap, failed_disks):
		"""
		Returns the bitmap of the disk blocks required for recovery
		assuming that we are recovering the last element of failed_disks
		"""
		return chunk_disk_bmap - bitmap(failed_disks), self.n - len(failed_disks)

class NK(Encoding):
	"""Encoding with k blocks sufficient for recovery"""
	def __init__(self, n, k):
		Encoding.__init__(self, n)
		self.k = k

	def is_data_lost(self, chunk_disk_bmap, failed_disks):
		"""Returns true if the data can not be recovered"""
		return len(failed_disks) > self.n - self.k

	def get_recovery_bitmap(self, chunk_disk_bmap, failed_disks):
		"""
		Returns the bitmap of the disk blocks required for recovery
		assuming that we are recovering the last element of failed_disks
		"""
		good_disk_bmap = chunk_disk_bmap - bitmap(failed_disks)
		good_disk_count = self.n - len(failed_disks)
		excess = good_disk_count - self.k
		if excess:
			assert excess > 0
			# Remove excessive blocks starting from failed disk (just to randomize selection)
			r = failed_disks[-1]
			mask = (1 << (r + 1)) - 1
			bits, count = select_hi_bits(good_disk_bmap & mask, excess)
			good_disk_bmap -= bits
			if count < excess:
				bits, count2 = select_hi_bits(good_disk_bmap, excess - count)
				good_disk_bmap -= bits
				assert count + count2 == excess
		return good_disk_bmap, self.k

class LRC22(Encoding):
	"""Encoding with 2 local and 2 global parity"""
	def __init__(self, n):
		assert n % 2 == 0
		Encoding.__init__(self, n)

	def is_data_lost(self, chunk_disk_bmap, failed_disks):
		"""Returns true if the data can not be recovered"""
		if len(failed_disks) > 4:
			return True
		if len(failed_disks) <= 3:
			return False		
		# Consider 2 low order bits as global parity followed by 2 local groups l bits each
		# This is just approximation since we don't keep any block roles information so
		# the roles may be changed after recovery
		l = (self.n - 2) / 2 # local group size
		group1, _ = select_hi_bits(chunk_disk_bmap, l)
		group2, _ = select_hi_bits(chunk_disk_bmap - group1, l)
		p1_bit, p2_bit = hi_bit(group1), hi_bit(group2) # local parity bits
		e1, e2, p1, p2, pg = 0, 0, 1, 1, 2 # data errors and parity bits counters
		for d in failed_disks:
			d_bit = 1 << d
			if d_bit & group1:
				if d_bit == p1_bit:
					p1 = 0
				else:
					e1 += 1
			elif d_bit & group2:
				if d_bit == p2_bit:
					p2 = 0
				else:
					e2 += 1
			else:
				pg -= 1
		assert pg >= 0
		parities = pg # Number of useful parity bits
		if e1 and p1: parities += 1
		if e2 and p2: parities += 1
		# Return True if errors are theoretically unrecoverable
		return e1 + e2 > parities

	def get_recovery_bitmap(self, chunk_disk_bmap, failed_disks):
		"""
		Returns the bitmap of the disk blocks required for recovery
		assuming that we are recovering the last element of failed_disks
		"""
		l = (self.n - 2) / 2           # local group size
		failed = 1 << failed_disks[-1] # failed disk being recovered
		other = bitmap(failed_disks) - failed
		group1, _ = select_hi_bits(chunk_disk_bmap, l)
		if failed & group1 != 0 and other & group1 == 0:
			return group1 - failed, l - 1
		group2, _ = select_hi_bits(chunk_disk_bmap - group1, l)
		if failed & group2 != 0 and other & group2 == 0:
			return group2 - failed, l - 1
		# Recover from all remaining disks in any case more complex than local group recovery
		return chunk_disk_bmap - failed - other, self.n - len(failed_disks)

class RandomAllocator:
	"""Random chunk allocator"""
	@staticmethod
	def allocate(ndisks, nchunks, n):
		"""Allocate given number of chunks"""
		disk_list = range(ndisks)
		# The list of disks per every chunk
		return [random.sample(disk_list, n) for _ in range(nchunks)]

class PGAllocator:
	"""Placement groups based chunk allocator"""
	def __init__(self, ngroups):
		self.ngroups = ngroups

	def build_groups(self, ndisks, n):
		"""Build allocation groups with n disks each out of the total ndisks"""
		disk_list = range(ndisks)
		# The list of disks per every placement group
		groups = [None] * self.ngroups
		ch_start = ndisks
		for i in range(self.ngroups):
			ch_end = ch_start + n
			if ch_end > ndisks:
				random.shuffle(disk_list)
				ch_start, ch_end = 0, n
			groups[i] = disk_list[ch_start:ch_end]
			ch_start = ch_end
		return groups

	def allocate(self, ndisks, nchunks, n):
		"""Allocate given number of chunks"""
		groups = self.build_groups(ndisks, n)
		# The list of disks per every chunk
		return [groups[i % self.ngroups] for i in range(nchunks)]

class ReplicationScheduler:
	"""Replication scheduler base class"""
	def __init__(self, cluster):
		self.cluster = cluster
		self.tasks = None

	def is_idle(self):
		return not self.tasks

	def prepare(self, tasks):
		raise NotImplementedError

	def schedule(self):
		raise NotImplementedError

class RandomSingle(ReplicationScheduler):
	"""Schedule replication by randomly choosing single chunk"""
	def prepare(self, tasks):
		random.shuffle(tasks)
		self.tasks = tasks

	def schedule(self):
		t = self.tasks[-1]
		failed_disks, chunk = t
		new_d, _ = self.cluster.allocate_disk_for_recovery(chunk)
		if len(failed_disks) > 1:
			self.tasks[-1] = (failed_disks[:-1], chunk)
		else:
			self.tasks = self.tasks[:-1]
		self.cluster.on_chunks_replicated(chunk, failed_disks[-1], new_d)

class PrioritySingle(ReplicationScheduler):
	"""Schedule replication by choosing single chunk with maximum lost replica count"""
	def prepare(self, tasks):
		self.tasks = [
			(
				-len(failed_disks),
				failed_disks, chunk
			) for failed_disks, chunk in tasks
		]
		heapq.heapify(self.tasks)

	def schedule(self):
		t = self.tasks[0]
		failed_cnt, failed_disks, chunk = t
		new_d, _ = self.cluster.allocate_disk_for_recovery(chunk)
		if failed_cnt < -1:
			heapq.heapreplace(self.tasks, (failed_cnt + 1, failed_disks[:-1], chunk))
		else:
			heapq.heappop(self.tasks)
		self.cluster.on_chunks_replicated(chunk, failed_disks[-1], new_d)

class GroupReplicationScheduler(ReplicationScheduler):
	"""Group replication scheduler base class"""
	def prepare_scheduler(self):
		"""Prepare scheduling parameters"""
		self.min_busy_disks = min_replication_coverage * self.cluster.ndisks
		self.max_busy_disks = min(
				max_replication_coverage * self.cluster.ndisks,
				self.cluster.healthy_disks() - self.cluster.encoding.n
			)

	def prepare_tasks(self, repl_tasks):
		"""Preprocess replication tasks"""
		tasks = [
			(
				failed_disks, chunk,
				self.cluster.encoding.get_recovery_bitmap(self.cluster.chunks[chunk], failed_disks)
			) for failed_disks, chunk in repl_tasks
		]
		for failed_disks, _, _ in tasks:
			random.shuffle(failed_disks)
		random.shuffle(tasks)
		return tasks

	def schedule_tasks(self, tasks, busy_disks, busy_disks_cnt, next_tasks):
		"""
		Schedule tasks given the task list, busy disks bitmap and busy disks count.
		Returns the updated arguments in the same order.
		"""
		skip_cnt, done_tasks = 0, []
		for i, t in enumerate(tasks):
			if busy_disks_cnt > self.max_busy_disks:
				break

			failed_disks, chunk, (repl_bmap, repl_disks_cnt) = t
			if repl_bmap & busy_disks:
				skip_cnt += 1
				if busy_disks_cnt >= self.min_busy_disks and skip_cnt > max_replication_attempts:
					break
				continue

			new_d, new_d_bit = self.cluster.allocate_disk_for_recovery(chunk, busy_disks)
			busy_disks |= repl_bmap
			busy_disks |= new_d_bit
			busy_disks_cnt += repl_disks_cnt + 1

			self.cluster.on_chunks_replicated(chunk, failed_disks[-1], new_d)
			if len(failed_disks) > 1:
				failed_disks = failed_disks[:-1]
				next_tasks.append((failed_disks, chunk, self.cluster.encoding.get_recovery_bitmap(self.cluster.chunks[chunk], failed_disks)))

			done_tasks.append(i)

		# Drop completed tasks
		return self.drop_tasks(tasks, done_tasks), busy_disks, busy_disks_cnt

	@staticmethod
	def drop_tasks(tasks, indexes):
		"""Remove completed tasks from the list in linear time"""
		for i in reversed(indexes):
			if i < len(tasks) - 1:
				tasks[i] = tasks[-1]
			tasks = tasks[:-1]
		return tasks

class RandomGroup(GroupReplicationScheduler):
	"""Schedule replication by randomly choosing group of chunks"""
	def prepare(self, tasks):
		self.tasks = self.prepare_tasks(tasks)
		self.prepare_scheduler()

	def schedule(self):
		self.tasks, _, _ = self.schedule_tasks(self.tasks, 0, 0, self.tasks)

class PriorityGroup(GroupReplicationScheduler):
	"""Schedule replication by choosing group of chunks with maximum lost replica count"""
	def prepare(self, tasks):
		ngroups = self.cluster.encoding.n - 1
		task_groups = [[] for _ in range(ngroups)]
		max_lost = 0
		for failed_disks, chunk in tasks:
			lost = len(failed_disks)
			task_groups[ngroups-lost].append((
				failed_disks, chunk
			))
			if lost > max_lost:
				max_lost = lost
		assert max_lost > 0
		self.tasks = [self.prepare_tasks(tasks) for tasks in task_groups[ngroups-max_lost:]]
		self.prepare_scheduler()

	def schedule(self):
		busy_disks, busy_disks_cnt = 0, 0
		for i, ptasks in enumerate(self.tasks):
			_tasks = self.tasks[i+1] if i + 1 < len(self.tasks) else None
			self.tasks[i], busy_disks, busy_disks_cnt = self.schedule_tasks(ptasks, busy_disks, busy_disks_cnt, _tasks)

	def is_idle(self):
		return not self.tasks or not any(map(len, self.tasks))

class Cluster:
	"""Storage cluster simulator"""
	def __init__(self, ndisks, nchunks, encoding, chunk_allocator, repl_scheduler):
		"""Initialize cluster"""
		self.ndisks = ndisks
		self.encoding = encoding
		self.chunks_orig = chunk_allocator.allocate(ndisks, nchunks, encoding.n)
		# The number of chunks per every disk
		self.disk_chunks = [0] * self.ndisks
		# The bitmap of disks per every chunk
		self.chunks = [0]*nchunks
		for i, chunk_disks in enumerate(self.chunks_orig):
			chunk_bmap = 0
			for d in chunk_disks:
				self.disk_chunks[d] += 1
				chunk_bmap |= 1 << d
			self.chunks[i] = chunk_bmap		
		assert sum(self.disk_chunks) == self.total_blocks()
		# The faulted disks set
		self.fdisks = set()
		self.replicator = repl_scheduler(self)
		self.lost_chunks = []

	def total_chunks(self):
		"""Returns the total number of allocated chunks"""
		return len(self.chunks)

	def total_blocks(self):
		"""Returns the total number of allocated blocks"""
		return len(self.chunks) * self.encoding.n

	def healthy_disks(self):
		"""Returns the number of not yet failed disks"""
		return self.ndisks - len(self.fdisks)

	def is_data_lost(self):
		"""Returns True if some chunks are lost due to disk failures"""
		return len(self.lost_chunks) > 0

	def lost_chunks_count(self):
		"""Returns the number of chunks lost"""
		return len(self.lost_chunks)

	def has_faulted_disks(self):
		return bool(self.fdisks)

	def disk_fault(self, d):
		"""Simulate fault of the disk number d and rebuild replication queue"""
		assert d not in self.fdisks
		assert sum(self.disk_chunks) == self.total_blocks()

		self.fdisks.add(d)
		fdisks_bmap = bitmap(self.fdisks)
		replicate = [
				(
					reduce(lambda l,d: l+[d] if c&(1<<d) else l, self.fdisks, []), # the list of failed disks
					i                                                               # the chunk index
				) for i, c in enumerate(self.chunks) if (c & fdisks_bmap) != 0
		]
		if not replicate:
			return 0

		# check if we lost some chunks
		healthy = self.healthy_disks()
		self.lost_chunks = filter(
				lambda (failed_disks, ch): len(failed_disks) > healthy or self.encoding.is_data_lost(self.chunks[ch], failed_disks),
				replicate
			)
		if self.lost_chunks:
			assert all(map(lambda (failed_disks, ch): set(failed_disks) <= self.fdisks, self.lost_chunks))
			return -1

		# initiate replication
		self.replicator.prepare(replicate)
		return len(replicate)

	def random_disk_fault(self):
		"""Choose disk randomly and simulate its fault if its not yet faulted"""
		d = random.randrange(self.ndisks)
		if d not in self.fdisks and self.disk_chunks[d]:
			self.disk_fault(d)
			return True
		return False

	def disk_recover(self, d):
		"""Mark disk as healthy"""
		assert self.disk_chunks[d] == 0
		self.fdisks.remove(d)

	def allocate_disk_for_recovery(self, c, exclude_bmap = 0):
		"""Allocate disk for chunk recovery given its index. Returns disk index and corresponding bit."""
		exclude_bmap |= self.chunks[c]
		exclude_bmap |= bitmap(self.fdisks)
		# consider original set of disks first
		for d in self.chunks_orig[c]:
			d_bit = 1 << d
			if d_bit & exclude_bmap == 0:
				return d, d_bit

		# choose new disk randomly from healthy ones
		avail = (1 << self.ndisks) - 1 - exclude_bmap
		r = 1 + random.randrange(self.ndisks)
		random_range = avail & ((1 << r) - 1)
		d = hi_bit_index(random_range if random_range else avail)
		return d, 1 << d

	def on_chunks_replicated(self, c, old_d, new_d):
		"""Called by replication scheduler on chunk replication"""
		self.disk_chunks[old_d] -= 1
		self.disk_chunks[new_d] += 1
		self.chunks[c] -= 1 << old_d
		self.chunks[c] |= 1 << new_d
		if self.disk_chunks[old_d] <= 0:
			# Recover failed disk after replication completion
			assert self.disk_chunks[old_d] == 0
			self.disk_recover(old_d)

	def recover_faults(self):
		"""
		Returns False if replication queue is empty. Otherwise replicate some chunks
		from replication queue and returns True.
		"""
		assert not self.lost_chunks
		if self.replicator.is_idle():
			assert not self.fdisks
			return False
		self.replicator.schedule()
		return True

Params = namedtuple('Params', ('fault_rate', 'N', 'C', 'encoding', 'allocator', 'repl_scheduler'))		

def simulate_once(params):
	"""Run disk fault simulation and return the running time to failure"""
	simulation_time = 0.
	c = Cluster(params.N, params.C, params.encoding, params.allocator, params.repl_scheduler)
	while not c.is_data_lost():
		if c.has_faulted_disks():
			c.recover_faults()
			simulation_time += 1
			if random.random() < params.fault_rate:
				c.random_disk_fault()
		else:
			simulation_time += random.expovariate(params.fault_rate)
			c.random_disk_fault()

	print >> sys.stderr, simulation_time
	return simulation_time, c.lost_chunks_count()

def simulate_disk_faults(rounds, params):
	"""Run fault simulation given number of round and returns the meant runtime to failure"""
	p = Pool(max(1, cpu_count()-1))
	results = p.imap_unordered(simulate_once, [params]*rounds)
	p.close()
	p.join()
	times, lost_chunks = zip(*results)
	return sum(times)/float(rounds), sum(lost_chunks)/float(rounds)

if __name__ == '__main__':
	N, C = 20, 2000
	failure_rate = 0.001
	rounds = 100

	for failure_rate in (0.001, 0.0007, 0.0005):
		print '(5, 2) N=%s, C=%s, failure_rate=%s' % (N, C, failure_rate)
		sys.stdout.flush()
		print simulate_disk_faults(rounds, Params(failure_rate, N, C, NK(5, 2), RandomAllocator, PriorityGroup))


