"""
Storage recovery model for LRC encoding
"""

from cluster_recovery_graph import RecoveryGraph

def lrc22_recovery_graph(N, C, n):
	"""Build recovery graph for encoding with 2 local and 2 global parity blocks"""
	g = RecoveryGraph(N, C, n)
	# label states by 5 numbers (e1, e2, l1, l2, gp):
	#   2 data error counts
	#   2 local parity error counts (0..1)
	#   global  parity error count  (0..2)	
	assert n > 4
	assert n % 2 == 0
	grp = (n - 2) / 2
	for e1 in range(grp):
		for e2 in range(grp):
			for l1 in range(2):
				for l2 in range(2):
					for gp in range(3):
						id = (e1, e2, l1, l2, gp)     # state id
						lvl = e1 + e2 + l1 + l2 + gp  # fault level
						# Find the number of parity blocks useful for data reconstruction
						parities = 2 - gp # global ones
						if e1 and not l1: parities += 1
						if e2 and not l2: parities += 1
						# Check if errors are theoretically recoverable
						if e1 + e2 > parities:
							# Failed state
							assert lvl > 3
							g.add_state(id)
							continue
						blks_left = float(n - lvl)
						if e1 + l1 == 1 or e2 + l2 == 1:
							# locally recoverable
							g.add_state(id, lvl, grp - 1)
						else:
							# recover using all available block
							g.add_state(id, lvl, blks_left)
						# Add all possible failure transitions
						data_grp = grp - 1
						if e1 < data_grp: g.add_fault(id, (e1 + 1, e2, l1, l2, gp), (data_grp - e1) / blks_left)
						if e2 < data_grp: g.add_fault(id, (e1, e2 + 1, l1, l2, gp), (data_grp - e2) / blks_left)
						if not l1:        g.add_fault(id, (e1, e2, 1, l2, gp), 1 / blks_left)
						if not l2:        g.add_fault(id, (e1, e2, l1, 1, gp), 1 / blks_left)
						if gp < 2:        g.add_fault(id, (e1, e2, l1, l2, gp + 1), (2 - gp) / blks_left)
	return g

if __name__ == '__main__':
	import sys
	from cluster_recovery_graph import simulate_disk_faults, nk_recovery_graph

	N, C, n = 50, 2500, 16
	lrc = lrc22_recovery_graph(N, C, n)
	nk3 = nk_recovery_graph(N, C, n, n - 3)
	nk4 = nk_recovery_graph(N, C, n, n - 4)
	nk2 = nk_recovery_graph(N, C, n/2, n/2 - 2)
	rounds = 100
	for failure_rate in (0.004,):
		print 'N=%s C=%s n=%s failure_rate=%s' % (N, C, n, failure_rate)
		sys.stdout.flush()
		print simulate_disk_faults(rounds, lrc, failure_rate), simulate_disk_faults(rounds, nk3, failure_rate), simulate_disk_faults(rounds, nk4, failure_rate), simulate_disk_faults(rounds, nk2, failure_rate)
