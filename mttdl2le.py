"""
Calculates mean time to data lost estimations for disk and latent block failures given the following parameters:
 N    - total number of disks
 C    - total number of chunks
 n    - data blocks tuple length
 B    - number of disk blocks per disk B=C*n/N
 Td   - time to failure for the particular disk
 Tb   - time to failure for the particular block on the disk
 Ts   - scrabbling time per disk block, so the total scrubbing time is TS = Ts*B
All times are with respect to the recovery time (so its assumed to be equal to 1)
We are considering (n, n-2) data storage scheme (redundancy 2)
"""

def nk2_disk_faults(N, C, n, Td):
	"""The rate of the faults due to the disk failures"""
	return C**2*n*n*(n-1)*(n-1)*(n-1)/(2*N*N*Td*Td*Td)

def nk2_block_faults(N, C, n, Tb, Ts):
	"""The rate of the faults due to the block failures"""
	TS = Ts*C*n/N
	return C*n*(n-1)*(n-2)*TS*TS/(Tb*Tb*Tb)

def nk2_disk_faults1(N, C, n, Td, Tb, Ts):
	"""The rate of data lost on the first disk fault"""
	TS = Ts*C*n/N
	return C*n*(n-1)*(n-2)*TS*TS/(Tb*Tb*Td)

def nk2_disk_faults2(N, C, n, Td, Tb, Ts):
	"""The rate of data lost on the second disk fault"""
	TS = Ts*C*n/N
	return C**2*n*n*(n-1)*(n-1)*(n-2)*TS/(2*N*N*Td*Td*Tb)

def total_faults(N, C, n, Td, Tb, Ts):
	"""The total fault rate"""
	return nk2_disk_faults(N, C, n, Td) + nk2_block_faults(N, C, n, Tb, Ts) \
			+ nk2_disk_faults1(N, C, n, Td, Tb, Ts) \
			+ nk2_disk_faults2(N, C, n, Td, Tb, Ts)

def relative_fault_rate(N, B, n, Td, Tb, Ts):
	"""The total fault rate relative to the fault rate considering disk faults only"""
	d = Td*B/Tb
	return 1 + (n-2)*d*Ts/(n-1) + 2*N*(n-2)*(1+d/B)*d*d*Ts*Ts/(B*(n-1)*(n-1))

if __name__ == '__main__':
	import numpy as np
	import matplotlib.pyplot as plt
	N, C, n = 50, 2500, 5
	TB = 1e5
	Tb = TB * C * n / N
	Ts = 1.

	Td = np.logspace(4, 8)
	Fd = nk2_disk_faults(N, C, n, Td)
	Ft = total_faults(N, C, n, Td, Tb, Ts)
	Fb = nk2_block_faults(N, C, n, Tb*np.ones(len(Td)), Ts)

	print total_faults(N, C, n, TB, Tb, Ts) / nk2_disk_faults(N, C, n, TB)

	plt.figure(1)

	np.savetxt('nk2_disk_faults.dat',  np.vstack((Td, 1./Fd)).T)
	np.savetxt('nk2_block_faults.dat', np.vstack((Td, 1./Fb)).T)
	np.savetxt('nk2_total_faults.dat', np.vstack((Td, 1./Ft)).T)

	plt.plot(Td, 1./Fd, '--', label='disk faults')
	plt.plot(Td, 1./Fb, '--', label='block faults')
	plt.plot(Td, 1./Ft, label='total faults')

	plt.xscale('log')
	plt.yscale('log')
	plt.title('Mean time to data loss in the presence\nof latent disk block failures')
	plt.xlabel('disk lifetime / replication time')
	plt.ylabel('MTTDL / replication time')
	plt.legend()

	plt.figure(2)

	df_fraction = nk2_disk_faults(N, C, n, Td) / Ft
	bf_fraction = nk2_block_faults(N, C, n, Tb, Ts) / Ft
	b1_fraction = nk2_disk_faults1(N, C, n, Td, Tb, Ts) / Ft
	b2_fraction = nk2_disk_faults2(N, C, n, Td, Tb, Ts) / Ft

	np.savetxt('nk2_df_fraction.dat', np.vstack((Td, df_fraction)).T)
	np.savetxt('nk2_bf_fraction.dat', np.vstack((Td, bf_fraction)).T)
	np.savetxt('nk2_b1_fraction.dat', np.vstack((Td, b1_fraction)).T)
	np.savetxt('nk2_b2_fraction.dat', np.vstack((Td, b2_fraction)).T)

	plt.plot(Td, df_fraction, label='disk faults')
	plt.plot(Td, bf_fraction, label='block faults')
	plt.plot(Td, b1_fraction, label='1st block recovery faults')
	plt.plot(Td, b2_fraction, label='2nd block recovery faults')

	plt.xscale('log')
	plt.title('The fraction of the total faults related\nto the particular data lost mechanism')
	plt.xlabel('disk lifetime / replication time')
	plt.ylabel('Total faults fraction')
	plt.legend()

	plt.figure(3)

	plt.title('Data lost vs scrabling rate')
	
	for i, k in enumerate((.1, 1., 10.)):
		Td = 1e5
		TB = Td * k
		Tb = TB * C * n / N
		Ts = np.logspace(0, 3)
		r = nk2_disk_faults(N, C, n, Td)/total_faults(N, C, n, Td, Tb, Ts)
		rt = 1/(1 + .75*Td*Ts/TB)

		np.savetxt('nk2_relative%d.dat' % i, np.vstack((Ts, r)).T)
		np.savetxt('nk2_relative%d_.dat' % i, np.vstack((Ts, rt)).T)

		plt.plot(Ts, r, label='$T_B=%gT_d$' % k)
		plt.plot(Ts, rt, '--')

	plt.xlabel('block scrabling time / replication time')
	plt.ylabel('MTTDL relative to pure disk faults case')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()

	plt.figure(4)

	plt.title('Data lost vs number of disks')

	for i, B in enumerate((50, 500, 5000)):
		N = np.arange(2, 100001, 2, dtype=float)
		C = N * B / n
		Td = 1e5
		TB = Td / 2
		Tb = TB * B
		Ts = 10
		r = nk2_disk_faults(N, C, n, Td)/total_faults(N, C, n, Td, Tb, Ts)
		np.savetxt('nk2_scaling%d.dat' % i, np.vstack((N, r)).T)
		plt.plot(N, r, label='B=%d' % B)

	plt.xlabel('N')
	plt.ylabel('MTTDL relative to pure disk faults case')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()

	plt.show()
