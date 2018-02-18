"""
Mean time to data lost estimations for IO bound replication
"""

def nk2mttdl(N, C, n, failure_rate):
	"""Mean time to data lost for (n, n-2) scheme assuming unit chunk replication time"""
	return 2.*N**5/(C**2*failure_rate**3*n**2*(n-1)**3)

def nk3mttdl(N, C, n, failure_rate):
	"""Mean time to data lost for (n, n-3) scheme assuming unit chunk replication time"""
	return 6.*N**9/(C**3*failure_rate**4*n**3*(n-1)**2*(n-2)**4)
