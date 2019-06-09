import numpy as np
def searchsorted2d(a: np.ndarray ,b: np.ndarray) -> np.ndarray:
	'''
	Inserts ith element of b into sorted ith row of a

	Parameters
	----------
	a: np.ndarray
		rxc matrix, each rows is sorted
	b: np.ndarray
		rx1 vector 

	Returns
	-------
	np.ndarray
		rx1 vector of locations 
	'''
	m,n = a.shape
	b = b.ravel()
	assert b.shape[0] == a.shape[0], "Number of values of b equal number of rows of a"
	max_num = np.maximum(a.max() - a.min(), b.max() - b.min()) + 1
	r = max_num*np.arange(a.shape[0])
	p = np.searchsorted( ((a.T+r).T).ravel(), b+r )
	return p - n*np.arange(m)
