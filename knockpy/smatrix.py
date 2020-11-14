import numpy as np
import scipy.fcluster.hierarchy as hierarchy
from . import mrc
from . import mac
from . import graphs
from . import utilities

def parse_method(method, groups, p):
	""" Decides which method to use to create the 
	knockoff S matrix """
	if method is not None:
		return method
	if np.all(groups == np.arange(1, p + 1, 1)):
		method = "mvr"
	else:
		if p > 1000:
			method = "asdp"
		else:
			method = "sdp"
	return method

def divide_computation(Sigma, max_block):
	"""
	Approximates Sigma as a block-diagonal matrix.
	:param Sigma: Covariance matrix.
	:param max_size: Maximum size of a block in the 
	covariance matrix.
	"""

def divide_computation(Sigma, max_block):
	"""
	Approximates a correlation matrix Sigma as a block-diagonal matrix
	using hierarchical clustering. Roughly follows the R knockoff package.
	"""
	
	# Correlation tree. We add noise to deal with highly structured Sigma.
	p = Sigma.shape[0]
	noise = np.random.randn(p,p)*1e-6
	noise += noise.T
	Sigma = Sigma + noise
	link = graphs.create_correlation_tree(Sigma)
	
	# Set up binary search
	max_clusters = p
	min_clusters = 1
	prev_max_clusters = p
	prev_min_clusters = 1
	
	# Binary search to create clusters
	for j in range(100):
		# Create new groups and check maximum size
		n_clusters = int((max_clusters + min_clusters) / 2)
		groups = hierarchy.cut_tree(link, n_clusters).reshape(-1) + 1
		current_max_block = knockadapt.utilities.calc_group_sizes(groups).max()
	   
		# Cache search info and check maximum size
		prev_max_clusters = max_clusters
		prev_min_clusters = min_clusters
		if current_max_block > max_block:
			min_clusters = n_clusters
		else:
			max_clusters = n_clusters
		# Break if nothing has changed between iterations
		if min_clusters == prev_min_clusters and max_clusters == prev_max_clusters:
			if current_max_block > max_block:
				groups = hierarchy.cut_tree(link, n_clusters + 1).reshape(-1) + 1
			break
			
	return merge_groups(groups, max_block)

def merge_groups(groups, max_block):
	"""
	Merges groups of variables together while ensuring all new groups
	have size less than max_block.
	"""
	p = groups.shape[0]
	new_groups = np.zeros(p)
	# Loop through groups and concatenate them together
	# until we exceed max size
	current_size = 0
	new_group_id = 1
	old_group_ids = np.unique(groups)
	np.random.shuffle(old_group_ids)
	# Iterate through old groups
	for old_group_id in old_group_ids:
		flag = groups == old_group_id
		old_group_size = flag.sum()
		# Either count old group as part of new group
		if current_size + old_group_size <= max_block:
			current_size += old_group_size
		# Or add a new group
		elif old_group_size <= max_block:
			current_size = old_group_size
			new_group_id += 1
		else:
			raise ValueError(f"Group {old_group_id} has size {old_group_size}, exceeding max_block {max_block}")
		new_groups[flag] = new_group_id

	return new_groups

def compute_S_matrix(
	Sigma,
	groups=None,
	method=None,
	solver='cd',
	max_block=1000,
	num_processes=1,
	**kwargs
):
	"""
	Wraps a variety of S-matrix generation functions. 
	This uses a block-diagonal approximation of Sigma if
	the dimension of Sigma exceeds max_block.
	:param Sigma: covariance matrix
	:param groups: groups for group knockoffs
	:param method: Method for constructing
	S-matrix. One of mvr, maxent, sdp, asdp,
	equicorrelated, ci/ciknock.
	:param solver: Method for solving mrc knockoffs.
	One of 'cd' (coordinate descent) or 'psgd'
	(projected gradient descent).
	:param max_block: The maximum block in the block-diagonal
	approximation of Sigma.
	:param n_processes: Number of parallel processes to use
	if Sigma is approximated as a block-diagonal matrix. 
	:param **kwargs: kwargs to one of the downstream
	functions.
	"""
	# If S in kwargs, just return S (important
	# for chaining methods in metro sampling)
	kwargs = kwargs.copy()
	if 'S' in kwargs:
		if kwargs['S'] is not None:
			return kwargs['S']
		else:
			kwargs.pop('S')

	# Initial params
	p = Sigma.shape[0]
	if method is not None:
		method = str(method).lower()
	method = parse_method(method, groups, p)
	if groups is None:
		groups = np.arange(1, p + 1, 1)

	# Scale to correlation matrix
	scale = np.sqrt(np.diag(Sigma))
	scale_matrix = np.outer(scale, scale)
	Sigma = Sigma / scale_matrix

	# Possibly use block-diagonal approximation, either using
	# hierarchical clustering for non-grouped knockoffs or
	# randomly merging groups for group knockoffs.
	if p > max_block:
		if np.all(groups == np.arange(1, p + 1, 1)):
			blocks = divide_computation(Sigma, max_block)
		else:
			blocks = merge_groups(groups, max_block)
		Sigma_blocks = mrc.blockdiag_to_blocks(Sigma, blocks)
		# Recursive subcall for each block. Possibly use multiprocessing.
		S_blocks = utilities.apply_pool(
			func=compute_S_matrix,
			constant_inputs={
				'groups':groups,
				'method':method,
				'solver':solver,
				'max_block':p,
			},
			inputs=Sigma_blocks,
			n_processes=n_processes
		)
		S = np.zeros((p,p))
		block_id = 1
		for block in blocks:
			block_inds = np.where(groups == block_id)[0]
			block_inds = np.ix_(block_inds, block_inds)
			S[block_inds] = block
			block_id += 1
		return S * scale_matrix

	# Currently cd solvers cannot handle group knockoffs
	# (this is todo)
	if not np.all(groups == np.arange(1, p + 1, 1)):
		solver = 'psgd'
	if (method == 'mvr' or method == 'maxent') and solver == 'psgd':
		S = mrc.solve_mrc_psgd(
			Sigma=Sigma, groups=groups, **kwargs
		)
	elif method == 'mvr':
		S = mrc.solve_mvr(
			Sigma=Sigma, **kwargs
		)
	elif method == 'maxent':
		S = mrc.solve_maxent(
			Sigma=Sigma, **kwargs
		)
	elif method == "sdp":
		S = mac.solve_group_SDP(
			Sigma,
			groups,
			**kwargs,
		)
	elif method == 'ciknock' or method == 'ci':
		S = mrc.solve_ciknock(
			Sigma, **kwargs
		)
	elif method == "equicorrelated" or method == 'eq':
		S = mac.solve_equicorrelated(
			Sigma, groups, **kwargs
		)
	else:
		raise ValueError(f"Unrecognized method {method}")

	# Rescale and return
	return S * scale_matrix