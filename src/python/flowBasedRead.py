# Flow-based read/haplotype class
# This will be a class that would hold a read in flow base
from . import readExpander
from . import utils
import numpy as np
import pysam

class FlowBasedRead : 
	'''Class that helps working with flow based reads

	This is the class that allows for comparison between 
	reads in flow base etc. 

	Attributes
	----------
	seq: str
		Original read sequence
	r_seq: str
		Reverse complement read sequence
	key: np.ndarray
		sequence in flow base
	rkey: np.ndarray
		reverse complement sequence in flow base
	flow_order: str
		sequence of flows
	r_flow_order: str

	Methods
	-------
	__init__ ()
	'''



	

	

