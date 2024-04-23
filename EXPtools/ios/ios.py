import numpy as np
import pyEXP


class Coefficients:
    def __init__(self, path, basename, codebase):
        """


        """

        self.path = path
        self.basename = basename
        self.codebase = codebase
        

        if self.codebase == "agama":
            self.coefs = np.loadtxt(path + basename, skiprows=10)

    def agama_scf(self, ):
		"""

		Adapted from: https://github.com/GalacticDynamics-Oxford/Agama/blob/master/py/example_basis_set.py
		"""
		Snlm = np.zeros((nmax+1, lmax+1, lmax+1))
		Tnlm = np.zeros((nmax+1, lmax+1, lmax+1))
		for n in range(nmax+1):
			for l in range(lmax+1):
				j=1 # j=0 is the value of n order
				for m in range(l):
					Tnlm[n, l, l-m] = agama_scf[n, j]*2**0.5
					j+=1
				for m in range(l+1):
					#if m<0:
					Snlm[n, l, m] = agama_scf[n, j]*(2**0.5 if m>0 else 1)
					j+=1
		return Snlm, Tnlm

        
        return coefs

    def gala_scf(self,):
		## TODO: have to decide what is the standard file gala write coefficients perhaps yaml? double-check with APW
        return 0

	def exp(self, config):
		return 0



