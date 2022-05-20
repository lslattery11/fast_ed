#
import scipy
import scipy.sparse as sparse
import scipy.stats
import numpy
import numpy as np
import itertools
#import cmath

#convert site (n,m) to an index
def site_to_idx(depth,site):
    row = site[0]
    column = site[1]
    idx = depth*row+column
    return idx

#J1J2_2D Model.
class J1J2_2D:
  def __init__(self, width, depth, J1,J2, h, blockidx=None):
    #Dimensions of lattice
    self.width = width
    self.depth = depth
    #NN coupling
    self.J1= J1
    #NNN coupling
    self.J2 = J2
    #on site pauli x term
    self.h = h
    #Z sector
    self.blockidx=blockidx
    #print "Got field configuration: " , self.fields
 
    self.sectorbasis = {}
    self.state_from_baseidx = {}
    basiscounter= 0
    # Create basis
    for i in range(0,2**(width*depth)):
    #Use thisif statement to set z sector symmetry.
      #nup= bin(i).count('1')
      #if nup==self.blockidx:
      self.sectorbasis[i]= basiscounter
      self.state_from_baseidx[basiscounter]=i 
      basiscounter = basiscounter + 1

    self.dim=basiscounter

    self.matrix=None
    self.eigenvalues=None
    self.eigenvectors=None
    self.expiHt=None

  def create_matrix_block(self):
    rowidx=[]
    colidx=[]
    data=[]


    #create list of NN pairs (OBC) index as a single integer for a site.
    NN_pairs = []
    sites = list(itertools.product(range(self.width),range(1)))
    for site in sites:
        for i in range(self.depth-1):
            hNN = (site[0],site[1]+i+1)
            NN_pairs.append([site_to_idx(self.depth,hNN),site_to_idx(self.depth,(site[0],site[1]+i))])
    sites = list(itertools.product(range(1),range(self.depth)))
    for site in sites:
        for i in range(self.width-1):
            vNN = (site[0]+i+1,site[1])
            NN_pairs.append([site_to_idx(self.depth,vNN),site_to_idx(self.depth,(site[0]+i,site[1]))])
    #create list of NNN pairs (OBC) index as a single integer for a site.
    NNN_pairs = []
    sites = list(itertools.product(range(self.width-1),range(1)))
    for site in sites:
        for i in range(self.depth-1):
            rdNNN = (site[0]+1,site[1]+i+1)
            ldNNN = (site[0]+1,site[1]+i)
            NNN_pairs.append([site_to_idx(self.depth,rdNNN),site_to_idx(self.depth,(site[0],site[1]+i))])
            NNN_pairs.append([site_to_idx(self.depth,ldNNN),site_to_idx(self.depth,(site[0],site[1]+i+1))])
    #print(NNN_pairs)
    # diagonal part
    for s in self.sectorbasis:
        el=0.0
        for i,j in NN_pairs:
            #if spin up on site sz1 = 1 else sz1 = -1
            sz1 = 1-2*float((s&(1<<i))>0)
            #if spin up on site sz1 = 1 else sz2 = -1
            sz2 = 1-2*float((s&(1<<j))>0)
            el+= self.J1*sz1*sz2
        for i,j in NNN_pairs:
            #if spin up on site sz1 = 1 else sz1 = -1
            sz1 = 1-2*float((s&(1<<i))>0)
            #if spin up on site sz1 = 1 else sz2 = -1
            sz2 = 1-2*float((s&(1<<j))>0)
            el+= self.J2*sz1*sz2
            
        for i in range(0,self.width*self.depth):
            #if spin up on site sz = 1 else sz = -1
            sz = 1-2*float((s&(1<<i))>0)
            el += -self.h*sz
        stateidx=self.sectorbasis[s]
        rowidx.append(stateidx)
        colidx.append(stateidx)
        data.append(el)
    #for offdiag part
    for s in self.sectorbasis:
      stateidx=self.sectorbasis[s]
      for i,j in NN_pairs:
          el=0.0
          #for the sigmax sigmax and sigmay sigmay part we flip the bits
          #flip bits
          s2=(s^(1<<i))
          s2=(s2^(1<<j))
          #for the sigma x sigma x we add -J to element.
          el += self.J1
          #for sigma y sigma y we need need to add -J(+-1j)(+-1j) to element
          #if spin up on site sz1 = -1 else sz1 = 1
          sz1 = 2*float((s&(1<<i))>0)-1
          #if spin up on site sz1 = -1 else sz2 = 1
          sz2 = 2*float((s&(1<<j))>0)-1
          el += -self.J1*sz1*sz2
          stateidx2=self.sectorbasis[s2]
          rowidx.append(stateidx)
          colidx.append(stateidx2)
          data.append(el)
          
      for i,j in NNN_pairs:
          el=0.0
          #for the sigmax sigmax and sigmay sigmay part we flip the bits
          #flip bits
          s2=(s^(1<<i))
          s2=(s2^(1<<j))
          #for the sigma x sigma x we add -J to element.
          el += self.J2
          #for sigma y sigma y we need need to add -J(+-1j)(+-1j) to element
          #if spin up on site sz1 = -1 else sz1 = 1
          sz1 = 2*float((s&(1<<i))>0)-1
          #if spin up on site sz1 = -1 else sz2 = 1
          sz2 = 2*float((s&(1<<j))>0)-1
          el += -self.J2*sz1*sz2
          stateidx2=self.sectorbasis[s2]
          rowidx.append(stateidx)
          colidx.append(stateidx2)
          data.append(el)
    
    self.dim=len(self.sectorbasis)
    self.matrix=scipy.sparse.csr_matrix( (data, (rowidx,colidx)),shape=(self.dim,self.dim))
    #	print self.matrix


  def diagonalize(self):
    print ("Dense matrix needs ", self.dim**2*8/1024/1024, " MiB of memory.")
    self.eigenvalues, self.eigenvectors = numpy.linalg.eigh(self.matrix.todense())


  def calc_spectrum(self):
    print ("Dense matrix needs ", self.dim**2*8/1024/1024, " MiB of memory.")
    self.eigenvalues = numpy.linalg.eigvalsh(self.matrix.todense())

  def sparse_calc_spectrum(self,k=3):
    self.eigenvalues, self.eigenvectors = scipy.sparse.linalg.eigsh(self.matrix.T,k,which='SA')


  def timeevolve_exact(self, state, t):
    if self.eigenvalues==None:
      print( "ERROR: You need to diagonalize the Hamiltonian before using this routine!")
      return -1

    expiEt=numpy.exp(-1.0j * self.eigenvalues * t) 
    diag=scipy.sparse.dia_matrix((expiEt,[0]),shape=(len(expiEt),len(expiEt)))
    #diag=numpy.diag(expiEt)
    self.expiHt= numpy.dot(  (self.eigenvectors), ( diag.dot(numpy.conjugate(self.eigenvectors.T)) ) )
    psit = numpy.dot( self.expiHt, state )
    return psit


  # Only works for pure states, not formulated with density matrices
  def EE_sites(self, psi, sites):
    num_sites = len(sites)
    not_sites = range(self.L)
    for i in sites: not_sites.remove(i)
    dim_A = 2**num_sites
    dim_notA = 2**(self.L-num_sites)
    folded_psi = np.zeros((dim_A, dim_notA))
    # How to quickly retrieve the binary number on a list of sites
    for s in self.sectorbasis:
      s_A = sum([(2**i)*((s&(1<<sites[i]))>0) for i in range(num_sites)])
      s_notA = sum([(2**i)*((s&(1<<not_sites[i]))>0) for i in range(self.L-num_sites)])
      folded_psi[s_A, s_notA] = psi[self.sectorbasis[s]]

    U, S, V = np.linalg.svd(folded_psi)
    entanglement = 0.0
    for value in S:
      entanglement += -np.log(value**2) * value**2
    print(S**2)
    return entanglement


  ### Figuring out how to efficiently compute EE through observables
  def ns_and_sigmas(self, psi, i, j):
    if i==j:
      print( 'i has to be different than j' )
      return
    n_i = 0.0
    n_j = 0.0
    n_ij = 0.0
    si_sj = 0.0
    for s in self.sectorbasis:
      mask_i = 1<<i
      mask_j = 1<<j
      if s&mask_i>0:
        n_i += psi[self.sectorbasis[s]]**2.0
      if s&mask_j>0:
        n_j += psi[self.sectorbasis[s]]**2.0
      if s&mask_i>0 and s&mask_j>0:
        n_ij += psi[self.sectorbasis[s]]**2.0
      if s&mask_i==0 and s&mask_j>0:
        s1 = (s^mask_i)^(mask_j)
        si_sj += psi[self.sectorbasis[s]]*psi[self.sectorbasis[s1]]
    return n_i, n_j, n_ij, si_sj
      
  ## Cheap implementation that eventually should be optimized
  def n_i(self, psi, i):
    result = 0.0
    for s in self.sectorbasis:
      mask_i = 1<<i
      if s&mask_i>0:
        result += psi[self.sectorbasis[s]]**2.0
    return result


  def n_ij(self, psi, i, j):
    if i==j:
      print( 'i has to be different than j' )
      return
    result = 0.0
    for s in self.sectorbasis:
      mask_i = 1<<i
      mask_j = 1<<j
      if s&mask_i>0 and s&mask_j>0:
        result += psi[self.sectorbasis[s]]**2.0
    return result


  def sigmapi_sigmamj(self, psi, i, j):
    if i==j:
      print( 'i has to be different than j' )
      return
    result = 0.0
    for s in self.sectorbasis:
      mask_i = 1<<i
      mask_j = 1<<j
      if s&mask_i==0 and s&mask_j>0:
        s1 = (s^mask_i)^(mask_j)
        result += psi[self.sectorbasis[s]]*psi[self.sectorbasis[s1]]
    return result
