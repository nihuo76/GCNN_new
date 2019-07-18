Ham: [18x18x343 double   —> Hamiltonian(norb, norb, ncell)
norb: 18 —> number of orbitals in total (including all orbitals from all atoms)
ncell: 343  —> number of unit cells 
cell_position: [343x3 double]   —> cell positions (as integer translation of unit cell along x, y, and z)
cell_weight: [343x1 double]   —> weighting factor of each cell (not important for our learning here)
atom: [1x1 struct]  —> contains information about atoms (see the sublist below)
H: [3x3 double] —> unit cell
efermi: 2.3968  —> fermi level
nelectron: 35  —> number of electrons per unit cell


>> Hr.atom
         natom: 4
             X: [4x3 double]          —> atom position (cartesian)
             S: [4x3 double]          —> atom position (fractional) 
             Z: [4x1 double]          —> atomic number of each atom
         ntype: 3                         —> number of element types
       element: {'Ca'  'Zn'  'Ga’}    —> element type
     atom2type: [1 1 2 3]            —> mapping atom to element type
    element_LM: {{4x1 cell}  {6x1 cell}  {4x1 cell}}   —> element LM channels (with detailed orbital LM symbol)
      atom2orb: [4x2 double]       —> mapping each atom’s start/end orbital indices to the total norb)
     orbsymbol: {'sp'  'sp'  'sd'  'sp’}   —> orbital symbols for each atom

Let me know if you have any questions.