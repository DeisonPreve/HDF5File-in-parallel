# Preliminaries and mesh
from dolfin import *
import numpy.linalg as LA
#from mshr import *
from fenics import *
import sympy as sp
import numpy as np
from mpi4py import MPI as mpi
import h5py

#  run the command directly in a terminal like this ( I have generated (from FreeFem++) BIG.mesh):
#  dolfin-convert vertebra+vite_orign.mesh vertebra+vite_orign.xml
'''
#mesh = Mesh('vertebra+viti_testa_viti_FINE.xml') #viti+vertebra_coarse.xml
mesh = UnitCubeMesh(10, 10, 10)
mesh_f = File ("./cubo.pvd")
mesh_f << mesh
mesh_eigen = Mesh('sphere.xml')
mesh_s = File ("./esfera.pvd")
mesh_s << mesh_eigen

mesh = Mesh("CUBEto.xml")
fd = MeshFunction('size_t', mesh, "CUBEto_facet_region.xml");
cd = MeshFunction('size_t', mesh, "CUBEto_physical_region.xml");

hdf = HDF5File(mesh.mpi_comm(), "file.h5", "w")
hdf.write(mesh, "/mesh")
hdf.write(cd, "/cd")
hdf.write(fd, "/fd")
'''

mesh = Mesh()
hdf = HDF5File(MPI.comm_world, "file.h5", "r") #mesh.mpi_comm()
hdf.read(mesh, "/mesh", False)
cd =  MeshFunction("size_t", mesh, mesh.topology().dim())
hdf.read(cd, "/cd")
fd = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
hdf.read(fd, "/fd")
MeshPartitioning.build_distributed_mesh(mesh)

fd.set_all(0)

ds = Measure('ds', domain=mesh, subdomain_data=fd)
dx = Measure('dx', domain=mesh, subdomain_data=cd)

n = FacetNormal(mesh)

# Initialization of the iterative procedure and output requests
deltaT  = 1.0
kappa=1E-5

pi=3.14159265359

# Define Space
V = FunctionSpace(mesh, 'CG', 1)
W = VectorFunctionSpace(mesh, 'CG', 1)
WW = FunctionSpace(mesh, 'DG', 0)
p, q = TrialFunction(V), TestFunction(V)
pnew, pold, Hold, Hold_star = Function(V), Function(V), Function(V), Function(V)

'''
T = FunctionSpace(mesh, 'CG', 1)

local_range = T.dofmap().ownership_range()
local_dim = local_range[1] - local_range[0]

# ############ E ##############################

file_vals = open ( 'E_interp_tutto_FINE.txt', 'r' )
list_vals = [ float ( line ) for line in file_vals.readlines() ]

array_vals = np.array ( list_vals)
E = Function ( T )
d2v = dof_to_vertex_map ( T )
global_vertex_numbers = mesh.topology().global_indices(0)
global_vertices = global_vertex_numbers[d2v[:local_dim]]
local_data = array_vals[global_vertices]
E.vector()[:] = local_data[:local_dim]
file = File ("./Vertebra_tutto_fine_gmres1/E.pvd")
file << E

'''

E=82*1e3
nuu = 0.4
lmbda =  E*nuu/(1-nuu**2)
mu =   E*(1-2*nuu)/(1-nuu**2)
Gc=3*1e-4

l= 1*1e-5

tol=1E-8
kappa=1e-6

epsilon_star = -0.0015

eigenstrain_matrix = Constant(((epsilon_star,0.0,0.0),(0.0,epsilon_star,0.0),(0.0,0.0,epsilon_star)))

# Constituive functions
def epsilon(u):
    return sym(grad(u))
    
r=Function(V)
# Positive strain
def strn_p(u):
    t = sym(grad(u))

    p1 = t[0,1]**2+t[0,2]**2+t[1,2]**2    
    qq =(t[0,0]+t[1,1]+t[2,2])/3 
    p2 =(t[0,0] - qq)**2 + (t[1,1] - qq)**2 + (t[2,2] - qq)**2 + 2*p1 
    pp = sqrt(p2/6)

    t = as_tensor(t)
    B =(1/pp)*(t - qq*Identity(3))
    B = as_tensor(B)
    r = det(B)/2

    fai = acos(r)/3
    # the eigenvalues satisfy eig3 <= eig2 <= eig1
    v00 = qq + 2 * pp * cos(fai)
    v22 = qq + 2 * pp * cos(fai + (2*pi/3))
    v11 = 3 * qq - v00 - v22

    a1=t[:,0]-v00*Constant((1., 0., 0.)) 
    a2=t[:,1]-v00*Constant((0., 1., 0.)) 
    a3=t[:,0]-v11*Constant((1., 0., 0.)) 
    a4=t[:,1]-v11*Constant((0., 1., 0.)) 


    w00s = a1[1]*a2[2] - a1[2]*a2[1]
    w10s = a1[2]*a2[0] - a1[0]*a2[2]
    w20s = a1[0]*a2[1] - a1[1]*a2[0]
  
    w01s = a3[1]*a4[2] - a3[2]*a4[1]
    w11s = a3[2]*a4[0] - a3[0]*a4[2]
    w21s = a3[0]*a4[1] - a3[1]*a4[0]

    w00 = w00s/sqrt(w00s**2+w10s**2+w20s**2)
    w10 = w10s/sqrt(w00s**2+w10s**2+w20s**2)
    w20 = w20s/sqrt(w00s**2+w10s**2+w20s**2)

    w01 = w01s/sqrt(w01s**2+w11s**2+w21s**2)
    w11 = w11s/sqrt(w01s**2+w11s**2+w21s**2)
    w21 = w21s/sqrt(w01s**2+w11s**2+w21s**2)


    w02s = w10*w21 - w20*w11

    w12s = w20*w01 - w00*w21
    w22s = w00*w11 - w10*w01

    w02 = w02s/sqrt(w02s**2+w12s**2+w22s**2)
    w12 = w12s/sqrt(w02s**2+w12s**2+w22s**2)
    w22 = w22s/sqrt(w02s**2+w12s**2+w22s**2) 


    wp = ([w00, w01, w02],[w10, w11, w12],[w20, w21, w22])
    wp = as_tensor(wp)

    wp_tr = ([w00,w10, w20],[w01,w11, w21], [w02, w12, w22])
    wp_tr = as_tensor(wp_tr)

    v00 = conditional(gt(v00,0.0),v00,0.0)
    v11 = conditional(gt(v11,0.0),v11,0.0)
    v22 = conditional(gt(v22,0.0),v22,0.0)

    vp = ([v00,0.0,0.0],[0.0,v11,0.0],[0.0,0.0,v22])
    vp = as_tensor(vp)  
    return wp*vp*wp_tr

# Negative strain
def strn_n(u):
    t = sym(grad(u))

    p1 = t[0,1]**2+t[0,2]**2+t[1,2]**2    
    qq =(t[0,0]+t[1,1]+t[2,2])/3 
    p2 =(t[0,0] - qq)**2 + (t[1,1] - qq)**2 + (t[2,2] - qq)**2 + 2*p1 
    pp = sqrt(p2/6)

    t = as_tensor(t)  
    B =(1/pp)*(t - qq*Identity(3))
    B = as_tensor(B)  
    r = det(B)/2

    fai = acos(r)/3
    # the eigenvalues satisfy eig3 <= eig2 <= eig1
    v00 = qq + 2 * pp * cos(fai)
    v22 = qq + 2 * pp * cos(fai + (2*pi/3))
    v11 = 3 * qq - v00 - v22

    a1=t[:,0]-v00*Constant((1., 0., 0.)) 
    a2=t[:,1]-v00*Constant((0., 1., 0.)) 
    a3=t[:,0]-v11*Constant((1., 0., 0.)) 
    a4=t[:,1]-v11*Constant((0., 1., 0.)) 


    w00s = a1[1]*a2[2] - a1[2]*a2[1]
    w10s = a1[2]*a2[0] - a1[0]*a2[2]
    w20s = a1[0]*a2[1] - a1[1]*a2[0]
  
    w01s = a3[1]*a4[2] - a3[2]*a4[1]
    w11s = a3[2]*a4[0] - a3[0]*a4[2]
    w21s = a3[0]*a4[1] - a3[1]*a4[0]

    w00 = w00s/sqrt(w00s**2+w10s**2+w20s**2)
    w10 = w10s/sqrt(w00s**2+w10s**2+w20s**2)
    w20 = w20s/sqrt(w00s**2+w10s**2+w20s**2)

    w01 = w01s/sqrt(w01s**2+w11s**2+w21s**2)
    w11 = w11s/sqrt(w01s**2+w11s**2+w21s**2)
    w21 = w21s/sqrt(w01s**2+w11s**2+w21s**2)

    w02s = w10*w21 - w20*w11
    w12s = w20*w01 - w00*w21
    w22s = w00*w11 - w10*w01

    w02 = w02s/sqrt(w02s**2+w12s**2+w22s**2)
    w12 = w12s/sqrt(w02s**2+w12s**2+w22s**2)
    w22 = w22s/sqrt(w02s**2+w12s**2+w22s**2) 

    wn = ([w00, w01, w02],[w10, w11, w12],[w20, w21, w22])
    wn = as_tensor(wn)

    wn_tr = ([w00,w10, w20],[w01,w11, w21], [w02, w12, w22])
    wn_tr = as_tensor(wn_tr)

    v00 = conditional(lt(v00,0.0),v00,0.0)
    v11 = conditional(lt(v11,0.0),v11,0.0)
    v22 = conditional(lt(v22,0.0),v22,0.0)

    vn = ([v00,0.0,0.0],[0.0,v11,0.0],[0.0,0.0,v22])
    vn = as_tensor(vn)  
    return wn*vn*wn_tr

def psi(u):
    return 0.5*lmbda*(0.5*(tr(epsilon(u))+abs(tr(epsilon(u)))))**2+mu*tr(strn_p(u)*strn_p(u))
    
def H(uold,unew,Hold):
    return conditional(lt(psi(uold),psi(unew)),psi(unew),Hold)
    
# Constituive functions star
def epsilon_star(u):
    return sym(grad(u))-eigenstrain_matrix
    
r=Function(V)
# Positive strain star
def strn_p_star(u):
    t = sym(grad(u))-eigenstrain_matrix

    p1 = t[0,1]**2+t[0,2]**2+t[1,2]**2    
    qq =(t[0,0]+t[1,1]+t[2,2])/3 
    p2 =(t[0,0] - qq)**2 + (t[1,1] - qq)**2 + (t[2,2] - qq)**2 + 2*p1 
    pp = sqrt(p2/6)

    t = as_tensor(t)
    B =(1/pp)*(t - qq*Identity(3))
    B = as_tensor(B)
    r = det(B)/2

    fai = acos(r)/3
    # the eigenvalues satisfy eig3 <= eig2 <= eig1
    v00 = qq + 2 * pp * cos(fai)
    v22 = qq + 2 * pp * cos(fai + (2*pi/3))
    v11 = 3 * qq - v00 - v22

    a1=t[:,0]-v00*Constant((1., 0., 0.)) 
    a2=t[:,1]-v00*Constant((0., 1., 0.)) 
    a3=t[:,0]-v11*Constant((1., 0., 0.)) 
    a4=t[:,1]-v11*Constant((0., 1., 0.)) 


    w00s = a1[1]*a2[2] - a1[2]*a2[1]
    w10s = a1[2]*a2[0] - a1[0]*a2[2]
    w20s = a1[0]*a2[1] - a1[1]*a2[0]
  
    w01s = a3[1]*a4[2] - a3[2]*a4[1]
    w11s = a3[2]*a4[0] - a3[0]*a4[2]
    w21s = a3[0]*a4[1] - a3[1]*a4[0]

    w00 = w00s/sqrt(w00s**2+w10s**2+w20s**2)
    w10 = w10s/sqrt(w00s**2+w10s**2+w20s**2)
    w20 = w20s/sqrt(w00s**2+w10s**2+w20s**2)

    w01 = w01s/sqrt(w01s**2+w11s**2+w21s**2)
    w11 = w11s/sqrt(w01s**2+w11s**2+w21s**2)
    w21 = w21s/sqrt(w01s**2+w11s**2+w21s**2)


    w02s = w10*w21 - w20*w11

    w12s = w20*w01 - w00*w21
    w22s = w00*w11 - w10*w01

    w02 = w02s/sqrt(w02s**2+w12s**2+w22s**2)
    w12 = w12s/sqrt(w02s**2+w12s**2+w22s**2)
    w22 = w22s/sqrt(w02s**2+w12s**2+w22s**2) 


    wp = ([w00, w01, w02],[w10, w11, w12],[w20, w21, w22])
    wp = as_tensor(wp)

    wp_tr = ([w00,w10, w20],[w01,w11, w21], [w02, w12, w22])
    wp_tr = as_tensor(wp_tr)

    v00 = conditional(gt(v00,0.0),v00,0.0)
    v11 = conditional(gt(v11,0.0),v11,0.0)
    v22 = conditional(gt(v22,0.0),v22,0.0)

    vp = ([v00,0.0,0.0],[0.0,v11,0.0],[0.0,0.0,v22])
    vp = as_tensor(vp)  
    return wp*vp*wp_tr

# Negative strain star
def strn_n_star(u):
    t = sym(grad(u))-eigenstrain_matrix

    p1 = t[0,1]**2+t[0,2]**2+t[1,2]**2    
    qq =(t[0,0]+t[1,1]+t[2,2])/3 
    p2 =(t[0,0] - qq)**2 + (t[1,1] - qq)**2 + (t[2,2] - qq)**2 + 2*p1 
    pp = sqrt(p2/6)

    t = as_tensor(t)  
    B =(1/pp)*(t - qq*Identity(3))
    B = as_tensor(B)  
    r = det(B)/2

    fai = acos(r)/3
    # the eigenvalues satisfy eig3 <= eig2 <= eig1
    v00 = qq + 2 * pp * cos(fai)
    v22 = qq + 2 * pp * cos(fai + (2*pi/3))
    v11 = 3 * qq - v00 - v22

    a1=t[:,0]-v00*Constant((1., 0., 0.)) 
    a2=t[:,1]-v00*Constant((0., 1., 0.)) 
    a3=t[:,0]-v11*Constant((1., 0., 0.)) 
    a4=t[:,1]-v11*Constant((0., 1., 0.)) 


    w00s = a1[1]*a2[2] - a1[2]*a2[1]
    w10s = a1[2]*a2[0] - a1[0]*a2[2]
    w20s = a1[0]*a2[1] - a1[1]*a2[0]
  
    w01s = a3[1]*a4[2] - a3[2]*a4[1]
    w11s = a3[2]*a4[0] - a3[0]*a4[2]
    w21s = a3[0]*a4[1] - a3[1]*a4[0]

    w00 = w00s/sqrt(w00s**2+w10s**2+w20s**2)
    w10 = w10s/sqrt(w00s**2+w10s**2+w20s**2)
    w20 = w20s/sqrt(w00s**2+w10s**2+w20s**2)

    w01 = w01s/sqrt(w01s**2+w11s**2+w21s**2)
    w11 = w11s/sqrt(w01s**2+w11s**2+w21s**2)
    w21 = w21s/sqrt(w01s**2+w11s**2+w21s**2)

    w02s = w10*w21 - w20*w11
    w12s = w20*w01 - w00*w21
    w22s = w00*w11 - w10*w01

    w02 = w02s/sqrt(w02s**2+w12s**2+w22s**2)
    w12 = w12s/sqrt(w02s**2+w12s**2+w22s**2)
    w22 = w22s/sqrt(w02s**2+w12s**2+w22s**2) 

    wn = ([w00, w01, w02],[w10, w11, w12],[w20, w21, w22])
    wn = as_tensor(wn)

    wn_tr = ([w00,w10, w20],[w01,w11, w21], [w02, w12, w22])
    wn_tr = as_tensor(wn_tr)

    v00 = conditional(lt(v00,0.0),v00,0.0)
    v11 = conditional(lt(v11,0.0),v11,0.0)
    v22 = conditional(lt(v22,0.0),v22,0.0)

    vn = ([v00,0.0,0.0],[0.0,v11,0.0],[0.0,0.0,v22])
    vn = as_tensor(vn)  
    return wn*vn*wn_tr

def psi_star(u):
    return 0.5*lmbda*(0.5*(tr(epsilon_star(u))+abs(tr(epsilon_star(u)))))**2+mu*tr((strn_p_star(u))*(strn_p_star(u)))

def H_star(uold_star,unew_star,Hold_star):
    return conditional(lt(psi(uold_star),psi(unew_star)),psi(unew_star),Hold_star)

# Boundary conditions

top = CompiledSubDomain("near(x[2], 1.0) && on_boundary")
bot = CompiledSubDomain("near(x[2], 0.0) && on_boundary")
back = CompiledSubDomain("near(x[1], 0.0) && on_boundary")
front = CompiledSubDomain("near(x[1], 1.0) && on_boundary")
right = CompiledSubDomain("near(x[0], 1.0) && on_boundary")
left  = CompiledSubDomain("near(x[0], 0.0) && on_boundary")

#def left1(x, on_boundary):
#    return abs(x[1]-4.5) < 1e-01 and on_boundary
load = Expression("t", t = 0.0, degree=1)

bctop =  DirichletBC(W.sub(2), load, top ) 
bcbot=  DirichletBC(W.sub(2), Constant(0.0), bot)  
#bcback =  DirichletBC(W.sub(1), Constant(0.0), back ) 
#bcfront=  DirichletBC(W.sub(1), Constant(0.0), front) 
#bcright =  DirichletBC(W.sub(0), Constant(0.0), right ) 
#bcleft=  DirichletBC(W.sub(0), Constant(0.0), left) 

#bc_u = [bcbot, bctop, bcright, bcleft, bcback, bcfront] 
bc_u = [bcbot, bctop] 

bot.mark(fd,1)

def sigma(u):
    return 2.0*mu*epsilon(u)+lmbda*tr(epsilon(u))*Identity(len(u))

def sigma_p(u):
    return 2.0*mu*strn_p(u)+lmbda*(0.5*(tr(epsilon(u))+abs(tr(epsilon(u)))))*Identity(len(u))  

def sigma_n(u):
    return 2.0*mu*strn_n(u)+lmbda*(0.5*(tr(epsilon(u))-abs(tr(epsilon(u)))))*Identity(len(u)) 
    
def sigma_star(u):
    return 2.0*mu*epsilon_star(u)+lmbda*tr(epsilon_star(u))*Identity(len(u))

def sigma_p_star(u):
    return 2.0*mu*strn_p_star(u)+lmbda*(0.5*(tr(epsilon_star(u))+abs(tr(epsilon_star(u)))))*Identity(len(u))  

def sigma_n_star(u):
    return 2.0*mu*strn_n_star(u)+lmbda*(0.5*(tr(epsilon_star(u))-abs(tr(epsilon_star(u)))))*Identity(len(u)) 

x = SpatialCoordinate(mesh)

du = TrialFunction(W) 
v = TestFunction(W)

# Variational form
unew, uold, unew_star, uold_star = Function(W), Function(W), Function(W), Function(W)

# set up solution functions

u = Function(W,name='displacement')

# The way the eigenvalues are computed we cannot allow a constant value of u at start

u_array = u.vector().get_local()
u_array = np.random.rand(len(u_array))
u.vector()[:] = u_array

#solver_disp = LinearVariationalSolver(p_disp)

Traction_vite = Expression("t", t = 0.0, degree=1)

F_u = (((1.0-pold)**2+kappa)*inner(grad(v),sigma_p(u))+inner(grad(v),sigma_n(u)))*dx(2)+(((1.0-pold)**2+kappa)*inner(grad(v),sigma_p_star(u))+inner(grad(v),sigma_n_star(u)))*dx(1) 
J_u = derivative(F_u, u, du ) 
p_disp = NonlinearVariationalProblem(F_u, u, bc_u, J_u)
solver_disp  = NonlinearVariationalSolver(p_disp)

prm = solver_disp.parameters
prm["newton_solver"]["absolute_tolerance"] = 1E-2
prm["newton_solver"]["relative_tolerance"] = 1E-2
'''prm["newton_solver"]["maximum_iterations"] = 50
prm["newton_solver"]["relaxation_parameter"] = 1.0
if iterative_solver:
prm["linear_solver"] = "gmres"
prm["preconditioner"] = "ilu"
prm["krylov_solver"]["absolute_tolerance"] = 1E-3
prm["krylov_solver"]["relative_tolerance"] = 1E-3
prm["krylov_solver"]["maximum_iterations"] = 1000
prm["krylov_solver"]["gmres"]["restart"] = 40
prm["krylov_solver"]["preconditioner"]["ilu"]["fill_level"] = 0
set_log_level(PROGRESS)
'''
iterative_solver = True

prm["newton_solver"]["linear_solver"] = "gmres"
prm['newton_solver']['preconditioner'] = 'hypre_euclid'
'''
prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-9
prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-7
prm['newton_solver']['krylov_solver']['maximum_iterations'] = 1000
prm['newton_solver']['krylov_solver']['monitor_convergence'] = True
prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True
prm['newton_solver']['krylov_solver']['gmres']['restart'] = 40
#prm['newton_solver']['krylov_solver']['preconditioner']['same_nonzero_pattern'] = True
prm['newton_solver']['krylov_solver']['preconditioner']['ilu']['fill_level'] = 0
PROGRESS = 16
info(prm, True)
set_log_level(PROGRESS)
'''

E_phi = (Gc*l*inner(grad(p),grad(q))+((Gc/l)+2.0*H(uold,unew,Hold))*inner(p,q)-2.0*H(uold,unew,Hold)*q)*dx(2)+(Gc*l*inner(grad(p),grad(q))+((Gc/l)+2.0*H_star(uold_star,unew_star,Hold_star))*inner(p,q)-2.0*H_star(uold_star,unew_star,Hold_star)*q)*dx(1)
p_phi  = LinearVariationalProblem(lhs(E_phi), rhs(E_phi),pnew)
solver_phi  = LinearVariationalSolver(p_phi)

# Initialization of the iterative procedure and output requests
t = 0
u_r1 = -1.0*1e-4
u_r2 = -3.0*1e-7
deltaT  = 1.0

conc_f  = File ("./Vertebra_tutto_fine_gmres01/phi.pvd")
fname = open('ForcevsDisp_tutto_fine_gmres01.txt', 'w')
file = File("./Vertebra_tutto_fine_gmres01/mesh1.pvd") 

T1 = 10
T2 = 0
Tmax=T1+T2

# Staggered scheme
while t<=Tmax:
    t += deltaT
    if(t<=T1):
       load.t+=u_r1
    if(t>T1):
       load.t+=u_r2

    solver_disp.solve()

    unew.assign(u)
    uold.assign(unew)
    
    unew_star.assign(u)
    uold_star.assign(unew_star)

    solver_phi.solve()

    pold.assign(pnew)

    Hold.assign(project(psi(unew), WW))
    
    Hold_star.assign(project(psi_star(unew_star), WW))
    
	
    print ('Iterations:', iter, ', Total time', t)

    if (t % 1) == 0: # stampa ogni 20 timestep
        conc_f << pnew
    
        umean = (unew[2]+unew_star[2])*ds(1)

        Traction = dot(sigma(unew)+sigma(unew_star),n)
        fz = Traction[2]*ds(1)
        fname.write(str(assemble(umean)) + "\t")
        fname.write(str(assemble(fz)) + "\n")

        print ('traction:', iter, ', Trac', str(assemble(fz)))

        mesh1 = Mesh(mesh)
        X = mesh1.coordinates()
        X += np.vstack(map(u, X))
        file << mesh1
	    	    
fname.close()
print ('Simulation completed') 