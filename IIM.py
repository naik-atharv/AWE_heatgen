# -*- coding: utf-8 -*-
'Automatic extrapolation based on sign of bp-bm'
'Curvature calculation is automated and safety added for alpha computation'
import os 
import numpy as np 
import matplotlib.pyplot as plt 
os.chdir(r'C:\Users\anaik1\OneDrive - Delft University of Technology\PhD\Year 1\NS codes\heat_gen_AWE2\ndim')
from IIMfns import irregfl,idxfn,proj,Sfn,Cij,sysmatIIM,extrapol2,edgeid,assemble_B,IIM_schur,qanjfn
from IIMfns import kapinterp,extrap_stencils_pos,extrap_stencils_neg,projdebug,IIM_schur2
from VPfns  import chi_smooth,sysmatVP,rhsVP_ND,sdfbump,bcval,bc_el,bc_T,qcalc,sdfbump2,sdfbump3,sdf_bump
from VPfns  import redim
from AWEfns import AWEvals,Tanal,dTfn,Tanal2
from scipy.sparse.linalg import spsolve 
from scipy.sparse import bmat, eye, csc_matrix
import matplotlib.colors as colors
from sdf import sdfcomp1,sdfcomp2,sdarc,sdseg,sdarc2,sdseg2,sdfcomp4,larc1,larc4
from plotting import cont3
from FVMfns1 import aperture,sysmat,FVMstuff,cdcalc1
# facarr    = np.array([0.05,0.1,0.2,0.3,0.4]) #0.1,0.2,0.3,0.4,0.5,0.6 #1,0.5,0.25,
# for fac in facarr:
fac       = 0.5
name = 'rounded_more' #cc #cv
n    = 1500 #total no of points including ghost cells
bm   = 72   #thermal conductivity of phi<0
bp   = 0.58   #0.58 #thermal conductivity of phi>0
br   = bm/bp
L    = 10**(-4)
dx   = L/(n-2)
x    = np.linspace(-(L/2)-dx/2,(L/2)+dx/2,n)/L
y    = x 
dx   = dx/L
X,Y  = np.meshgrid(x,y,indexing = 'ij')
offs = 1/1e3 #0.01*10**(-4) #offset to prevent mesh alignment
R    = 0.08+offs #round(0.08*L + offs, 6)
Rf   = fac*R
x0,y0= (0,-(0+offs)) #0
toff = 0 #np.arcsin(Rf/(R+Rf)) #(np.pi/4)*fac
ts   = 0*np.pi+toff
te   = 1*np.pi-toff
usdf,phi = sdfcomp4(X,Y,x0,y0,R,Rf) #np.minimum(df1,df2)
# phi = Y-y0
# phi = sdfcomp2(X,Y,x0,y0,R,ts,te)
# phi,usdf = sdfcomp1(X,Y,x0,y0,R,ts,te)
cont3(X*L,Y*L,phi*L,phi*L,'coolwarm',50,0,None) #Plot sdf
lelec = L #larc1(x0,y0,R,ts,te,L)  #perimeter of the electrode surface
# lelec = larc4(x0,y0,Rf,R,ts,te,toff,X,L) #cross check once more!!!
#%
fmat    = irregfl(phi,n)
matind  = idxfn(n) 
#Volume Penalization params:
gi         =-10*10**3           #BV current density, A/m^2
eta,ibv= AWEvals(gi)    #Overpotential, Electrical conductivity, BV current den.
kel        = 27.1 #S/m, 1M KOH
dv,nv      = bcval(kel,lelec,L,gi,0)          # Outer Boundary condition Values 
etaN       = 1e-07  #Penalization parameter
nsm        = 1      #Number of smear cells
G,m,mode   = 1,0,1  #1,1/12,4 #-gi/kel,L/12,4 #-gi/kel,0,1  #G is non dimensional current density
# G          = G*L/m    #Constant Current (A) case, for constant voltage comment it out
deltaphi   = gi*L/kel
## FVM params:
Lx,Ly   = aperture(n,-phi,dx,dx)
A,rhs   = sysmat(n,Lx,Ly,matind,dx,dx,dv[0],nv[0])
pm,flx,area,Aarr,rhs = FVMstuff(n,rhs,-phi,matind,x,y,dx,dx,R,G,m,x0,y0,L,mode)
mask    = np.where(Aarr>(1e-12)*dx**2,True,False) 
active  = np.asarray(matind[mask]).ravel() #ravel and np just for safety
rhs_act = rhs[active]
A_act   = A[np.ix_(active, active)]
#%
solnact = spsolve(A_act,rhs_act)
solnvec = np.ones((n-2)**2)*0
solnvec[active] = solnact

phel    = np.zeros_like(phi)
for i in range(1, n-1):
    for j in range(1, n-1):
        phel[i, j] = solnvec[matind[i, j]]
phel  = redim(0,deltaphi,phel) #dimensionalize the field
phel  = bc_el(phel,dv[0],nv[0],n,dx*L,kel) #boundary conditions
ix,iy = cdcalc1(phel,-phi,kel,n,dx*L)
cden       = np.hypot(ix,iy)

cont3(X*L,Y*L,np.ma.masked_where(phi<0,phel),phi*L,'viridis',50,1,'phi_'+name) #Plot potential
print('FVM done')
#%
Q     = (ix**2 + iy**2)/kel
jump  = -(eta-0.249)*gi     #Overpotential heating
deltaT= dTfn(n,round(y0*L+0.5*L,6),L,Q,bm,nv[1],-jump)
##Immersed Interface method
nx,ny,xs,ys,alp,kapc= proj(n,dx,phi,X,Y) #projdebug(R,X,Y,phi,rad)
alp[1:-1,1:-1]      = - phi[1:-1,1:-1]
kapint              = kapinterp(n,kapc,x,y,fmat,xs,ys)
f                   = (np.where(phi>0,-Q/bp,-Q/bm))*(L**2)/(deltaT) 
S                   = Sfn(phi)
##IIM stuff
E_id,N_id,npsi = edgeid(phi,n,fmat)
bj = 1 - br #non-dim
if(bj<0):
    st_nodes,i_owner,j_owner = extrap_stencils_pos(n,phi,E_id,N_id,npsi)
else:
    st_nodes,i_owner,j_owner = extrap_stencils_neg(n,phi,E_id,N_id,npsi)
M_list,jmp_array,stlst= extrapol2(x,y,st_nodes,i_owner,j_owner,npsi,xs,ys,nx,ny,kel,G*gi,m,L,mode) 

B = assemble_B(n,npsi,matind,M_list,st_nodes)
# A, rhs = IIM_schur2(n,matind,-Q,dx,S,kapint,fmat,fj,dv[1],nv[1]) #test bc
# solnvec = spsolve(A,rhs) #test bc
A,E,rhs = IIM_schur(n,matind,f,dx,S,alp,xs,ys,kapint,fmat,G*gi,m,L,mode,kel,E_id,N_id,npsi,dv[1]-dv[1],nv[1]*L/deltaT,deltaT,bp) #n,matind,f,dx,S,alpha,xs,ys,kapint,fmat,G,m,L,mode,kel,E_id,N_id,npsi,dv,nv,bp
# lam   = bm/(bp-bm) if bj < 0 else bp/(bp-bm) #dimensional
lam = br/(bj) if bj < 0 else 1/(bj)
B_sparse = csc_matrix(B) 
block_mat = bmat([
[A,         E],
[B_sparse, lam * eye(npsi, format='csc')]
    ], format='csc')
print('IIM begins')
#%
# rhs_total = np.concatenate([rhs, np.ones(npsi)*jump/(bj)]) #dimensional
rhs_total = np.concatenate([rhs, (jmp_array/bj)*(L/deltaT/bp)]) 
sol_total = spsolve(block_mat, rhs_total)
#%
solnvec = sol_total[:(n-2)**2]
soln    = np.zeros_like(phi)
for i in range(1, n-1):
    for j in range(1, n-1):
        soln[i, j] = solnvec[matind[i, j]]
soln  = redim(dv[1],deltaT,soln) #re-dimensionalize
soln       = bc_T(soln,dv[1],nv[1],n,dx,bp)
qx,qy      = qcalc(soln,phi,bm,bp,n,dx)
cden       = np.hypot(ix,iy)
qden       = np.hypot(qx,qy)

cont3(X*L,Y*L,soln-dv[1],phi*L,'coolwarm',100,2,'Temp_'+name) #Plot Temperature
np.savez('Results/backup/'+name+'.npz',X=X,Y=Y,phi=phi,phel=phel,kel=kel,n=n,dx=dx,dv=dv,nv=nv,soln=soln,bp=bp,bm=bm) #save
# np.savez('Results/backup/ccLo12_.npz',X=X,Y=Y,phi=phi,phel=phel,kel=kel,n=n,dx=dx,dv=dv,nv=nv,soln=soln,bp=bp,bm=bm)

#%%
del phi,fmat,matind,A,rhs,solnvec,phel,ix,iy,Q,Lx,Ly,Aarr,active,mask
del nx,ny,xs,ys,alp,kapc,kapint,S,E_id,N_id,st_nodes,i_owner,j_owner,M_list,B,E,B_sparse,rhs_total,sol_total,soln,qx,qy,cden,jmp_array,qden
del A_act,block_mat,f,rhs_act,solnact,X,Y

#%% Validation wrt analytical solution
'Use this to validate the case for a flat plate, where the phi=Y-y0'

# al       = round(y0+0.5*L,6) #for dimensional coordinates
al         = round(y0*L+0.5*L,6) #for non-dimensional coordinates
Tan        = Tanal2(n,x*L,dx*L,al,L,Q,bm,bp,dv[1],nv[1],-jump)
plt.figure()
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.plot(x[1:-1]*L,Tan[1:-1]-dv[1],'k-',label='Analytical')
# plt.plot(y0*L,deltaT,'r.',label='Interface')
plt.plot(x[1:-1]*L,soln[n//2,1:-1]-dv[1],'r--',label='Numerical') 
plt.title('Validation') 
plt.ylabel('T')
plt.xlabel('y')
plt.legend()
# plt.plot(xarr[1:-1],Tan[1:-1])
#%%
plt.figure()
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# plt.plot(x[1:-1],soln[n//2,1:-1])
# cden = np.hypot(ix,iy)**2/kel#/1e12
# qden = np.hypot(qx,qy)
# cden= soln-dv[1]
# plt.contourf(X[1:-1,1: -1]+L/2,Y[1:-1,1:-1]+L/2,soln[1:-1,1:-1].T-dv[1],levels=50,cmap='coolwarm')
# plt.colorbar()
# plt.contourf(X,Y,phi.T,levels=40)#,cmap='coolwarm'
# plt.colorbar()
# plt.contour(X[1:-1,1: -1]+L/2, Y[1:-1,1: -1]+L/2, phi[1:-1,1: -1].T, levels=[0], colors='black',linestyles='dotted')  
# plt.contourf(X[1:-1,1:-1],Y[1:-1,1:-1],phel[1:-1,1:-1].T,levels=20)
# plt.contourf(X[1:-1,1:-1],Y[1:-1,1:-1],qden[1:-1,1:-1].T,levels=50)
# plt.streamplot(x,y,ix,iy,density=1)
# plt.title(r'$T$ - $T_a$  ( K )')

# plt.contourf(X[1:-1,1:-1], Y[1:-1,1:-1],
#               (Q[1:-1,1:-1] + 1e-20).T,  # avoid log(0)
#               levels=50,
#                 norm=colors.LogNorm(vmin=1e-16, vmax=np.max(cden)))

plt.xlabel('z (m)')
plt.ylabel('x (m)')
# plt.title(r'Current density magnitude $(\; A/m^2 \; )$')
# plt.title(r'Ohmic heating rate $(\; W/m^3 \; )$')
plt.colorbar()
# plt.savefig('Presentation/bumps.png',dpi=300)

#%%
#%% Verify computation of jump 
from FVMfns1 import fn1,fn4
starr = np.array(stlst)
cpsi = np.ones(npsi)
for i in range(npsi):
    if(mode==1):
     cden =-kel*fn1(m,G,starr[i,0],starr[i,1],L)
     eta,ibv = AWEvals(cden)
     cpsi[i] =-(eta-0.249)*cden
    elif(mode==4):
     cden =-kel*fn4(m,G,starr[i,0],starr[i,1])
     eta,ibv = AWEvals(cden)
     cpsi[i] =-(eta-0.249)*cden