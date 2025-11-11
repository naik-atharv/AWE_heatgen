# -*- coding: utf-8 -*-
'Automatic extrapolation based on sign of bp-bm'
'Curvature calculation is automated and safety added for alpha computation'
import os 
import numpy as np 
import matplotlib.pyplot as plt 
os.chdir(r'C:\Users\anaik1\OneDrive - Delft University of Technology\PhD\Year 1\NS codes\heat_gen_AWE2')
from IIMfns import irregfl,idxfn,proj,Sfn,Cij,sysmatIIM,extrapol2,edgeid,assemble_B,IIM_schur,qanjfn
from IIMfns import kapinterp,extrap_stencils_pos,extrap_stencils_neg,projdebug,IIM_schur2
from VPfns  import chi_smooth,sysmatVP,rhsVP_ND,sdfbump,bcval,bc_el,bc_T,qcalc,sdfbump2,sdfbump3,sdf_bump
from AWEfns import AWEvals,Tanal
from scipy.sparse.linalg import spsolve 
from scipy.sparse import bmat, eye, csc_matrix
import matplotlib.colors as colors
from sdf import sdfcomp1,sdfcomp2,sdarc,sdseg,sdarc2,sdseg2,sdfcomp4,larc1,larc4
from plotting import contours1,contours2
from FVMfns1 import aperture,sysmat,FVMstuff,cdcalc1
# facarr    = np.array([0.05,0.1,0.2,0.3,0.4]) #0.1,0.2,0.3,0.4,0.5,0.6 #1,0.5,0.25,
fac       = 0
# llim,ulim = 500,1000  
# for fac in facarr:
n    = 500 #total no of points including ghost cells
bm   = 72   #thermal conductivity of phi<0
bp   = 0.58 #thermal conductivity of phi>0
# jump= 0
L    = 10**(-4)
dx   = L/(n-2)
x    = np.linspace(-(L/2)-dx/2,(L/2)+dx/2,n)
y    = x 
bj   = bp-bm
X,Y  = np.meshgrid(x,y,indexing = 'ij')
offs = 1e-6 #0.01*10**(-4) #offset to prevent mesh alignment
R    = round(0.08*L + offs, 6)
# Rf   = fac*R
x0,y0= (0,-(0+offs)) #0
# toff = np.arcsin(Rf/(R+Rf)) #(np.pi/4)*fac
# ts   = np.pi+toff
# te   = 2*np.pi-toff
# usdf,phi = sdfcomp4(X,Y,x0,y0,R,Rf) #np.minimum(df1,df2)
phi = Y-y0
# phi = sdfcomp2(X,Y,x0,y0,R,ts,te)
plt.figure()
plt.contour(X[:,:],Y[:,:],phi[:,:].T,levels=60,cmap='coolwarm')
plt.colorbar()
plt.contour(X[1:-1,1: -1],Y[1:-1,1: -1],phi[1:-1,1: -1].T,levels=[0],colors='black') #,linestyles='dotted'
plt.xlabel('z (m)')
plt.ylabel('x (m)')
plt.title(r'signed distance fn')
plt.show()
lelec = L #larc1(x0,y0,R,ts,te,L)  #perimeter of the electrode surface
# lelec = larc4(x0,y0,Rf,R,ts,te,toff,X,L) #cross check once more!!!
#%
fmat    = irregfl(phi,n)
matind  = idxfn(n) 
#Volume Penalization params:
gi         = -10*10**3              #BV current density, A/m^2
eta,ibv= AWEvals(gi)    #Overpotential, Electrical conductivity, BV current den.
kel        = 27.1 #S/m, 1M KOH
dv,nv      = bcval(kel,lelec,L,gi,fac*0)          #Boundary condition values 
etaN       = 1e-07  #Penalization parameter
nsm        = 1      #Number of smear cells
G,m,mode   = -gi/kel,L/12,4 #-gi/kel,L/12,4 
G          = G*L/m    #Constant Current (A) case, for constant voltage comment it out
## FVM params:
Lx,Ly   = aperture(n,-phi,dx,dx)
A,rhs   = sysmat(n,Lx,Ly,matind,dx,dx,dv[0],nv[0])
pm,flx,area,Aarr,rhs = FVMstuff(n,rhs,-phi,matind,x,y,dx,dx,R,G,m,x0,y0,L,mode)
mask    = np.where(Aarr>(1e-12)*dx**2,True,False) 
active  = np.asarray(matind[mask]).ravel() #ravel and np just for safety
rhs_act = rhs[active]
A_act   = A[np.ix_(active, active)]
#%%
solnact = spsolve(A_act,rhs_act)
solnvec = np.ones((n-2)**2)*0
solnvec[active] = solnact
# #Volume Penalization
# chi     = chi_smooth(phi,dx,nsm)
# keff    = kel*(1-chi)+etaN*(chi)
# f       = rhsVP_ND(n,gi,phi,chi,dx)
# A,rhs   = sysmatVP(n,dx,matind,keff,dv[0],nv[0],f)
# solnvec = spsolve(A,rhs)
phel    = np.zeros_like(phi)
for i in range(1, n-1):
    for j in range(1, n-1):
        phel[i, j] = solnvec[matind[i, j]]
phel  = bc_el(phel,dv[0],nv[0],n,dx,kel)
ix,iy = cdcalc1(phel,-phi,kel,n,dx)
plt.figure(),plt.contour(X[1:-1,1:-1],Y[1:-1,1:-1],phel[1:-1,1:-1],levels=50),plt.colorbar(),plt.title('Electric Potential (V)'),plt.savefig('cc_phiL12.png',dpi=300)
plt.show()
# plt.figure(),plt.plot(x,phel[n//2,:])
#%
Q     = (ix**2 + iy**2)/kel
jump  = -(eta-0.249)*gi       #Overpotential heating
##Immersed Interface method
nx,ny,xs,ys,alp,kapc= proj(n,dx,phi,X,Y) #projdebug(R,X,Y,phi,rad)
kapint              = kapinterp(n,kapc,x,y,fmat,xs,ys)
f                   = np.where(phi>0,-Q/bp,-Q/bm)
S                   = Sfn(phi)
# fj                  = 0 #-(gi**2)/(kel*bp)
##IIM stuff
E_id,N_id,npsi = edgeid(phi,n,fmat)
if(bj<0):
    st_nodes,i_owner,j_owner = extrap_stencils_pos(n,phi,E_id,N_id,npsi)
else:
    st_nodes,i_owner,j_owner = extrap_stencils_neg(n,phi,E_id,N_id,npsi)
M_list,jmp_array,stlst= extrapol2(x,y,st_nodes,i_owner,j_owner,npsi,xs,ys,nx,ny,kel,G,m,L,mode) 

B = assemble_B(n,npsi,matind,M_list,st_nodes)
# A, rhs = IIM_schur2(n,matind,-Q,dx,S,kapint,fmat,fj,dv[1],nv[1]) #test bc
# solnvec = spsolve(A,rhs) #test bc
A,E,rhs = IIM_schur(n,matind,f,dx,S,alp,xs,ys,kapint,fmat,G,m,L,mode,kel,E_id,N_id,npsi,dv[1],nv[1],bp) #n,matind,f,dx,S,alpha,xs,ys,kapint,fmat,G,m,L,mode,kel,E_id,N_id,npsi,dv,nv,bp
lam   = bm/(bp-bm) if bj < 0 else bp/(bp-bm)

B_sparse = csc_matrix(B) 
block_mat = bmat([
[A,         E],
[B_sparse, lam * eye(npsi, format='csc')]
    ], format='csc')
#%
# rhs_total = np.concatenate([rhs, np.ones(npsi)*jump/(bj)])
rhs_total = np.concatenate([rhs, jmp_array/bj])
sol_total = spsolve(block_mat, rhs_total)
#%
solnvec = sol_total[:(n-2)**2]
soln    = np.zeros_like(phi)
for i in range(1, n-1):
    for j in range(1, n-1):
        soln[i, j] = solnvec[matind[i, j]]
soln       = bc_T(soln,dv[1],nv[1],n,dx,bp)
qx,qy      = qcalc(soln,phi,bm,bp,n,dx)
cden       = np.hypot(ix,iy)
qden       = np.hypot(qx,qy)

gx,gy      = cdcalc1(phel,abs(phi),1,n,dx) #gradient of potential

plt.figure(),plt.contour(X[1:-1,1:-1],Y[1:-1,1:-1],soln[1:-1,1:-1]-dv[1],levels=50,cmap='coolwarm'),plt.colorbar(),plt.title(r'$T$-$T_a$ (K)'),plt.savefig('cc_TL12.png',dpi=300)
# plt.figure(),plt.contourf(X[1:-1,1:-1],Y[1:-1,1:-1],cden[1:-1,1:-1]-dv[1],levels=50,cmap='coolwarm'),plt.colorbar(),plt.title(r'$T$-$T_a$ (K)')
#%% Plotting routines
# if fac==facarr[0]: #fixing colorbar across plots
#     maxcol1 = np.max(phel)
#     # maxcol2 = np.max(phel[llim:ulim,llim:ulim])
#     # mincol2 = np.min(phel[llim:ulim,llim:ulim])
#     maxtemp = np.max(soln)-dv[1]
#     maxcden = np.max(cden)
# levels2 = (np.linspace(0, 1, 60))** 2 #scale this with maxval of contour before plotting
# levels1     = np.linspace(0, 1, 80)
# levels2     = np.linspace(0, 1, 30)
# maxtemp = np.max(soln)-dv[1] #comment after first sim
# maxcol1 = np.max(phel)       #comment after first sim
# # contours1(X,Y,phi,phel,cden,ix,iy,facarr,fac,maxcol1,phel_level)
# # contours2(X,Y,gx,gy,phel,phi,n,dx,facarr,fac,ulim,llim,maxcol2,mincol2)
np.savez('backup/ccLo12_.npz',X=X,Y=Y,phi=phi,phel=phel,kel=kel,n=n,dx=dx,dv=dv,nv=nv,soln=soln,bp=bp,bm=bm)

# plt.figure()
# plt.contour(X[1:-1,1: -1]*1e6,Y[1:-1,1:-1]*1e6,soln[1:-1,1:-1].T-dv[1],levels=80,cmap='coolwarm') #levels=levels1*maxtemp
# plt.colorbar()
# plt.contour(X[1:-1,1: -1]*1e6, Y[1:-1,1: -1]*1e6, phi[1:-1,1: -1].T, levels=[0], colors='black',linestyles='dotted')  
# plt.xlabel(r'z $( \mu m)$')
# plt.ylabel(r'x $( \mu m)$')
# plt.title(r'$T$ - $T_a$  ( K )')
# # #plt.savefig(r'Group_Meeting/rounded/Temp_dep_'+str(np.argmax(facarr/fac==1)+1)+'.png',dpi=300)
# plt.savefig('Plots/Temp3.png',dpi=300)
# plt.show()

# plt.figure()
# plt.contour(X[1:-1,1: -1]*1e6,Y[1:-1,1:-1]*1e6,phel[1:-1,1:-1].T,cmap='Spectral',levels=60) # levels=levels2*maxcol1
# plt.colorbar()
# plt.contour(X[1:-1,1: -1]*1e6, Y[1:-1,1: -1]*1e6, phi[1:-1,1: -1].T, levels=[0], colors='black',linestyles='dotted') 
# # plt.quiver(X[1:-1,1: -1]*1e6, Y[1:-1,1: -1]*1e6, ) 
# plt.xlabel(r'z $( \mu m)$')
# plt.ylabel(r'x $( \mu m)$')
# plt.title(r'Electric potential (V)')
# ##plt.savefig(r'Group_Meeting/rounded/phi_dep_'+str(np.argmax(facarr/fac==1)+1)+'.png',dpi=300)
# # plt.savefig('Plots/cosine_phi1.png',dpi=300)
# plt.show()

plt.figure()
plt.contourf(X[1:-1,1:-1]+L/2,Y[1:-1,1:-1]+L/2,cden[1:-1,1:-1],levels=200) #levels1*maxcden,vmax=maxcden/2,vmin=0
plt.colorbar()
plt.xlabel('z (m)')
plt.ylabel('x (m)')
plt.title(r'Current density magnitude $(\; A/m^2 \; )$')
# plt.savefig(r'Group_Meeting/rounded/cden_dep_'+str(np.argmax(facarr/fac==1)+1)+'.png',dpi=300)
plt.savefig('analysis/cv_cdenL4.png',dpi=300) #/const_volt
plt.show()
# # plt.figure()
# # plt.contourf(X[1:-1,1:-1]+L/2,Y[1:-1,1:-1]+L/2,ix[1:-1,1:-1].T,levels=60,cmap='coolwarm')
# # plt.colorbar()
# # plt.xlabel('z (m)')
# # plt.ylabel('x (m)')
# # plt.title(r'Current density x component $(\; A/m^2 \; )$')
# # # plt.savefig(r'rounded_plots/cdenx_'+str(np.argmax(facarr/fac==1)+1)+'.png',dpi=300)
# # plt.figure()
# # plt.contourf(X[1:-1,1:-1]+L/2,Y[1:-1,1:-1]+L/2,iy[1:-1,1:-1].T,levels=60,cmap='coolwarm')
# # plt.colorbar()
# # plt.xlabel('z (m)')
# # plt.ylabel('x (m)')
# # plt.title(r'Current density z component $(\; A/m^2 \; )$')
# # plt.savefig(r'rounded_plots/cdenz_'+str(np.argmax(facarr/fac==1)+1)+'.png',dpi=300)
# # plt.figure()
# # plt.contourf(X[1:-1,1:-1]+L/2,Y[1:-1,1:-1]+L/2,Q[1:-1,1:-1].T,levels=50)
# # plt.colorbar()
# # plt.xlabel('z (m)')
# # plt.ylabel('x (m)')
# # plt.title(r'Ohmic heating rate $(\; W/m^3 \; )$')
# ## plt.savefig(r'bump_diag_plot/Qohm_'+str(np.argmax(facarr/fac==1)+1)+'.png',dpi=300)
# del phi,chi,keff,fmat,matind,A,rhs,solnvec,phel,ix,iy,Q#
# del nx,ny,xs,ys,alp,kapc,kapint,S,E_id,N_id,st_nodes,i_owner,j_owner,M_list,B,E,B_sparse,rhs_total,sol_total,soln,qx,qy,cden
#%%
del phi,fmat,matind,A,rhs,solnvec,phel,ix,iy,Q,Lx,Ly,Aarr,active,mask
del nx,ny,xs,ys,alp,kapc,kapint,S,E_id,N_id,st_nodes,i_owner,j_owner,M_list,B,E,B_sparse,rhs_total,sol_total,soln,qx,qy,cden


#%% Validation wrt analytical solution
'Use this to validate the case for a flat plate, where the phi=Y-y0'

# n = 30
al       = round(y0+0.5*L,6)
Tan,xarr = Tanal(n,al,L,Q,bm,bp,dv[1],-jump)
plt.figure()
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.plot(xarr[1:-1]-0.5*L,Tan[1:-1],'b-',label='Analytical') #Tan,xarr = Tanal(100,al,L,Q,ks,kl,dv,C)
plt.plot(x[1:-1],soln[n//2,1:-1],'r--',label='Numerical') 
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