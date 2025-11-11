# -*- coding: utf-8 -*-
import numpy as np 
from scipy.sparse import lil_matrix
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
from AWEfns import AWEvals
from FVMfns1 import fn4,fn1

def irregfl(phi, n):
    mask = np.zeros_like(phi, dtype=bool) #assumes that the interface does not extend
    for i in range(1, n-1): #beyond the first nd last point
        for j in range(1, n-1):
            center = phi[i, j]
            neighbors = [phi[i+1,j], phi[i-1,j], phi[i,j+1], phi[i,j-1]]
            if any(center * nb < 0 for nb in neighbors):
                mask[i, j] = True
    return mask

def stencilind(i0,j0,ii0,jj0):
    iar = np.array([i0-1,i0,i0+1,i0,i0,ii0])
    jar = np.array([j0,j0,j0,j0-1,j0+1,jj0])
    
    return iar,jar

def projdebug(R,X,Y,phi,rad):
    kappa  = np.ones_like(phi)*(1/R)
    nx = X/R
    ny = Y/R
    alpha = -phi
    xs = X + alpha*nx
    ys = Y + alpha*ny
    
    return nx,ny,xs,ys,alpha,kappa                

def proj(n,dx,phidisc,X,Y):
    phi    = gaussian_filter(phidisc, sigma=1) #get rid of this shit asap
    alpha  = np.zeros_like(phi)
    dphidx = np.zeros_like(phi)
    dphidy = np.zeros_like(phi)
    kappa  = np.zeros_like(phi)
    pxx = np.zeros_like(phi)
    pyy = np.zeros_like(phi)
    pxy = np.zeros_like(phi)
    pnn = np.zeros_like(phi)
    nx = np.zeros_like(phi)
    ny = np.zeros_like(phi)
    xs = np.zeros_like(phi)
    ys = np.zeros_like(phi)
    eps= 1e-12
    for i in range(1, n-1):
        for j in range(1, n-1):
            dphidx[i, j] = (phi[i+1, j] - phi[i-1, j])/(2*dx)
            dphidy[i, j] = (phi[i, j+1] - phi[i, j-1])/(2*dx)
            norm         = np.sqrt(dphidx[i, j]**2 + dphidy[i, j]**2)
            if(norm < eps): #undefined/ unstable stuff
                nx[i, j]  = 0
                ny[i, j]  = 0
                kappa[i,j]= 0
                alpha[i,j]= 0
            else:
                nx[i, j]     = dphidx[i,j]/norm
                ny[i, j]     = dphidy[i,j]/norm
                pxx[i,j]     = (phi[i+1,j]-2*phi[i,j]+phi[i-1,j])/dx**2
                pyy[i,j]     = (phi[i,j+1]-2*phi[i,j]+phi[i,j-1])/dx**2
                pxy[i,j]     = (phi[i+1,j+1]-phi[i-1,j+1]-phi[i+1,j-1]+phi[i-1,j-1])/(4*dx**2)
                kapnum       = (pxx[i,j]*dphidy[i,j]**2-2*pxy[i,j]*dphidx[i,j]*dphidy[i,j]+pyy[i,j]*dphidx[i,j]**2)
                kapden       = norm**3
                kappa[i,j]   = kapnum/kapden
                pnn[i,j]     = pxx[i,j]*nx[i,j]**2+2*nx[i,j]*ny[i,j]*pxy[i,j]+pyy[i,j]*ny[i,j]**2 
                discrim  = norm**2 - 2*pnn[i,j]*phi[i,j] #discriminant of quadratic equation for alpha
                if(abs(pnn[i,j])>eps and discrim>=0): #second order alpha possible, real roots of alpha exist
                        alpha1    = (-norm + np.sqrt(discrim))/pnn[i,j]
                        alpha2    = (-norm - np.sqrt(discrim))/pnn[i,j]
                        alpha[i,j]= alpha1 if abs(alpha1) < abs(alpha2) else alpha2                
                else: #resort to first order taylor expansion 
                        alpha[i,j]=-phi[i,j]/norm 
            
            xs[i,j]   = X[i,j] + alpha[i,j]*nx[i,j]
            ys[i,j]   = Y[i,j] + alpha[i,j]*ny[i,j]
            
    return nx,ny,xs,ys,alpha,kappa                                       
                         
def kapinterp(n,kapc,x,y,fmat,xs,ys):
    interp = RectBivariateSpline(x, y, kapc)
    kapint = np.ones_like(kapc)*0
    for i in range(1, n-1):
        for j in range(1, n-1):
            if fmat[i, j]:
                kapint[i, j] = interp(xs[i, j], ys[i, j])[0][0]
    return kapint              
            
def Sfn(phi):
    S = np.zeros_like(phi)
    S[phi>0] = 1 
    S[phi<0] = -1
    
    return S

def qanjfn(rad,R,bm,bp,j):
    a1 = 1
    b1 = 0.5
    c1 = 0
    a2 = bm/bp    + j/(bp*R)
    b2 = bm/(2*bp)- j/(4*bp*R**3)
    c2 = (1-bm/bp)*R**2 + 0.5*(1-bm/bp)*R**4 - 3*j*R/(4*bp)
    qan = np.where(rad<R,a1*rad**2+b1*rad**4+c1,a2*rad**2+b2*rad**4+c2)
    
    return qan,a1,b1,a2,b2
    
def idxfn(n):
    matind = np.ones([n,n],dtype=int)*-1
    for i in range(1,n-1):
        for j in range(1,n-1):
            matind[i,j] = (i-1)*(n-2) + (j-1)
    return matind

def Cij(n,fj,unj,alpha,cur,S,dx,fmat):
    C = np.zeros_like(alpha)
    for i in range(1,n-1):
        for j in range(1,n-1):
            if not fmat[i,j]:
                continue #skip regular
            nb = [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]
            for (l,k) in nb:
                if(S[l,k]*S[i,j]<0):
                    al = alpha[l,k]
                    clk = S[i,j]*(0.5*fj*al**2 -(al+0.5*cur*al**2)*unj)/dx**2
                    # clk = 1*(0.5*fj*al**2 -(al+0.5*cur*al**2)*unj)/dx**2
                    C[i,j] += clk
    return C


def sysmatIIM(n,dx,matind,f,C,phi,qan):
    nint = (n - 2)**2
    A = lil_matrix((nint, nint))
    rhs = np.zeros(nint)
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            row = matind[i, j]
            # print('row',row)
            rhs[row] = f[i, j] - C[i, j] # To make the code converge, either - Cij is necessary or S=S*-1. The latter messes with extrapol
            A[row, row] = -4/dx**2
            if i < n - 2:
                A[row,matind[i+1, j]]   = 1/dx**2
            else:
                rhs[row]               -= qan[i+1,j]/dx**2
            if i > 1:
                A[row, matind[i-1, j]]  = 1/dx**2
            else:
                rhs[row]               -= qan[i-1,j]/dx**2 
            if j < n - 2:
                A[row, matind[i, j+1]]  =  1/dx**2
            else:
                rhs[row]               -= qan[i,j+1]/dx**2
            if j > 1:
                A[row, matind[i, j-1]]  =  1/dx**2
            else:
                rhs[row]               -= qan[i,j-1]/dx**2
            

    return A.tocsr(),rhs

        
def edgeid(phi,n,fmat):
    E_id   = np.full_like(phi, -1, dtype=int)
    N_id   = np.full_like(phi, -1, dtype=int)
    npsi = 0
    for i in range(1,n-1):
        for j in range(1,n-1):
            if not fmat[i,j]:
                continue 
            if(E_id[i,j]==-1 and phi[i,j]*phi[i+1,j]<0):
                E_id[i,j]=npsi
                npsi    +=1
            if(N_id[i,j]==-1 and phi[i,j]*phi[i,j+1]<0):
                N_id[i,j]=npsi
                npsi    +=1
                
    return E_id,N_id,npsi

def extrap_stencils_pos(n, phi, E_id, N_id, npsi):
    i_owner = np.empty(npsi, dtype=int)
    j_owner = np.empty(npsi, dtype=int)
    st_nodes = [[] for _ in range(npsi)]
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            pid = E_id[i, j]
            if pid >= 0:
                if phi[i, j] > 0:
                    i_owner[pid], j_owner[pid] = i, j
                else:
                    i_owner[pid], j_owner[pid] = i + 1, j
            pid = N_id[i, j]
            if pid >= 0:
                if phi[i, j] > 0:
                    i_owner[pid], j_owner[pid] = i, j
                else:
                    i_owner[pid], j_owner[pid] = i, j + 1
    for pid in range(npsi):
        i0, j0 = i_owner[pid], j_owner[pid]
        for p in range(i0 - 4, i0 + 5):
            if p < 1 or p >= n - 1:
                continue
            for q in range(j0 - 4, j0 + 5):
                if q < 1 or q >= n - 1:
                    continue
                if phi[p, q] > 0:
                    st_nodes[pid].append((p, q))
    return st_nodes, i_owner, j_owner

def extrap_stencils_neg(n, phi, E_id, N_id, npsi):
    i_owner = np.empty(npsi, dtype=int)
    j_owner = np.empty(npsi, dtype=int)
    st_nodes = [[] for _ in range(npsi)]
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            pid = E_id[i, j]
            if pid >= 0:
                if phi[i, j] < 0:
                    i_owner[pid], j_owner[pid] = i, j
                else:
                    i_owner[pid], j_owner[pid] = i + 1, j
            pid = N_id[i, j]
            if pid >= 0:
                if phi[i, j] < 0:
                    i_owner[pid], j_owner[pid] = i, j
                else:
                    i_owner[pid], j_owner[pid] = i, j + 1
    for pid in range(npsi):
        i0, j0 = i_owner[pid], j_owner[pid]
        for p in range(i0 - 4, i0 + 5):
            if p < 1 or p >= n - 1:
                continue
            for q in range(j0 - 4, j0 + 5):
                if q < 1 or q >= n - 1:
                    continue
                if phi[p, q] < 0:
                    st_nodes[pid].append((p, q))
    return st_nodes, i_owner, j_owner              

def extrapol2(x,y,st_nodes,i_owner,j_owner,npsi,xs,ys,nx,ny,kel,G,m,Lparam,mode):
    # print('L',L)
    nxint = RectBivariateSpline(x, y, nx)
    nyint = RectBivariateSpline(x, y, ny)  
    M_list = []
    j_list = []
    stlist = []
    for pid in range(npsi):
      rows = []
      for (p,q) in st_nodes[pid]:
          xp,yq = x[p],y[q]
          rows.append([1,xp,yq,xp**2,xp*yq,yq**2,xp**3,yq*xp**2,xp*yq**2,yq**3])
      A     = np.array(rows)
      xsp   = xs[i_owner[pid], j_owner[pid]]
      ysq   = ys[i_owner[pid], j_owner[pid]]
      norm  = np.sqrt(nxint(xsp,ysq)[0][0]**2+nyint(xsp,ysq)[0][0]**2)
      nxs   = nxint(xsp,ysq)[0][0]/norm
      nys   = nyint(xsp,ysq)[0][0]/norm
      L     = np.array([0,nxs,nys,2*nxs*xsp,nxs*ysq+nys*xsp,2*nys*ysq,3*nxs*xsp**2,2*nxs*xsp*ysq+nys*xsp**2,nxs*ysq**2+2*nys*xsp*ysq,3*nys*ysq**2])
      M_pid = L @ np.linalg.pinv(A)
      M_list.append(M_pid) 
      jump  = jumpfn(xsp,ysq,kel,G,m,Lparam,mode)
      j_list.append(jump)
      stlist.append(np.array([xsp,ysq]))
      # u_vals = np.array([qan[p,q] for (p,q) in st_nodes[pid]])
      # un_plus.append(M_pid @ u_vals)
    return M_list,np.array(j_list),stlist
            
def assemble_B(n,npsi,matind,M_list,st_nodes):
    B = np.zeros([npsi,(n-2)**2])
    for p in range(npsi):
        indcs= np.array(st_nodes[p])
        cols = matind[indcs[:,0],indcs[:,1]]
        B[p,cols] = M_list[p]
        
    return B

def IIM_schur(n,matind,f,dx,S,alpha,xs,ys,kapint,fmat,G,m,L,mode,kel,E_id,N_id,npsi,dv,nv,bp): 
    nint   = (n - 2)**2
    inv_h2 = 1.0/dx**2 
    A = lil_matrix((nint, nint))
    E = lil_matrix((nint, npsi))
    rhs = np.zeros(nint)
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            row      = matind[i, j]
            rhs[row] = f[i, j] 
            A[row, row] = -4/dx**2
            if i < n - 2:
                A[row,matind[i+1, j]]   = 1/dx**2
            else:
                A[row,matind[1,j]]      = 1/dx**2
            if i > 1:
                A[row, matind[i-1, j]]  = 1/dx**2
            else:
                A[row,matind[n-2,j]]    = 1/dx**2
            if j < n - 2:
                A[row, matind[i, j+1]]  = 1/dx**2
            else:
                A[row,row]             += 1/dx**2
                rhs[row]               += nv/dx/bp #source term is scaled with k, so nv/k required
                #this assumes that the flux is kdT/dx and not -kdT/dx?
            if j > 1:
                A[row, matind[i, j-1]]  =  1/dx**2
            else:
                A[row,row]             -= 1/dx**2
                rhs[row]               -= 2*dv/dx**2
            if(fmat[i,j]):
                for l,k in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
                    if(S[l,k]*S[i,j]>=0):
                        continue 
                    al   = alpha[l,k]
                    cur  = kapint[l,k]
                    fj   = 0*jumpsource(xs[l,k],ys[l,k],kel,bp,G,m,L,mode) #xst,yst,kel,bp,G,m,L,mode
                    rhs[row] -= S[i,j]*(0.5*fj*al**2)*inv_h2
                    coeff     =-S[i,j]*(al+0.5*cur*al**2)*inv_h2
                    if   l==i-1: #map back to an edge, one irregular edge has one jump scalar
                        pid   = E_id[i-1,j]
                    elif l==i+1:
                        pid   = E_id[i,j]
                    elif k==j-1:
                        pid   = N_id[i,j-1]
                    elif k==j+1:
                        pid   = N_id[i,j]
                    else:
                        print('E error')
                    E[row,pid] += coeff #ideally = instead of +=                          
                        
    return A.tocsr(), E.tocsr(), rhs

def IIM_schur2(n,matind,f,dx,S,kapint,fmat,fj,dv,nv):
    nint   = (n - 2)**2
    A = lil_matrix((nint, nint))
    rhs = np.zeros(nint)
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            row      = matind[i, j]
            rhs[row] = f[i, j] 
            A[row, row] = -4/dx**2
            if i < n - 2:
                A[row,matind[i+1, j]]   = 1/dx**2
            else:
                A[row,matind[1,j]]      = 1/dx**2
            if i > 1:
                A[row, matind[i-1, j]]  = 1/dx**2
            else:
                A[row,matind[n-2,j]]    = 1/dx**2
            if j < n - 2:
                A[row, matind[i, j+1]]  =  1/dx**2
            else:
                A[row,row]             += 1/dx**2
                rhs[row]               -= nv/dx
            if j > 1:
                A[row, matind[i, j-1]]  =  1/dx**2
            else:
                A[row,row]             -= 1/dx**2
                rhs[row]               -= 2*dv/dx**2

    return A.tocsr(), rhs

def jumpfn(xst,yst,kel,G,m,L,mode):
    'Calculates the magnitude of jump. ival=kel*fn_(m,G) because electric potential is solved by sending kel to rhs'
    Pelt = -0.249 #V
    if mode==1:
     ival = fn1(m,G,xst,yst,L)
    elif mode==4:
     ival = fn4(m,G,xst,yst)
    else:
     ival = 0
     print('wrong mode')
    cden    =-kel*ival #negative value needs to me manually enforced. Figure out the -negative cden stuff
    eta,ibv = AWEvals(cden)
    jump    = -(eta+Pelt)*cden  
    return jump

def jumpsource(xst,yst,kel,bp,G,m,L,mode):
    if mode==1:
        ival = fn1(m,G,xst,yst,L)
    elif mode==4:
        ival = fn4(m,G,xst,yst)
    else:
        ival = 0
        print('wrong mode')
    fj = -ival**2/(bp*kel) #- sign because source term in our formulation is -Q/cond.
    return fj 