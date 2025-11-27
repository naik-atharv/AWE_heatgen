# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import lil_matrix

def bcval(kel,lelec,L,ibv,frac): #is not non-dimensional yet!!
    'frac = size of the additional domain/size of computational domain'
    iout = ibv*lelec/L  #avg current leaving the domain
    Qv   = (iout**2)/kel#avg ohmic heating/m^3
    vol  = frac*L*L     #volume of the extended domain
    qin  =-Qv*vol/L     #Heat flux into the computational domain, -ve because oriented in dirn opposite to axis
    dv = np.array([0,298]) #Dirichlet values of Electric Potential [0] and Temperature [1]
    nv = np.array([0,qin]) #Neumann values of Electric Potential and [0] Temperature [1]
    return dv,nv

def redim(phiL,deltaphi,phel): #re-dimensionalize the non-dimensional field
    return deltaphi*phel + phiL
    
def sdfbump(X,Y,x0,y0,R):
    phi_flat = Y-y0
    phi_bump = np.sqrt((X - x0)**2 + (Y - y0)**2) - R
    phi = np.minimum(phi_flat,phi_bump)
    
    return phi 
    
def chi_smooth(phi,dx,nsm):
    eps = nsm*dx
    chi = np.zeros_like(phi)
    inside  = phi < -eps
    outside = phi > eps
    smear   = ~inside & ~outside  # transition zone

    chi[inside] = 1.0
    chi[smear]  = 1 - 0.5 * (
        1 + phi[smear] / eps + (1 / np.pi) * np.sin(np.pi * phi[smear] / eps) )
    return chi

def sysmatVP(n,dx,matind,keff,dv,nv,f):
    nint  = (n - 2)**2
    A     = lil_matrix((nint,nint))
    rhs   = np.zeros(nint) #no source term 
    keffx = (keff[1:,:]+keff[:-1,:])*0.5
    keffy = (keff[:,1:]+keff[:,:-1])*0.5
    for j in range(1,n-1):
        for i in range(1,n-1):
            row = matind[i,j]
            ke  = keffx[i,j]
            kw  = keffx[i-1,j]
            kn  = keffy[i,j]
            ks  = keffy[i,j-1]
            A[row,row] = (ks+kn+ke+kw)/dx**2
            rhs[row]   = f[i,j]
            if i < n-2:
                A[row,matind[i+1,j]] = -ke/dx**2
            else:
                A[row,matind[1,j]]   = -ke/dx**2
            if i > 1:
                A[row,matind[i-1,j]] = -kw/dx**2
            else:
                A[row,matind[n-2,j]] = -kw/dx**2
            if j < n-2:
                A[row,matind[i,j+1]] = -kn/dx**2
            else:
                A[row,row]          += kn/dx**2
                rhs[row]            += 2*kn*dv/dx**2
            if j > 1:
                A[row,matind[i,j-1]] = -ks/dx**2
            else:
                A[row,row]          -= ks/dx**2
                rhs[row]            -= ks*nv/dx
  
    return A.tocsr(), rhs

def rhsVP_ND(n,gi,phi,chi,dx):
    "Since current cons. has no source term, creating f here itself"
    f      = np.zeros_like(phi)
    #Inner boundary condition
    chiix  = (chi[:,1:]+chi[:,:-1])*0.5
    chiiy  = (chi[1:,:]+chi[:-1,:])*0.5
    dphiidx= (phi[:,1:]-phi[:,:-1])/dx
    dphiidy= (phi[1:,:]-phi[:-1,:])/dx
    chiibx = gi*chiix*dphiidx
    chiiby = gi*chiiy*dphiidy
    divchiib= np.zeros_like(phi)
    lapphii = np.zeros_like(phi)
    divchiib[1:-1,1:-1] += (chiibx[1:-1,1:]-chiibx[1:-1,:-1])/dx
    divchiib[1:-1,1:-1] += (chiiby[1:,1:-1]-chiiby[:-1,1:-1])/dx
    lapphii[1:-1,1:-1]   = (-4*phi[1:-1,1:-1]+phi[1:-1,2:]+phi[1:-1,:-2]+phi[2:,1:-1]+phi[:-2,1:-1])/dx**2

    rhs = (1-chi)*f + divchiib -gi*chi*lapphii 

    return rhs   

def cdcalc(soln,phi,kel,n,dx):
    'Calculates the current density from the electric potential'
    ix = np.zeros_like(phi)
    iy = np.zeros_like(phi)
    for i in range(1,n-1):
        for j in range(1,n-1):
            if(phi[i,j]<0):
                continue
            else:
                ix[i,j] = -kel*(soln[i+1,j]-soln[i-1,j])/(2*dx)
                iy[i,j] = -kel*(soln[i,j+1]-soln[i,j-1])/(2*dx)
    
    return ix,iy

def qcalc(soln,phi,bm,bp,n,dx):
    'Calculates the heat flux from the Temperature'
    qx = np.zeros_like(phi)
    qy = np.zeros_like(phi)
    for i in range(1,n-1):
        for j in range(1,n-1):
                k = bm if phi[i,j]<0 else bp
                qx[i,j] = -k*(soln[i+1,j]-soln[i-1,j])/(2*dx)
                qy[i,j] = -k*(soln[i,j+1]-soln[i,j-1])/(2*dx)
    
    return qx,qy

def bc_el(soln,dv,nv,n,dx,kel):
    'Imposes the boundary conditions for electric potential'
    soln[0,:]   = soln[n-2,:]           #periodic in i
    soln[n-1,:] = soln[1,:]             #periodic in i
    soln[:,0]   = soln[:,1] + nv*dx/kel #neumann at j=1
    soln[:,n-1] = 2*dv - soln[:,n-2]    #dirichlet bc at j=n-2
    
    return soln

def bc_T(soln,dv,nv,n,dx,k):
    'Imposes the boundary conditions for Temperature'
    soln[0,:]   = soln[n-2,:]           #periodic in i
    soln[n-1,:] = soln[1,:]             #periodic in i
    soln[:,0]   = 2*dv - soln[:,1]      #dirichlet bc at j=1 
    soln[:,n-1] = soln[:,n-2] - nv*dx/k #neumann at j=n-2
    
    return soln
 
def sdfbump2(X,Y,x0,y0,R,n):
      rad = np.sqrt((X-x0)**2 + (Y-y0)**2)
      phi = np.zeros_like(X)
      # print('in')
      for i in range(n):
          for j in range(n):
              if(Y[i,j]>y0): 
                if (rad[i,j]>R):
                  phi[i,j] = np.minimum(rad[i,j]-R,Y[i,j]-y0)
                else:
                  phi[i,j] = rad[i,j]-R
              else:
                  # phi[i,j] = -1
                  d1 = abs(Y[i,j]-y0)
                  d2 = np.sqrt(d1**2 + (X[i,j]+R)**2)
                  d3 = np.sqrt(d1**2 + (X[i,j]-R)**2)
                  d4 = rad[i,j] + R                  
                  phi[i,j] = -1*np.min(np.array([d1,d2,d3,d4])) if abs(X[i,j]-x0)>=R else -1*np.min(np.array([d2,d3,d4]))
     
      return phi   
    
    
def sdfbump3(X,Y,x0,y0,R,n):
      rad = np.sqrt((X-x0)**2 + (Y-y0)**2)
      phi = np.zeros_like(X)
      # print('in')
      for i in range(n):
          for j in range(n):
              if(Y[i,j]>y0): 
                if (rad[i,j]>R):
                  phi[i,j] = np.minimum(rad[i,j]-R,Y[i,j]-y0)
                else:
                  phi[i,j] = rad[i,j]-R
              else:
                  # phi[i,j] = -1
                  d1 = abs(Y[i,j]-y0)
                  d2 = np.sqrt(d1**2 + (X[i,j]+R)**2)
                  d3 = np.sqrt(d1**2 + (X[i,j]-R)**2)
                  d4 = rad[i,j] + R                  
                  phi[i,j] = -1*np.min(np.array([d1,d2,d3,d4])) 
     
      return phi   
  

def sdf_bump(X, Y, x0, y0, R):
    """
    Signed-distance function for a composite 1-D curve in 2-D:
        * left horizontal half-line  : (x ≤ x0−R , y = y0)
        * right horizontal half-line : (x ≥ x0+R , y = y0)
        * upper semicircle           : centre (x0,y0), radius R, θ∈[0,π]

    +ve above the entire curve, –ve below.
    Parameters
    ----------
    X, Y : ndarray, same shape
        Query grid.
    x0, y0 : float
        Centre of the semicircle.
    R : float
        Radius of the semicircle.

    Returns
    -------
    phi : ndarray, same shape as X
        Signed distances.
    """
    dx  = X - x0
    dy  = Y - y0
    rad = np.hypot(dx, dy)                 # distance to the centre

    # 1. Distance to the arc (only where the orthogonal projection lies on θ∈[0,π])
    theta      = np.arctan2(dy, dx)        # range (−π, π]
    on_arc_proj = (theta >= 0.0) & (theta <= np.pi)
    d_arc = np.where(on_arc_proj, np.abs(rad - R), np.inf)

    # 2. Distance to the left horizontal ray (x ≤ x0−R, y = y0)
    mask_L = dx <= -R
    d_left = np.where(mask_L,                # perpendicular hits the ray
                      np.abs(dy),
                      np.hypot(dx + R, dy))  # closest to the left end-point

    # 3. Distance to the right horizontal ray (x ≥ x0+R, y = y0)
    mask_R = dx >=  R
    d_right = np.where(mask_R,
                       np.abs(dy),
                       np.hypot(dx - R, dy))

    # 4. Unsigned distance = minimum of the three candidates
    d_unsigned = np.minimum(np.minimum(d_arc, d_left), d_right)

    # 5. Decide the sign -------------------------------------------------------
    #    “Above the curve” means:
    #       – y > y0       if x ≤ x0−R  or  x ≥ x0+R         (straight parts)
    #       – y > y_arc(x) if |x−x0| < R                     (over the bump)
    y_curve = np.empty_like(Y)
    y_curve[mask_L | mask_R] = y0
    inside = ~(mask_L | mask_R)
    # Clip for numerical safety when dx==±R
    y_curve[inside] = y0 + np.sqrt(np.maximum(R**2 - dx[inside]**2, 0.0))

    sign = np.where(Y > y_curve, 1.0, -1.0)

    # 6. Final signed distance
    return sign * d_unsigned
  