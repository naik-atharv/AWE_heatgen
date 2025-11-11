# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix

def aperture(n,phi,dx,dy):
    Lx  = np.zeros([n+1,n])
    Ly  = np.zeros([n,n+1])
    
    for i in range (1,n-1):
        for j in range(1,n-1):
            pimjm = 0.25*(phi[i,j]+phi[i-1,j]+phi[i-1,j-1]+phi[i,j-1]) #pimjm        
            pipjm = 0.25*(phi[i,j]+phi[i+1,j]+phi[i+1,j-1]+phi[i,j-1]) #pipjm        
            pipjp = 0.25*(phi[i,j]+phi[i+1,j]+phi[i+1,j+1]+phi[i,j+1]) #pipjp        
            pimjp = 0.25*(phi[i,j]+phi[i-1,j]+phi[i-1,j+1]+phi[i,j+1]) #pimjp
            if(pimjm>0 and pimjp>0):
                Lx[i,j] = 0
            elif(pimjm<0 and pimjp<0): 
               Lx[i,j]  = dy 
            elif(pimjm<0 and pimjp>0): 
               Lx[i,j]  = dy*(pimjm)/(pimjm-pimjp)
            elif(pimjm>0 and pimjp<0): 
               Lx[i,j]  = dy*(pimjp)/(pimjp-pimjm)
            
            if(pipjm>0 and pipjp>0):
               Lx[i+1,j] = 0
            elif(pipjm<0 and pipjp<0):
               Lx[i+1,j] = dy
            elif(pipjm<0 and pipjp>0):
               Lx[i+1,j] = dy*(pipjm)/(pipjm-pipjp)
            elif(pipjm>0 and pipjp<0):
               Lx[i+1,j] = dy*(pipjp)/(pipjp-pipjm)
               
            if(pimjm>0 and pipjm>0):
               Ly[i,j]  = 0  
            elif(pimjm<0 and pipjm<0):
               Ly[i,j]  = dx 
            elif(pimjm<0 and pipjm>0):
               Ly[i,j]  = dx*(pimjm)/(pimjm-pipjm)
            elif(pimjm>0 and pipjm<0):
               Ly[i,j]  = dx*(pipjm)/(pipjm-pimjm) 
            
            if(pimjp>0 and pipjp>0):
               Ly[i,j+1] = 0
            elif(pimjp<0 and pipjp<0):
               Ly[i,j+1] = dx
            elif(pimjp<0 and pipjp>0):
               Ly[i,j+1] = dx*(pimjp)/(pimjp-pipjp)
            elif(pimjp>0 and pipjp<0):
               Ly[i,j+1] = dx*(pipjp)/(pipjp-pimjp)   
               
    return Lx,Ly

def idxfn(n):
    matind = np.ones([n,n],dtype=int)*-1
    for i in range(1,n-1):
        for j in range(1,n-1):
            matind[i,j] = (i-1)*(n-2) + (j-1)
    return matind

def heron(a,b,c):
    s    = 0.5*(a+b+c)
    area = (s*(s-a)*(s-b)*(s-c))**0.5
    return area

def hyp(ar):
    return np.hypot(ar[0],ar[1])

def fncalc(Pint,G,R,m,x0,y0,L,mode): 
    if len(Pint)==0:
        favg = 0
    elif len(Pint)==2:
        P11 = Pint[0][0]
        P22 = Pint[1][0]
        t11 = np.arctan2(P11[1]-y0,P11[0]-x0)
        t22 = np.arctan2(P22[1]-y0,P22[0]-x0)
        if(mode==1):
         f11 = fn1(m,G,P11[0],P11[1],L)
         f22 = fn1(m,G,P22[0],P22[1],L)
        if (mode==2):
         f11 = fn2(m,G,R,t11) #m*G*(R**m)*np.cos(m*t11)/r11**(m+1)
         f22 = fn2(m,G,R,t22) #m*G*(R**m)*np.cos(m*t22)/r22**(m+1)
        elif (mode==3):
         f11 = fn3(G,P11[0],P11[1],t11)
         f22 = fn3(G,P22[0],P22[1],t22)
        elif (mode==4): #here m is the active width of the bc
         f11 = fn4(m,G,P11[0],P11[1])
         f22 = fn4(m,G,P22[0],P22[1])
        favg= 0.5*(f11+f22)      
    else:
        print('wtf')
    return favg

def fn1(m,G,x,y,L): #linear current density
    f = G + m*(1 - 2*abs(x)/L)    
    return f
    
def fn2(m,G,R,t): #Validation case
    return m*G*np.cos(m*t)/R

def fn3(G,x,y,t): #Validation case 
    return G*np.sin(G*x)*np.cosh(G*y)*np.cos(t)-G*np.cos(G*x)*np.sinh(G*y)*np.sin(t)

def fn4(m,G,x,y): #Cosine current density
    f = (G/2)*(1+np.cos(np.pi*abs(x)/m)) if abs(x)<=m else 0
    return f


def geometric_int(T,G):
    pos = []
    neg = []
    eps = 1e-14
    Pint= []
    for i in range(3):
        if(T[i][1]>=eps):
            pos.append(T[i])
        elif(T[i][1]<=-eps):
            neg.append(T[i])
    if(  len(pos)==3 and len(neg)==0):
        dA,dgam = 0.0,0.0
    elif(len(pos)==0 and len(neg)==3):
        a  = hyp(neg[0][0]-neg[1][0])
        b  = hyp(neg[1][0]-neg[2][0])
        c  = hyp(neg[2][0]-neg[0][0])
        dA,dgam = heron(a,b,c), 0.0
    elif(len(pos)==2 and len(neg)==1):
        s1 = (neg[0][1])/(neg[0][1]-pos[0][1])       
        s2 = (neg[0][1])/(neg[0][1]-pos[1][1])
        P11= (1-s1)*neg[0][0] + (s1)*pos[0][0]
        P22= (1-s2)*neg[0][0] + (s2)*pos[1][0]
        a  = hyp(neg[0][0]-P11)
        b  = hyp(neg[0][0]-P22)
        c  = hyp(P22-P11)
        dA,dgam = heron(a,b,c),c
        Pint.extend([[P11],[P22]])
    elif(len(pos)==1 and len(neg)==2): 
        s1 = (neg[0][1])/(neg[0][1]-pos[0][1])
        s2 = (neg[1][1])/(neg[1][1]-pos[0][1])
        P11= (1-s1)*neg[0][0] + (s1)*pos[0][0]
        P22= (1-s2)*neg[1][0] + (s2)*pos[0][0]
        a  = hyp(neg[0][0]-P22)
        b  = hyp(neg[1][0]-P22)
        c  = hyp(neg[0][0]-neg[1][0])
        d  = hyp(neg[0][0]-P22)
        e  = hyp(neg[0][0]-P11)
        f  = hyp(P22-P11)
        dA,dgam = heron(a,b,c)+heron(d,e,f),f
        Pint.extend([[P11],[P22]])
    else:
        print('wtf:lpos,lneg=',len(pos),len(neg))
    # fnavg = fncalc(Pint)
    fnavg = G
    return dA,dgam,fnavg

def geometric_int2(T,G,R,m,x0,y0,L,mode): 
    pos = []
    neg = []
    eps = 1e-14
    Pint= []
    for i in range(3):
        if(T[i][1]>=eps):
            pos.append(T[i])
        elif(T[i][1]<=-eps):
            neg.append(T[i])
    if(  len(pos)==3 and len(neg)==0):
        dA,dgam = 0.0,0.0
    elif(len(pos)==0 and len(neg)==3):
        a  = hyp(neg[0][0]-neg[1][0])
        b  = hyp(neg[1][0]-neg[2][0])
        c  = hyp(neg[2][0]-neg[0][0])
        dA,dgam = heron(a,b,c), 0.0
    elif(len(pos)==2 and len(neg)==1):
        s1 = (neg[0][1])/(neg[0][1]-pos[0][1])       
        s2 = (neg[0][1])/(neg[0][1]-pos[1][1])
        P11= (1-s1)*neg[0][0] + (s1)*pos[0][0]
        P22= (1-s2)*neg[0][0] + (s2)*pos[1][0]
        a  = hyp(neg[0][0]-P11)
        b  = hyp(neg[0][0]-P22)
        c  = hyp(P22-P11)
        dA,dgam = heron(a,b,c),c
        Pint.extend([[P11],[P22]])
    elif(len(pos)==1 and len(neg)==2): 
        s1 = (neg[0][1])/(neg[0][1]-pos[0][1])
        s2 = (neg[1][1])/(neg[1][1]-pos[0][1])
        P11= (1-s1)*neg[0][0] + (s1)*pos[0][0]
        P22= (1-s2)*neg[1][0] + (s2)*pos[0][0]
        a  = hyp(neg[0][0]-P22)
        b  = hyp(neg[1][0]-P22)
        c  = hyp(neg[0][0]-neg[1][0])
        d  = hyp(neg[0][0]-P22)
        e  = hyp(neg[0][0]-P11)
        f  = hyp(P22-P11)
        dA,dgam = heron(a,b,c)+heron(d,e,f),f
        Pint.extend([[P11],[P22]])
    else:
        print('wtf:lpos,lneg=',len(pos),len(neg))
    # fnavg = fncalc(Pint)
    fnavg = fncalc(Pint,G,R,m,x0,y0,L,mode)
    return dA,dgam,fnavg


def sysmat(n,Lx,Ly,matind,dx,dy,dv,nv):
    'Assemble system matrix and rhs'
    A = lil_matrix(((n - 2)**2, (n - 2)**2))
    rhs       = np.zeros([(n-2)**2])
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            row = matind[i, j]
            A[row, row] = -(Lx[i,j]/dx+Lx[i+1,j]/dx+Ly[i,j]/dy+Ly[i,j+1]/dy)
            if i < n - 2:
                A[row,matind[i+1, j]]   = Lx[i+1,j]/dx
            else: #periodic
                A[row,matind[1, j]]   = Lx[i+1,j]/dx 
            if i > 1:
                A[row, matind[i-1, j]]  = Lx[i,j]/dx
            else: #periodic
                A[row, matind[n-2, j]]  = Lx[i,j]/dx 
            if j < n - 2:
                A[row, matind[i, j+1]]  = Ly[i,j+1]/dy
            else:
                A[row,row]             -= Ly[i,j+1]/dy
                rhs[row]               -= 2*dv*Ly[i,j+1]/dy
            if j > 1:
                A[row, matind[i, j-1]]  = Ly[i,j]/dy
            else: #assumption: flux = L*dphi/dx. Havent included a negative sign in discretization. 0 so doesnt matter
                A[row,row]             += Ly[i,j]/dy
                rhs[row]               += nv*Ly[i,j]
    A = A.tocsr()
    return A,rhs
    
def FVMstuff(n,rhs,phi,matind,x,y,dx,dy,R,G,m,x0,y0,L,mode):
    perimeter = 0  #validation 
    area      = 0  #validation
    flux      = 0  #validation
    Aarr      = np.zeros([n,n])
    for i in range (1,n-1):
        for j in range(1,n-1):
            phi0 = 0.25*(phi[i,j]+phi[i-1,j]+phi[i-1,j-1]+phi[i,j-1]) #pimjm
            P0   = np.array([x[i]-dx/2,y[j]-dy/2])
            phi1 = 0.25*(phi[i,j]+phi[i+1,j]+phi[i+1,j-1]+phi[i,j-1]) #pipjm
            P1   = np.array([x[i]+dx/2,y[j]-dy/2])
            phi2 = 0.25*(phi[i,j]+phi[i+1,j]+phi[i+1,j+1]+phi[i,j+1]) #pipjp
            P2   = np.array([x[i]+dx/2,y[j]+dy/2])
            phi3 = 0.25*(phi[i,j]+phi[i-1,j]+phi[i-1,j+1]+phi[i,j+1]) #pimjp
            P3   = np.array([x[i]-dx/2,y[j]+dy/2])
            T1 = [[P0,phi0],[P2,phi2],[P3,phi3]]
            T2 = [[P0,phi0],[P1,phi1],[P2,phi2]]
            dA1,dg1,fnavg1 = geometric_int2(T1,G,R,m,x0,y0,L,mode) #geometric_int(T1,-G/R) 
            dA2,dg2,fnavg2 = geometric_int2(T2,G,R,m,x0,y0,L,mode)#geometric_int(T2,-G/R)
            Aarr[i,j]      = dA1 + dA2
            rhs[matind[i,j]] -= fnavg1*dg1 + fnavg2*dg2
            perimeter += dg1 + dg2
            flux +=  fnavg1*dg1 + fnavg2*dg2
            area      += dA1 + dA2
    return perimeter,flux,area,Aarr,rhs
             
    
def qanfn1(rad,G):
    rad = np.maximum(rad,1e-20)
    qan = np.where(rad!=0,G*np.log(rad),0)
    return qan

def qanfn2(G,m,rad,R,theta):
    rad = np.maximum(rad,1e-20)
    qan = np.where(rad!=0,G*np.cos(m*theta)*(R/rad)**m,0)
    return qan

def qanfn3(X,Y,G):
    qan = np.cos(G*X)*np.cosh(G*Y)
    return qan

# def cdcalc(soln,phi,kel,n,dx):
#     'Calculates the current density from the electric potential'
#     ix = np.zeros_like(phi)
#     iy = np.zeros_like(phi)
#     for i in range(1,n-1):
#         for j in range(1,n-1):
#             if(phi[i,j]<0):
#                 continue
#             else:
#                 ix[i,j] = -kel*(soln[i+1,j]-soln[i-1,j])/(2*dx)
#                 iy[i,j] = -kel*(soln[i,j+1]-soln[i,j-1])/(2*dx)
    
#     return ix,iy

def cdcalc1(soln,phi,kel,n,dx):
    'Send -phi as input argument'
    ix = np.zeros_like(phi)
    iy = np.zeros_like(phi)
    for i in range(1,n-1):
        for j in range(1,n-1):
            if(phi[i,j]>0):
                continue
            if((phi[i,j]<0)and(phi[i+1,j]<0)and(phi[i-1,j]<0)):
                ix[i,j] = -kel*(soln[i+1,j]-soln[i-1,j])/(2*dx)
            elif((phi[i,j]<0)and(phi[i+1,j]<0)and(phi[i-1,j]>0)and(phi[i+2,j]<0)):
                ix[i,j] =  -kel*(4*soln[i+1,j]-soln[i+2,j]-3*soln[i,j])/(2*dx)
            else:
                print('Simple approach failing for i flux at i,j=',i,j)
            if((phi[i,j]<0)and(phi[i,j+1]<0)and(phi[i,j-1]<0)):
                iy[i,j] = -kel*(soln[i,j+1]-soln[i,j-1])/(2*dx)
            elif((phi[i,j]<0)and(phi[i,j+1]<0)and(phi[i,j-1]>0)and(phi[i,j+2]<0)):
                iy[i,j] =  -kel*(4*soln[i,j+1]-soln[i,j+2]-3*soln[i,j])/(2*dx)
            else:
                print('Simple approach failing for j flux at i,j=',i,j)
                
    return ix,iy
            
            

def bc_el(soln,dv,nv,n,dx,kel):
    'Imposes the boundary conditions for electric potential'
    soln[0,:]   = soln[n-2,:]           #periodic in i
    soln[n-1,:] = soln[1,:]             #periodic in i
    soln[:,0]   = soln[:,1] + nv*dx/kel #neumann at j=1
    soln[:,n-1] = 2*dv - soln[:,n-2]    #dirichlet bc at j=n-2
    
    return soln

def netflux(soln,Aarr,dy):
    'Computes the flux in and flux out for a flat electrode case. For validation purposes'
    A      = Aarr[1,:] #for phi=Y-y0 case, sdf is same for each row
    ind    = np.argmax(A!=0)
    dpdy   = (soln[1:-1,ind+1]-soln[1:-1,ind])/dy
    fluxin = np.sum(dpdy*dy)
    dpdy   = (soln[1:-1,-2]-soln[1:-1,-3])/dy
    fluxout= np.sum(dpdy*dy)
    return fluxin,fluxout
    
  
# def sysmat(n,Lx,Ly,matind,dx,dy,qan):
#     'Assemble system matrix and rhs'
#     A = lil_matrix(((n - 2)**2, (n - 2)**2))
#     rhs       = np.zeros([(n-2)**2])
#     for i in range(1, n - 1):
#         for j in range(1, n - 1):
#             row = matind[i, j]
#             A[row, row] = -(Lx[i,j]/dx+Lx[i+1,j]/dx+Ly[i,j]/dy+Ly[i,j+1]/dy)
#             if i < n - 2:
#                 A[row,matind[i+1, j]]   = Lx[i+1,j]/dx
#             else:
#                 rhs[row]               -= qan[i+1,j]*Lx[i+1,j]/dx
#             if i > 1:
#                 A[row, matind[i-1, j]]  = Lx[i,j]/dx
#             else:
#                 rhs[row]               -= qan[i-1,j]*Lx[i,j]/dx 
#             if j < n - 2:
#                 A[row, matind[i, j+1]]  = Ly[i,j+1]/dy
#             else:
#                 rhs[row]               -= qan[i,j+1]*Ly[i,j+1]/dy
#             if j > 1:
#                 A[row, matind[i, j-1]]  = Ly[i,j]/dy
#             else:
#                 rhs[row]               -= qan[i,j-1]*Ly[i,j]/dy
#     return A,rhs  
 
