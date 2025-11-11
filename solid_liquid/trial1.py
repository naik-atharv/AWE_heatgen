# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt

N  = 120
Ls = 100*10**(-6)
Le = 100*10**(-6)
ke = 21.5 #S/m
ks = 1.4*10**(7) #S/m
phib=-0.4 #phi_bus
phif= 0   #phi_farfield
phieq=-0.1   #equilibrium potential
i0  = 1   #exchange cd
vT  = 0.051  #thermal voltage/alpha
xe = np.linspace(0,Le,N)
xs = np.linspace(-Ls,0,N)
dx = xe[1] - xe[0]

def assemble(U,N):
    #Residual assembly, have not included conductivity in equations where it could be sent to the other side
    R = np.zeros(2*N+1)
    J = np.zeros([2*N+1,2*N+1])
    for i in range(2*N+1):
        if i==0:
            R[i] =  U[i]- phib
            J[i,i]= 1
        elif((i>0)and(i<N-1)):
            R[i] = (-2*U[i]+U[i+1]+U[i-1])/dx**2
            J[i,i-1] = 1/dx**2
            J[i,i]   =-2/dx**2
            J[i,i+1] = 1/dx**2            
        elif(i==N-1):
            R[i] = ks*(3*U[i]-4*U[i-1]+U[i-2])/(2*dx) + U[2*N]
            J[i,i-2] = ks*1/(2*dx)
            J[i,i-1] = ks*(-4)/(2*dx)
            J[i,i]   = ks*3/(2*dx)
            J[i,2*N] = 1
        elif(i==N):
            R[i]     = ke*(-3*U[i]+4*U[i+1]-U[i+2])/(2*dx)+ U[2*N]
            J[i,i+2] = ke*(-1)/(2*dx)
            J[i,i+1] = ke*(4)/(2*dx)
            J[i,i]   = ke*(-3)/(2*dx)
            J[i,2*N] = 1
        elif((i>N)and(i<2*N-1)):
            R[i] = (-2*U[i]+U[i+1]+U[i-1])/dx**2
            J[i,i-1] = 1/dx**2
            J[i,i]   =-2/dx**2
            J[i,i+1] = 1/dx**2        
        elif(i==2*N-1):
            R[i]  = U[i]-phif
            J[i,i]= 1
        elif(i==2*N):
            R[i] = U[i]-i0*np.sinh((U[N-1]-U[N]-phieq)/vT)
            J[i,i]   = 1
            J[i,N-1]=-(i0/vT)*np.cosh((U[N-1]-U[N]-phieq)/vT)
            J[i,N]  = (i0/vT)*np.cosh((U[N-1]-U[N]-phieq)/vT)
            
    return R,J
            
def newton(U0,tol=1e-6,maxit=500):
    U = U0.copy()
    for k in range(maxit):
        R,J = assemble(U, N)
        r   = np.linalg.norm(R,2)
        if r< tol:
            break
        dU  = np.linalg.solve(J,-R)
        U   = U + dU
    return U, r, k
        
U0 = np.zeros(2*N+1)
U0[:N]    = phib
U0[N:2*N] = phif
U0[2*N]   = 0

U, r, k = newton(U0)
print(f"converged in {k} iters, ||R||2={r:.3e}")    

phie = np.zeros_like(xe)
phis = np.zeros_like(xs)
phis = U[:N] 
phie = U[N:2*N]
j    = U[2*N]

plt.figure(1)
plt.plot(xs*1e6,phis,'b',label=r'$\phi_s$')
# plt.legend()
# plt.figure(2)
plt.plot(xe*1e6,phie,'k',label=r'$\phi_e$')
plt.xlabel(r'$x (\mu m)$')
plt.ylabel('$\phi (V) $')
plt.legend()