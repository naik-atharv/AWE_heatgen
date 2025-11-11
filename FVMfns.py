# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

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

def fncalc(Pint,G,R,m,x0,y0,mode):
    if len(Pint)==0:
        favg = 0
    elif len(Pint)==2:
        P11 = Pint[0][0]
        P22 = Pint[1][0]
        t11 = np.arctan2(P11[1]-y0,P11[0]-x0)
        t22 = np.arctan2(P22[1]-y0,P22[0]-x0)
        if (mode==2):
         f11 = fn2(m,G,R,t11) #m*G*(R**m)*np.cos(m*t11)/r11**(m+1)
         f22 = fn2(m,G,R,t22) #m*G*(R**m)*np.cos(m*t22)/r22**(m+1)
        elif (mode==3):
         f11 = fn3(G,P11[0],P11[1],t11)
         f22 = fn3(G,P22[0],P22[1],t22)
        favg= 0.5*(f11+f22)
    else:
        print('wtf')
    return favg

def fn2(m,G,R,t):
    return m*G*np.cos(m*t)/R

def fn3(G,x,y,t):
    return G*np.sin(G*x)*np.cosh(G*y)*np.cos(t)-G*np.cos(G*x)*np.sinh(G*y)*np.sin(t)

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

def geometric_int2(T,G,R,m,x0,y0,mode):
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
    fnavg = fncalc(Pint,G,R,m,x0,y0,mode)
    return dA,dgam,fnavg

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

    
 
