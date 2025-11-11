# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt

def sdarc(X,Y,xc,yc,R,ts,te):
     """Unsigned distance for a circular arc, ts=theta_start, te=theta_end"""
     "here, ts and te corespond to the interval which DOESNT have an arc"
     lex   = xc + R*np.cos(te)
     ley   = yc + R*np.sin(te)
     rex   = xc + R*np.cos(ts)
     rey   = yc + R*np.sin(ts)
     theta = np.mod(np.arctan2(Y-yc,X-xc),2*np.pi)
     r     = np.sqrt((X-xc)**2+(Y-yc)**2)
     dl    = np.sqrt((X-lex)**2+(Y-ley)**2)
     dr    = np.sqrt((X-rex)**2+(Y-rey)**2)
     dedge = np.minimum(dl,dr)
     
     usdf  = np.where((theta>=ts)&(theta<=te),dedge,abs(r-R))
     
     return usdf
 
def sdseg(X,Y,xA,xc,y0,R,theta):
    "Unsigned distance for a line segment. [xA,yA] lies at the x end of domain"
    xB    = xc+R*np.cos(theta)
    yA,yB = y0,y0    
   
    dxA = X - xA
    dyA = Y - yA
    dxAB= xB- xA
    dyAB= yB- yA
    t  = (dxA*dxAB + dyA*dyAB)/((dxAB)**2 + (dyAB)**2)
    tst= np.where(t<0,0,t)
    tst= np.where(t>1,1,t)
    
    xQ = xA + tst*dxAB
    yQ = yA + tst*dyAB
    
    usdf = np.sqrt((X-xQ)**2 + (Y-yQ)**2)
    
    return usdf

def sdfcomp1(X,Y,x0,y0,R,ts,te): 
    "Creates the sdf for an outward circular bump on a flat electrode, give ts=pi,te=2pi"
    xc  = x0
    yc  = y0 + R*np.sin(ts-np.pi) #shift upwards
    d1  = sdarc(X,Y,xc,yc,R,ts,te)
    d2  = sdseg(X,Y,np.min(X),xc,y0,R,ts)
    d3  = sdseg(X,Y,np.max(X),xc,y0,R,te)
    r   = np.sqrt((X-xc)**2+(Y-yc)**2)
    usdf= np.minimum(d1,d2)
    usdf= np.minimum(usdf,d3)
    sign= np.where(Y>y0,1,-1) #y0+R*np.sin(ts)
    sign= np.where(r>R,sign,-1)
    sdf = usdf*sign
    return sdf,usdf

def larc1(x0,y0,R,ts,te,L): #coresponds to sdfcomp1
    l1   = (x0 + R*np.cos(ts)) - (-L/2)
    l2   = L/2 - (x0 + R*np.cos(te))
    larc = R*(te-ts) 
    
    return l1+l2+larc

def sdfcomp2(X,Y,x0,y0,R,ts,te):
    "Creates the sdf for an inward circular depression on a flat electrode,give ts=0,te=pi"
    xc  = x0
    yc  = y0 - R*np.sin(ts) #shift downwards
    d1  = sdarc(X,Y,xc,yc,R,ts,te)
    d2  = sdseg(X,Y,np.min(X),xc,y0,R,te)
    d3  = sdseg(X,Y,np.max(X),xc,y0,R,ts)
    r   = np.sqrt((X-xc)**2+(Y-yc)**2)
    usdf= np.minimum(d1,d2)
    usdf= np.minimum(usdf,d3)
    sign= np.where(Y>y0,1,-1) #y0+R*np.sin(ts)
    sign= np.where(r>R,sign,1)
    sdf = usdf*sign
    return sdf

def sdarc2(X,Y,xc,yc,R,ts,te,fac):
     """Unsigned distance for a circular arc, ts=theta_start, te=theta_end"""
     "here, ts and te corespond to the interval which has an arc"
     'fac helps decide whether +1 inside arc or -1'
     lex   = xc + R*np.cos(te)
     ley   = yc + R*np.sin(te)
     rex   = xc + R*np.cos(ts)
     rey   = yc + R*np.sin(ts)
     theta = np.mod(np.arctan2(Y-yc,X-xc),2*np.pi)
     r     = np.sqrt((X-xc)**2+(Y-yc)**2)
     dl    = np.sqrt((X-lex)**2+(Y-ley)**2)
     dr    = np.sqrt((X-rex)**2+(Y-rey)**2)
     dedge = np.minimum(dl,dr)
     window= (theta>ts)&(theta<te)
     
     usdf  = np.where(window,abs(r-R),dedge)
     sgn   = np.where(window,-fac*np.sign(r-R),1)
     
     return usdf,sgn
 
def sdseg2(X,Y,xA,xB,yA,yB):
    "Unsigned distance for a line segment. [xA,yA] lies at the x end of domain"

    dxA = X - xA
    dyA = Y - yA
    dxAB= xB- xA
    dyAB= yB- yA
    t  = (dxA*dxAB + dyA*dyAB)/((dxAB)**2 + (dyAB)**2)
    tst= np.where(t<0,0,t)
    tst= np.where(t>1,1,t)
    
    xQ = xA + tst*dxAB
    yQ = yA + tst*dyAB
    
    usdf = np.hypot(X - xQ, Y - yQ)
    
    cross= (X-xA)*dyAB - (Y-yA)*dxAB #z-component of cross product , figure out when it gives correct sign
    sgn  = np.sign(cross) 
    
    return usdf,sgn



# def sdfcomp3(X,Y,x0,y0,R,rcur): #re check the sign function 
#     "Creates the sdf for an inward circular depression on a flat electrode with rounded edges. rcur=radius of curvature of round edge"
#     toff = np.arcsin(rcur/R)
#     ts   = np.pi+toff
#     te   = 2*np.pi-toff
#     xc1  = x0-(R+rcur)
#     xc2  = x0+(R+rcur)
#     yc1,yc2= y0-rcur,y0-rcur
#     d1 = sdarc2(X,Y,x0,y0,R,ts,te) #primary arc
#     d2 = sdarc2(X,Y,xc1,yc1,rcur,0,np.pi/2)     #secondary arc for x<0
#     d3 = sdarc2(X,Y,xc2,yc2,rcur,np.pi/2,np.pi) #secondary arc for x>0
#     d4 = sdseg2(X,Y,np.min(X),xc1,y0,y0)            # line segment for x<0 
#     d5 = sdseg2(X,Y,np.max(X),xc2,y0,y0)            # line segment for x>0
#     usdf= np.minimum(d1,np.minimum(d2,np.minimum(d3,np.minimum(d4,d5))))         
#     r    = np.sqrt((X-x0)**2+(Y-y0)**2)
#     r1   = np.sqrt((X-xc1)**2+(Y-yc1)**2)
#     r2   = np.sqrt((X-xc2)**2+(Y-yc2)**2)
#     t1   = np.mod(np.arctan2(Y-yc1,X-xc1),2*np.pi)
#     t2   = np.mod(np.arctan2(Y-yc2,X-xc2),2*np.pi)
#     sign = np.where((Y>y0),1,-1)
#     sign = np.where((Y<y0)&(r<R),1,sign)
#     sign = np.where((Y<y0)&(r1>rcur)&(r>R)&(t1>=0)&(t1<np.pi/2)&(X<=-R),1,sign)
#     sign = np.where((Y<y0)&(r2>rcur)&(r>R)&(t2>=np.pi/2)&(t2<=np.pi)&(X>=R),1,sign)
#     sdf  = usdf*sign
#     # sign = np.where((Y<y0)&(r2>rcur),1,sign)
#     return sdf,sign

def sdfcomp4(X,Y,x0,y0,R,Rf): #re check the sign function 
    "Creates the sdf for an inward circular depression on a flat electrode with rounded edges. rcur=radius of curvature of round edge"
    toff = np.arcsin(Rf/(R+Rf))
    ts   = np.pi+toff
    te   = 2*np.pi-toff
    xc1,yc1 = x0-(R+Rf)*np.cos(toff),y0-Rf
    xc2,yc2 = x0+(R+Rf)*np.cos(toff),y0-Rf 
    d1,sgn1 = sdarc2(X,Y,x0,y0,R,ts,te,1)                  #primary arc
    d2,sgn2 = sdarc2(X,Y,xc1,yc1,Rf,toff,np.pi/2,-1)        #secondary arc for x<0
    d3,sgn3 = sdarc2(X,Y,xc2,yc2,Rf,np.pi/2,np.pi-toff,-1)  #secondary arc for x>0
    d4,sgn4 = sdseg2(X,Y,np.min(X),xc1,y0,y0)            # line segment for x<0 
    d5,sgn5 = sdseg2(X,Y,np.max(X),xc2,y0,y0)            # line segment for x>0
    # usdf= np.minimum(d1,np.minimum(d2,np.minimum(d3,np.minimum(d4,d5))))         
    # usdf= np.minimum(d1,np.minimum(d2,d3))
    d_all  = np.stack([d1,d2,d3,d4,d5],axis=0)
    sgn_all= np.stack([sgn1,sgn2,sgn3,-sgn4,sgn5],axis=0) #figure why it fails for sg4
    idx    = np.argmin(d_all,axis=0)    
    n      = X.shape[0]
    row    = np.arange(n)[:,None]
    col    = np.arange(n)[None,:]
    sgn    = sgn_all[idx,row,col]
    usdf   = d_all[idx,row,col]
    return usdf,sgn*usdf

def larc4(x0,y0,Rf,R,ts,te,toff,X,L): #coresponding to sdfcomp4
    l1    = (x0-(R+Rf)*np.cos(toff)) - (-L/2)
    l2    = (L/2) - (x0+(R+Rf)*np.cos(toff))
    larc1 = R*(te-ts)
    larc2 = 2*Rf*(np.pi/2-toff)
    return l1+l2+larc1+larc2