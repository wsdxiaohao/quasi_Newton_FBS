#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 15:21:59 2021

@author: shida
"""
import sys
import numpy as np
import scipy as sp
def prox_proj_ball(p,mu):#p is a 1-D array p[:n] for p_x part p[n:N] for p_y part
    N=int(len(p))
    n=int((N+1)/2)
    p_proj=np.zeros(N)
    p_norm=np.sqrt(p[:n]**2+p[n:N]**2)
    p_norm=p_norm/mu
    denominate=np.maximum(p_norm,1)
    p_proj[:n]=p[:n]/denominate
    p_proj[n:N]=p[n:N]/denominate
    return p_proj

def PDHG(OP,Metric,z):
    #Load parameters
    tau=Metric['Tau']
    sig=Metric['Sigma']
    mu=OP['mu']
    K=OP['K']
    A=OP['A']
    b=OP['b']
    # initialization
    x=z['x']
    p=z['p']
   
    num_x=len(x)
    num_p=len(p)

    xk=x
    pk=p

    #PDHG
    ######################        primal          ################################
    #xkp1=xk-tau*K.T.dot(pk)-tau*A.T.dot(qk)
    xkp1=(tau*(b-K.T.dot(pk))+xk)/(1+tau)
    ######################        dual p          ################################
    D=np.ones(num_p)
    #pkp1=prox_proj_ball((pk+sig*K.dot(2*xkp1-xk)),mu)
    #pkp1=proxP_conjf(D,pk+sig*K.dot(2*xkp1-xk),mu)
    p_temp=pk+sig*(K.dot(2*xkp1-xk))#2*x_kp1-x_k
        
    #prox_proj(p_temp)_onto_ball with radius mu
    pkp1=prox_proj_ball(p_temp,mu)
    
   
    zkp1={'x':xkp1,'p':pkp1}
    return zkp1

################################# PDHG for M+Q ############################################

def PDHG_MQ(OP,Metric,z,y):
    #Load parameters
    tau=Metric['Tau']
    sig=Metric['Sigma']
    mu=OP['mu']
    K=OP['K']
    A=OP['A']
    b=OP['b']
    # initialization
    x=z['x']
    p=z['p']
    
    
    num_x=len(x)
    num_p=len(p)
    
    y1=y[0:num_x]
    y2=y[num_x:(num_x+num_p)]
    xk=x
    pk=p

    #PDHG
    ######################        primal          ################################
    #xkp1=xk-tau*K.T.dot(pk)-tau*A.T.dot(qk)
    xkp1=(-tau*y1+tau*(b-K.T.dot(pk))+xk)/(1+tau)
    ######################        dual p          ################################
    D=np.ones(num_p)
    #pkp1=prox_proj_ball((pk+sig*K.dot(2*xkp1-xk)),mu)
    #pkp1=proxP_conjf(D,pk+sig*K.dot(2*xkp1-xk),mu)
    p_temp=pk+sig*(K.dot(2*xkp1-xk))-sig*y2#2*x_kp1-x_k
        
    #prox_proj(p_temp)_onto_ball with radius mu
    pkp1=prox_proj_ball(p_temp,mu)
    
   
    zkp1={'x':xkp1,'p':pkp1}
    return zkp1




def L_function(Preconditioner,OP,Metric,z,a):
    #Load parameters
    Dx=Preconditioner['Dx']
    Dp=Preconditioner['Dp']
    #U=Preconditioner['U']
    Ux=Preconditioner['Ux']
    Up=Preconditioner['Up']
    #######
    num_x=len(Dx)
    num_p=len(Dp)
    ##### JM_T
    
    ztemp={'x':Dx*z['x']-Ux*a,'p':Dp*z['p']-Up*a}
    #ztemp={'x':Dx*z['x']+Ux*a,'p':Dp*z['p']+Up*a} 
    zafter=PDHG(OP,Metric,ztemp)
    
    #####
    x=zafter['x']
    p=zafter['p']
    JTM=np.concatenate([x,p])
    JTM.flatten()
    D=np.concatenate([Dx,Dp])
    D.flatten()

    U=np.concatenate([Ux,Up])
    Zk = np.concatenate([z['x'],z['p']])
    return    U.T.dot(JTM-Zk)-a #U.T.dot(1/D*JTM-U*a)-a
    #     U.flatten()
#     return U.T.dot(1/D*JTM-U*a)-a

def LfunctionMplusQ(Preconditioner,OP,Metric,z,a):
    #Load parameters
    Dx=Preconditioner['Dx']
    
    Dp=Preconditioner['Dp']
    
    #U=Preconditioner['U']
    
    U=Preconditioner['U']
    #######
    num_x=len(Dx)
    num_p=len(Dp)
    ##### 
    #ztemp={'x':Dx*z['x']-Ux*a,'p':Dp*z['p']-Up*a}
    ztemp={'x':z['x'],'p':z['p']}
    zafter=PDHG_MQ(OP,Metric,ztemp,-U*a)
    
    #####
    x=zafter['x']
    p=zafter['p']
    JTM=np.concatenate([x,p])
    JTM.flatten()
    D=np.concatenate([Dx,Dp])
    D.flatten()
    #Zk = np.concatenate([z['x'],z['p']])
    #return    #U.T.dot(JTM-Zk)-a #U.T.dot(1/D*JTM-U*a)-a
    return U.T.dot(1/D*JTM-U*a)-a

def norm(v):
    return np.sqrt(v@v.T)
def Rootfinding(Preconditioner,OP,Metric,z):
    # we are going to use bisection method to solve L=0 when r=1
    #nonzeroL    = find( abs(L) > 100*np.finfo(float).eps );
    tol=1*np.finfo(float).eps
    maxit=200
    
    #Load parameters
    U=Preconditioner['U']
    
    x0=np.concatenate([z['x'],z['p']])
    #before processing
    
    bound=100+norm(U)*(2*norm(x0))#+norm(ProxB_l1(0,tau,D)))# coRootfinding_r1(x,D,v,tau,maxit):upute the bound before using bisection method
    k=0

    if L_function(Preconditioner,OP,Metric,z,bound) >0:
        c1=-bound
        c2=bound
    else:
        c1=bound
        c2=-bound
    ak=bound
    
    while k<maxit:
        k=k+1
        akm1=ak
        ak=(c1+c2)/2
        #print('C2=%.3f,ak=%.3f'%(c2,ak))
        if L_function(Preconditioner,OP,Metric,z,ak)>0:
            c2=ak
        else:
            c1=ak
        diff=np.abs(ak-akm1)
        if (k>1)&(diff<tol):
            return ak
        
        
def MQRootfinding(Preconditioner,OP,Metric,z):
    # we are going to use bisection method to solve L=0 when r=1
    #nonzeroL    = find( abs(L) > 100*np.finfo(float).eps );
    tol=1*np.finfo(float).eps
    maxit=200
    
    #Load parameters
    U=Preconditioner['U']
    
    x0=np.concatenate([z['x'],z['p']])
    #before processing
    
    bound=100+norm(U)*(2*norm(x0))#+norm(ProxB_l1(0,tau,D)))# coRootfinding_r1(x,D,v,tau,maxit):upute the bound before using bisection method
    k=0

    if LfunctionMplusQ(Preconditioner,OP,Metric,z,bound) >0:
        c1=-bound
        c2=bound
    else:
        c1=bound
        c2=-bound
    ak=bound
    
    while k<maxit:
        k=k+1
        akm1=ak
        ak=(c1+c2)/2
        #print('C2=%.3f,ak=%.3f'%(c2,ak))
        if LfunctionMplusQ(Preconditioner,OP,Metric,z,ak)>0:
            c2=ak
        else:
            c1=ak
        diff=np.abs(ak-akm1)
        if (k>1)&(diff<tol):
            return ak







def TVnorm(n,K,x):
    N=2*n #len of Kx
    Kx=K.dot(x)
    Dx_2=(Kx*(Kx));# entrywise squared Kx
    Dx_tv=np.sum(np.sqrt(Dx_2[0:n]+Dx_2[n:N])); #Total variation of Dx
    return Dx_tv



def cal_primal_dual_gap_ROF(A,K,b,x,p,mu):
    #computing primal and dual gap for ROF When A=identity
    #input: K matrix of derivative2D
    #       b is the noisy image
    #       x is the generated image
    #       p is the dual variable
    #       mu is the parameter before regularization term
    n=b.shape[0]
    res=A@x-b;
    Dx_tv=TVnorm(n,K,x);
    primal=0.5*(res).dot(res)+mu*Dx_tv; 
    #invA = sp.sparse.linalg.inv(A)
    #dual=-0.5*(invA@K.T.dot(p)-b).dot(invA@K.T.dot(p)-b)+0.5*b.dot(b);
    gap =primal#-dual
    return gap

def PDHG_precond(Preconditioner,OP,Metric,z,NumOfIter,param,check):
    #initial
    check=10
    Dx=Preconditioner['Dx']
    Dp=Preconditioner['Dp']
    num_x=len(Dx)
    num_p=len(Dp)
    K=OP['K']
    b=OP['b']
    U=Preconditioner['U']
    Ux=U[0:num_x]
    Up=U[(num_x):(num_p+num_x)]
    Vx=Ux
    Vp=Up
    tol=1*np.finfo(float).eps
    GAP=[]

    #run
    for iter in range(NumOfIter):
        z0=z
        #ztemp=z
        a=Rootfinding(Preconditioner,OP,Metric,z0)
        
        #ztemp={'x':Dx*z['x']-Ux*a,'p':Dp*z['p']-Up*a}
        
        #ztemp={'x':Dx*z['x']-U[0:num_x]*a,'p':Dp*z['p']-U[(num_x):(num_p+num_x)]*a}
        
        ztemp={'x':Dx*z['x']+Ux*a,'p':Dp*z['p']+Up*a}
        
        Dz=PDHG(OP,Metric,ztemp)
        z={'x':1/Dx*Dz['x'],'p':1/Dp*Dz['p']}
        
        ############## test special U ###########
        Ux=0.1*param*(z['x']-z0['x'])
        Up=0*param*(z['p']-z0['p'])
        
        #Vx=Ux/np.sqrt(1+norm(Ux)**2+norm(Up)**2)
        #Vp=Up/np.sqrt(1+norm(Ux)**2+norm(Up)**2)
        
        Preconditioner['Ux']=Ux
        Preconditioner['Up']=Up
        #########################################
        #checking breaking condition
        gap=cal_primal_dual_gap_ROF(K,b,z['x'],z['p'],OP['mu'])
        if (iter % check == 0):
            print ('iter: %d, gap: %f' % (iter, gap));
            print ('iter: %d, a: %f' % (iter, a));
            print('Ux=')
            print(Ux)
            #print ('iter: %d, Ux: %f,Up:%f' % (iter, z['Ux'],z['Up']));
            GAP.append(gap)
        if (gap < tol):
            breakvalue = 1;
            break;
    #########################################
    return z, GAP

def PDHGoriginal(OP,Metric,z,NumOfIter,check):
    #initial
    
    
    K=OP['K']
    b=OP['b']
    tol=1*np.finfo(float).eps
    GAP=[]

    #run
    for iter in range(NumOfIter):
        z0=z
        z=PDHG(OP,Metric,z0)
        #########################################
        #checking breaking condition
        gap=cal_primal_dual_gap_ROF(K,b,z['x'],z['p'],OP['mu'])
        if (iter % check == 0):
            print ('iter: %d, gap: %f' % (iter, gap));
            GAP.append(gap)
        if (gap < tol):
            breakvalue = 1;
            break;
    #########################################
    return z, GAP

def PDHG_MplusQ(Preconditioner, OP,Metric,z,NumOfIter,check=10):
    #M=Dx+uu^T    -K^*
    #   -K             Dy+vv^T
    ######## loading ###############
    U=Preconditioner['U']
    Dx=Precondioner['Dx']
    Dp=Preconditoner['Dp']
    u=U[0:num_x]
    v=U[(num_x):(num_p+num_x)]
    x=z['x']
    p=z['p']
    
    tol=1*np.finfo(float).eps
    GAP=[]
    for iter in range(NumOfIter):
        z0=z
        #########################
        # calculating a
        a=MQRootfinding(Preconditioner,OP,Metric,z0)
        #########################
        
        y=-U*a ############# or U*a
        #########################
        z=PDHG_MplusQ(OP,Metric,z0,y)
        #########################################
        #checking breaking condition
        gap=cal_primal_dual_gap_ROF(K,b,z['x'],z['p'],OP['mu'])
        if (iter % check == 0):
            print ('iter: %d, gap: %f' % (iter, gap));
            GAP.append(gap)
        if (gap < tol):
            breakvalue = 1;
            break;
        ##############   calculating U  ###############
        if iter%2==1:
            U[0:num_x]=0.99*(z['x']-z0['x'])
            Preconditioner['U'] = U
        else:
            U=0*U
            Preconditioner['U'] = U
        ###################################
    #########################################
    return z, GAP
