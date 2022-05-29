import math
import numpy as np
from numpy import random as rand
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize as minim
from scipy import optimize as opt
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import qubo_to_ising
from dimod import ising_to_qubo
import dwave.inspector as inspector
def Phi(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def phi(x):
    return 1.0/math.sqrt(2*math.pi)*math.exp(-x*x/2.0)

a = 3
b = 1
expect = b/a
eps = 0.05
board = 1
c = board
d = board - expect
N = 100   
R = 25 #math.floor(2+math.log2(c/eps))  
M = 2**R    
delta = c/(2**(R-2))  
def to_decimal_int(x):
    global R
    num = 0
    for i in range(R):
        num = num + x[R-1-i] * (2**i)
    return num

Q = {(0, 0): 0}

for i in range(R):
    temp1 = a*c/(2**i)
    for j in range(i,R):
        temp2 = a*c/(2**j)
        if j==i:
            Q[(i, j)] = temp1*(temp1 - 2*(a*d+b))
        else:
            Q[(i, j)] = 2*temp1*temp2

qpu_advantage = DWaveSampler(solver={'topology__type': 'pegasus'})
h = qubo_to_ising(Q)[0]
J = qubo_to_ising(Q)[1]
max_h = qpu_advantage.properties["h_range"][1]
min_h = qpu_advantage.properties["h_range"][0]
max_J = qpu_advantage.properties["j_range"][1]
min_J = qpu_advantage.properties["j_range"][0]
Factor = 0.1+max(max(max(list(h.values()))/max_h,0),max(min(list(h.values()))/min_h,0),max(max(list(J.values()))/max_J,0),max(min(list(J.values()))/min_J,0))
max(h.values())
min(h.values())
max(J.values())
min(J.values())
for i in range(R):
    h[i] /= Factor

for i in range(R):
    for j in range(i+1,R):
        J[(i,j)] /= Factor

max(h.values())
min(h.values())
max(J.values())
min(J.values())
Q = ising_to_qubo(h,J)[0]
sampler_auto = EmbeddingComposite(qpu_advantage)
sampleset = sampler_auto.sample_qubo(Q, num_reads = N, auto_scale=False, num_spin_reversal_transforms = math.floor(N/100))
inspector.show(sampleset)
sam = sampleset.record
XiFull=[]  
for i in range(M):
    XiFull.append(c * i/2**(R-1) - d)

KiFull = [] 
for i in range(M):
    KiFull.append(0)

m = len(sam) 
for i in range(m):  
    KiFull[to_decimal_int(sam[i][0])] += sam[i][2]

WiFull=[]  
tempint=KiFull[0]
for i in range(1,M):
    WiFull.append(tempint/N)
    tempint=tempint+KiFull[i]

Ki = [] 
Xi = []
for i in range(M): 
    if KiFull[i] != 0:
        Ki.append(KiFull[i])
        Xi.append(c*i/2**(R-1)-d)

m = len(Xi) 
Wi=[]  
tempint=0
for i in range(m):
    Wi.append(tempint/N)
    tempint=tempint+Ki[i]

Wi.append(1)
el=math.floor(m/2)-1
if Xi[el]<expect:
    while Xi[el]<expect:
        el=el+1
        if el>m:
            el=el-1
            break
else:
    while Xi[el]>=expect:
        el=el-1
        if el<0:
            break
    el=el+1

##################
a/=Factor**(1/2)
b/=Factor**(1/2)
##################
def metricL2Prime(s):
    global m
    global N
    global Xi
    global Ki
    global expect
    res=-N/math.sqrt(2)
    if s==0:
        return res
    else:
        for i in range(m):
            res=res+Ki[i]*math.exp(-(Xi[i]-expect)*(Xi[i]-expect)/2.0/s/s)
    return res

Qi = []
for i in range(M):
    Qi.append(0)

def calculate_Qi_QM(beta):
    global M
    global a
    global b
    global XiFull
    global Qi
    for i in range(M):
        Qi[i] = math.exp((-1)*beta*(a*XiFull[i]-b)*(a*XiFull[i]-b))
    
    QM = sum(Qi)
    return QM

def metricJSD(beta):
    global N
    global M
    global a
    global b
    global XiFull
    global KiFull
    global Qi
    QM = calculate_Qi_QM(beta)
    res = math.log(2.0/math.sqrt(N*QM))
    temp = 0
    for i in range(M):
        if KiFull[i]!=0:
            temp += KiFull[i] * (math.log(KiFull[i]) - math.log(KiFull[i]/N+Qi[i]/QM))
    
    res += temp/2.0/N
    temp = 0
    for i in range(M):
        temp += (a*XiFull[i]-b)*(a*XiFull[i]-b)*Qi[i]
    
    res -= beta*temp/2.0/QM
    temp = 0
    for i in range(M):
        if KiFull[i] == 0:
            temp -= Qi[i]*(beta*(a*XiFull[i]-b)*(a*XiFull[i]-b) + math.log(QM))
        else:
            temp += Qi[i]*math.log(KiFull[i]/N+Qi[i]/QM)
    
    res -= temp/2.0/QM
    res = math.sqrt(res)
    return res

expect_estim=sum(np.multiply(Xi,Ki))/N
sigma_estim=math.sqrt(sum(np.multiply(np.multiply(Xi-expect_estim,Xi-expect_estim),Ki))/(N-1))

SigmaL2=opt.root_scalar(metricL2Prime,bracket=[0, 2*sigma_estim+2],method='toms748',xtol=8.881784197001252e-16,rtol=8.881784197001252e-16).root
if SigmaL2 == 0:
    BetaL2 = 1e+309
else:
    BetaL2=0.5/a/a/SigmaL2/SigmaL2

BetaJSD=opt.minimize_scalar(metricJSD,method='golden',tol=8.881784197001252e-16,options={'xtol': 8.881784197001252e-16}).x
SigmaJSD = 1.0/a/math.sqrt(2*BetaJSD)
X=[]
for i in range(m):
    for j in range(Ki[i]):
        X.append(Xi[i])

density = gaussian_kde(X)
lag=0.01
width=3*sigma_estim
Left=max(Xi[0],expect-width)
Right=min(Xi[m-1],expect+width)
numberOfBins = 1 + math.floor(math.log2(N))
#numberOfBins = math.floor((Right-Left)*100)
x=np.arange(Left, Right, lag)


labels1=['Гистограмма частот','Гауссовская ЯОП','Плотность нормального\nраспределения']
y1=np.empty(len(x), dtype=float)
for i in range(len(x)):
    y1[i]=1.0/SigmaL2*phi((x[i]-expect)/SigmaL2)

plt.figure(figsize=(5,5))
plt.subplots_adjust(hspace=0.3)
##plt.suptitle('Сравнение с нормальным распределением, N = '+str(N)+', '+'R = '+str(R))
plt.hist(X, density=True, bins=numberOfBins,range=(Left,Right),color='c',histtype='stepfilled',label=labels1[0])
plt.plot(x,density(x),'--',color=(0,0,1),label=labels1[1])
plt.plot(x,y1,color=(1,0,0),label=labels1[2])
plt.legend(loc='best')
plt.grid(True)
LeftXi=0
RightXi=M
i=0
while XiFull[i]<Left:
    LeftXi+=1
    i+=1

i=M-1
while XiFull[i]>Right:
    RightXi-=1
    i-=1

labels2=['Гистограмма частот','Гауссовская ЯОП','Нормированные вероятности\nраспределения Больцмана']
QM=calculate_Qi_QM(BetaJSD)
y5=np.empty(M, dtype=float)
for i in range(M):
    y5[i]=Qi[i]/QM/delta*2

plt.figure(figsize=(5,5))
plt.subplots_adjust(hspace=0.3)
##plt.suptitle('Сравнение распределением Больцмана, N = '+str(N)+', '+'R =' +str(R))
plt.hist(X, density=True, bins=numberOfBins,range=(Left,Right),color='c',histtype='stepfilled',label=labels2[0])
plt.plot(x,density(x),'--',color=(0,0,1),label=labels2[1])
plt.plot(XiFull[LeftXi:RightXi+1],y5[LeftXi:RightXi+1],color=(1,0,0),label=labels2[2])
plt.legend(loc='best')
plt.grid(True)

# English versions
labels1=['Histogram','Gaussian KDE','PDF of normal distr.']
plt.figure(figsize=(5,5))
plt.subplots_adjust(hspace=0.3)
##plt.suptitle('Comparison with a normal distribution, N = '+str(N)+', '+'R = '+str(R))
plt.hist(X, density=True, bins=numberOfBins,range=(Left,Right),color='c',histtype='stepfilled',label=labels1[0])
plt.plot(x,density(x),'--',color=(0,0,1),label=labels1[1])
plt.plot(x,y1,color=(1,0,0),label=labels1[2])
plt.legend(loc='best')
plt.grid(True)
labels2=['Histogram','Gaussian KDE','Normalized Boltzmann\nprobabilities']
plt.figure(figsize=(5,5))
plt.subplots_adjust(hspace=0.3)
##plt.suptitle('Comparison with a Boltzmann distribution, N = '+str(N)+', '+'R =' +str(R))
plt.hist(X, density=True, bins=numberOfBins,range=(Left,Right),color='c',histtype='stepfilled',label=labels2[0])
plt.plot(x,density(x),'--',color=(0,0,1),label=labels2[1])
plt.plot(XiFull[LeftXi:RightXi+1],y5[LeftXi:RightXi+1],color=(1,0,0),label=labels2[2])
plt.legend(loc='best')
plt.grid(True)
print("SigmaL2 = %f\nBetaL2 = %f\nSigmaJSD = %f\nBetaJSD = %f"%(SigmaL2,BetaL2,SigmaJSD,BetaJSD))
plt.show()
