from dwave.system import DWaveSampler, EmbeddingComposite
import math
from math import floor
from math import exp
from math import log2
from math import log
from math import sqrt
from math import fabs
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.patches import Ellipse
import numpy.linalg as lin
from scipy.stats import gaussian_kde

def calc_Q(A,b,c,d,R,alpha=0):  # вычисление коэффициентов Гамильтониана 
    A = np.array(A)
    b = np.array(b)
    m = len(A)
    n = len(A[0])
    Q = {(0,0): 0}
    Ci = np.zeros([n],dtype=float)
    Bi = np.zeros([n],dtype=float)
    Aij = np.zeros([n,n],dtype=float)
    At = A.T
    for i in range(n):
        Bi[i] = At[i].dot(b)
    
    for i in range(n):
        for j in range(i,n):
            Aij[i][j] = At[i].dot(At[j])
    
    for i in range(n):
        for j in range(i+1,n):
            Aij[j][i] = Aij[i][j]
    
    for i in range(n):
        Ci[i] = sum(Aij[i])
    
    pows = np.empty([R],dtype=float)
    for i in range(R):
        pows[i] = c*2**(-i)
    
    for el in range(n*R):
        for p in range(el,n*R):
            i = floor(el/R)
            s = el-i*R
            j = floor(p/R)
            r = p-j*R
            if el==p:
                Q[(el,p)] = pows[s] * (pows[s]*(alpha + Aij[i][j]) - 2*Bi[i] - 2*d*(alpha + Ci[i]))
            else:
                if i==j:
                    Q[(el,p)] = 2*pows[r]*pows[s] * (alpha + Aij[i][j])
                else:
                    Q[(el,p)] = 2*pows[r]*pows[s] * Aij[i][j]
    
    return Q

def to_dec(x,n,R,c,d):
    sol = np.empty([n], dtype=float)
    for j in range(n):  # преобразуем в десятичное каждую компоненту 
        sol[j] = 0
        for s in range(R):
            sol[j] = sol[j] + x[j*R+s] / (2**s)
        sol[j] = c * sol[j] - d
    
    return sol

#A = [[-0.4,-3.1],[1.6,10]]
#A = [[2,-1],[1,2]]
A = np.array([[0.2,-1],[1,2]])
A =  np.array(A)
exact = np.array([0.2,0.4])
b = A.dot(exact)
math.floor(lin.cond(A))
kappa = math.floor(lin.cond(A))
N = 10000
eps = 0.01 # желаемый размер сетки
c = 5
d = c
R = math.floor(math.log2(c/eps))+1
m = len(A)
n = len(A[0])
A1 = [[ A[1][1], -A[0][1]],
      [-A[1][0],  A[0][0]]]
#exact = np.dot(lin.inv(A),b)
Q = calc_Q(A,b,c,d,R)
sampler_auto = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))
sampleset = sampler_auto.sample_qubo(Q, num_reads = N, num_spin_reversal_transforms=math.floor(N/100))
sam = sampleset.record
x = np.empty([N,n], dtype=float)    # массив всех решений с повторениями
sol = np.empty([n], dtype=float)
j = 0
for i in range(len(sam)):
    x[i+j] = to_dec(sam[i][0],n,R,c,d)
    for k in range(sam[i][2]-1):
        j = j + 1
        x[i+j] = to_dec(sam[i][0],n,R,c,d)

expect = sum(x)/N
print("Выборочное среднее =",expect)
shifts = np.empty([N], dtype=float) # нормы отклонений решений
for i in range(N):
    shifts[i] = lin.norm(exact-x[i])

xmin = min(min(x.T[0]),exact[0])-0.03
xmax = max(max(x.T[0]),exact[0])+0.03
ymin = min(min(x.T[1]),exact[1])-0.03
ymax = max(max(x.T[1]),exact[1])+0.03
Max = max(shifts)
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([x.T[0], x.T[1]])
kernel = gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape) # эмпирическая двумерная плотность

if A[0][1] == 0:  # строим прямые из системы Ax=b
    x00 = b[0]/A[0][0]
    x01 = b[0]/A[0][0]
    y00 = ymin
    y01 = ymax
else:
    x00 = xmin
    x01 = xmax
    y00 = (b[0]-A[0][0]*x00)/A[0][1]
    y01 = (b[0]-A[0][0]*x01)/A[0][1]

if A[1][1] == 0:
    x10 = b[1]/A[1][0]
    x11 = b[1]/A[1][0]
    y10 = ymin
    y11 = ymax
else:
    x10 = xmin
    x11 = xmax
    y10 = (b[1]-A[1][0]*x10)/A[1][1]
    y11 = (b[1]-A[1][0]*x11)/A[1][1]

fig = pl.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.set(xlim = [xmin, xmax],ylim = [ymin, ymax])
ax.imshow(np.rot90(Z), cmap=pl.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
ax.plot(x.T[0], x.T[1], 'k.', markersize=1)
ax.plot([x00,x01], [y00,y01], color = (1,0,0),label="Прямые из системы уравнений")
ax.plot([x10,x11], [y10,y11], color = (1,0,0))
S = np.dot(A1,np.transpose(A1)) # строим теор. оси симметрии гауссиана
if S[0][1] != 0 or S[0][0] != S[1][1]:
    K = -(2*S[0][1])/(S[1][1]-S[0][0] + math.sqrt(4*S[0][1]**2+(S[1][1]-S[0][0])**2))
    if K == 0:
        xx00 = xmin
        xx01 = xmax
        yy00 = exact[1]
        yy01 = exact[1]
        xx10 = exact[0]
        xx11 = exact[0]
        yy10 = ymin
        yy11 = ymax
    else:
        xx00 = xmin
        xx01 = xmax
        yy00 = K * (xx00 - exact[0]) + exact[1]
        yy01 = K * (xx01 - exact[0]) + exact[1]
        xx10 = xmin
        xx11 = xmax
        yy10 = -1/K * (xx10 - exact[0]) + exact[1]
        yy11 = -1/K * (xx11 - exact[0]) + exact[1]
    r=0.5
    temp1 = r*math.sqrt((1+K*K)/(S[1][1]+K*K*S[0][0]-2*S[0][1]*K))
    temp2 = r*math.sqrt((1+(-1/K)*(-1/K))/(S[1][1]+(-1/K)*(-1/K)*S[0][0]-2*S[0][1]*(-1/K)))
    ratio = max(temp1,temp2)/min(temp1,temp2)
    if temp1 >= temp2:
        Kmax = K
    else:
        Kmax = -1/K
    #ax.plot([xx00,xx01], [yy00,yy01],'--', color = (0,0,0), label="Сечение гауссиана и\nего оси симметрии")
    #ax.plot([xx10,xx11], [yy10,yy11],'--', color = (0,0,0))
    angle = 180/math.pi * math.acos(np.sign(Kmax)/math.sqrt(1+Kmax**2))
    semiax = max(min(exact[0]-xmin,xmax-exact[0]),min(exact[1]-ymin,ymax-exact[1]))
    koef = 1
    ellipse = Ellipse((exact[0], exact[1]), width=semiax * 2 * koef, height=semiax/ratio * 2 * koef,
                      facecolor='none',linestyle='--',linewidth=1,angle=angle,edgecolor=(0,1,0),
                      label="Сечение гауссиана",zorder=2)
    ax.add_patch(ellipse)
else:
    ellipse = Ellipse((exact[0], exact[1]), width=semiax * 1, height=semiax * 2,
                  facecolor='none',linestyle='--',linewidth=2,edgecolor='green',label="Сечение гауссиана")
    ax.add_patch(ellipse)


##ax.plot(exact[0],exact[1], 'o', markersize=6, color = (1,0,0),label="Точное решение")
ax.plot(expect[0],expect[1], '*', markersize=6, color = (0,1,0),label="Выборочное среднее")
pl.legend(loc='best')
pl.grid(True)
pl.show()






