from dwave.system import DWaveSampler, EmbeddingComposite
import math
import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import gaussian_kde
def calc_Q(a,b,с,h,d,R):   
    Q = {(0,0): 0}
    
    alpha = 0
    for el in range(2*R):
        for p in range(el,2*R):
            k = math.floor(el/R)
            i = el-k*R
            m = math.floor(p/R)
            j = p-m*R
            pow2ih = h/(2**i)
            if el==p:
                if k==0:
                    Q[(el,p)] = pow2ih * (pow2ih*(a*a + alpha) - 2*a*c - 2*d*(a*(a+b)+alpha))
                if k==1:
                    Q[(el,p)] = pow2ih * (pow2ih*(b*b + alpha) - 2*b*c - 2*d*(b*(a+b)+alpha))
            else:
                pow2jh=h/(2**(j-1))
                if k==m:
                    if k==0:
                        Q[(el,p)] = pow2ih*pow2jh*(a*a+alpha)
                    if k==1:
                        Q[(el,p)] = pow2ih*pow2jh*(b*b+alpha)
                else:
                    Q[(el,p)] = pow2ih*pow2jh*a*b
    
    return Q

def to_dec(x,n,R,c,d):
    sol = np.empty([n], dtype=float)
    for j in range(n):  # преобразуем в десятичное каждую компоненту 
        sol[j] = 0
        for s in range(R):
            sol[j] = sol[j] + x[j*R+s] / (2**s)
        sol[j] = c * sol[j] - d
    
    return sol


a = 20
b = 3
c = 3
N = 5000
R = 8
h = 1   
d = 1
norm = math.sqrt(a*a+b*b)
n=2
sampler_auto = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))
Q = calc_Q(a,b,c,h,d,R)
sampleset = sampler_auto.sample_qubo(Q, num_reads = N, num_spin_reversal_transforms = math.floor(N/100))
sam = sampleset.record

x = np.empty([N,n], dtype=float)
sol = np.empty([n], dtype=float)
j = 0
for i in range(len(sam)):
    x[i+j] = to_dec(sam[i][0],n,R,h,d)
    for k in range(sam[i][2]-1):
        j = j + 1
        x[i+j] = to_dec(sam[i][0],n,R,h,d)

expect = sum(x)/N
shifts = []
for i in range(N):
    shifts.append((a*x[i][0]+b*x[i][1]-c)/norm)

expect_shift = sum(shifts)/N
xmin = -d
xmax = 2*h-d
ymin = -d
ymax = 2*h-d
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([x.T[0], x.T[1]])
kernel = gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

xi=np.array([xmin, xmax])
yi=np.empty(len(xi),dtype=float)
for i in range(len(xi)):
    yi[i]=(c-a*xi[i])/b

fig = pl.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.set(xlim = [xmin, xmax],ylim = [ymin, ymax])
ax.plot(x.T[0], x.T[1], 'k.', markersize=1)
ax.imshow(np.rot90(Z), cmap=pl.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
ax.plot(xi, yi, color = (1,0,0))
ax.plot(expect[0],expect[1], '*', markersize=6, color=(0,1,0),label="Выборочное среднее")
pl.legend(loc='best')
pl.grid(True)
name = 'selection_'+str(a)+'x'+('+' if b>0 else '')+str(b)+'y='+str(c)+'_N='+str(N)+'_R='+str(R)+'.png'
fig.savefig(name)

density = gaussian_kde(shifts)
Left=min(shifts)
Right=max(shifts)
numberOfBins = 1 + math.floor(math.log2(N))
lag = 0.01
xi=np.arange(Left, Right, lag)
fig = pl.figure(figsize=(5,5))
pl.hist(shifts, density=True, bins=numberOfBins,
         range=(Left,Right),
         color='c',histtype='stepfilled',label="Гистограмма частот\nотклонений от прямой")
pl.plot(xi,density(xi),color=(0,0,1),label="Гауссовская ЯОП")
pl.legend(loc='best')
pl.grid(True)
name = 'projection_'+str(a)+'x'+('+' if b>0 else '')+str(b)+'y='+str(c)+'_N='+str(N)+'_R='+str(R)+'.png'
fig.savefig(name)
pl.show()
