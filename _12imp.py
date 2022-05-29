from dwave.system import DWaveSampler, EmbeddingComposite
import numpy as np
import matplotlib.pyplot as pl
from sklearn.linear_model import LinearRegression
import numpy.linalg as lin

def to_dec(x,n,R,c,d):
    sol = np.empty([n,1], dtype=float)
    for j in range(n):
        sol[j][0] = 0
        for s in range(R):
            sol[j][0] = sol[j][0] + x[j*R+s] / (2**s)
        sol[j][0] = c * sol[j][0] - d
    
    return sol

def find_l(x):  # ищем l такое, что 1/4<(2^l)*x<=1/2
    l = -20
    z = 2**l * np.fabs(x)
    while z<=1/2:
        z *= 2
        l += 1
    
    return l

def DWave(A,b,c,d,R,N,alpha=0): # вычисление коэффициентов Гамильтониана и вызов D-Wave
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
            i = int(np.floor(el/R))
            s = el-i*R
            j = int(np.floor(p/R))
            r = p-j*R
            if el==p:
                Q[(el,p)] = pows[s] * (pows[s]*(alpha + Aij[i][j]) - 2*Bi[i] - 2*d*(alpha + Ci[i]))
            else:
                if i==j:
                    Q[(el,p)] = 2*pows[r]*pows[s] * (alpha + Aij[i][j])
                else:
                    Q[(el,p)] = 2*pows[r]*pows[s] * Aij[i][j]
    
    sampler_auto = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))
    sampleset = sampler_auto.sample_qubo(Q,
                                         num_reads = N,
                                         num_spin_reversal_transforms=int(np.floor(N/100)))     
    return to_dec(sampleset.first[0],n,R,c,d)

def solve(a,b,c=1,d=1,R=6,Iter=3,N=2):
    n = len(a)
    a=np.array([a])
    norm = lin.norm(a)
    x = []  # здесь последовательные приближения
    log_errors = np.empty([Iter],dtype=float)
    xk = np.zeros([n,1], dtype=float)
    lk = 0
    for k in range(Iter):
        xk += (2**(-lk)) * DWave(a,(2**lk)*(b-a.dot(xk)), c,d,R,(20 if k==0 else N))
        x.append(xk)
        print("Ошибка = %g" % (np.fabs(a.dot(xk)-b)/norm))
        log_errors[k]=np.log(np.fabs(a.dot(xk)-b)/norm)
        lk = find_l(a.dot(xk)-b)
    
    linear = LinearRegression()
    steps = np.arange(Iter).reshape(-1, 1)
    linear.fit(steps,log_errors)
    shift = linear.intercept_
    coef = linear.coef_[0]
    fig = pl.figure(figsize=(5,5))
    if n==2:
        text = "Логарифмы расстояний до прямой"
    
    if n==3:
        text = "Логарифмы расстояний\nдо плоскости"
    
    if n>3:
        text = "Логарифмы расстояний\nдо гиперплоскости"
    
    pl.plot(np.arange(Iter),log_errors,'or',label=text)
    pl.plot([0,Iter-1],[shift, coef*(Iter-1) + shift],'b',label="Линейная регрессия")
    pl.grid(True)
    pl.legend(loc='best')
    name = 'improve1'+str(n)+'_logs_['+str(a[0][0])
    for i in range(1,n):
        name += ','+str(a[0][i])
    
    name += ']x='+str(b)+'_Iter='+str(Iter)+'_R='+str(R)+'_N='+str(N)+'.png'
    fig.savefig(name)
    print("Норма последнего приближения: %g" % lin.norm(x[Iter-1]))
    pl.show()

linear = LinearRegression()
steps = np.arange(Iter).reshape((-1,1))
linear.fit(steps,log_errors)
shift = linear.intercept_
coef = linear.coef_[0]
fig = pl.figure(figsize=(5,5))
fig = pl.figure(figsize=(5,5))
if n==2:
    text = "Логарифмы расстояний до прямой"

if n==3:
    text = "Логарифмы расстояний\nдо плоскости"

if n>3:
    text = "Логарифмы расстояний\nдо гиперплоскости"

pl.plot(np.arange(Iter),log_errors,'or',label=text)
pl.plot([0,Iter-1],[shift, coef*(Iter-1) + shift],'b',label="Линейная регрессия")
pl.grid(True)
pl.legend(loc='best')
name = 'improve1'+str(n)+'_logs_['+str(a[0][0])
for i in range(1,n):
    name += ','+str(a[0][i])

name += ']x='+str(b)+'_Iter='+str(Iter)+'_R='+str(R)+'_N='+str(N)+'.png'
fig.savefig(name)
print("Норма последнего приближения: %g" % lin.norm(x[Iter-1]))
pl.show()

solve([0.1,-0.7,0.3],-0.3,R=7,Iter=20)
##def call_DWave(a,b,c,h,d,R,N):  # вызов D-Wave, выдаёт два числа 
##    Q = {(0,0): 0}  # вычисление коэффициентов Гамильтониана
##    
##    alpha=0
##    for el in range(2*R):
##        for p in range(el,2*R):
##            k = math.floor(el/R)
##            i = el-k*R
##            m = math.floor(p/R)
##            j = p-m*R
##            pow2ih = h/(2**i)
##            if el==p:
##                if k==0:
##                    Q[(el,p)] = pow2ih * (pow2ih*(a*a + alpha) - 2*a*c - 2*d*(a*(a+b)+alpha))
##                if k==1:
##                    Q[(el,p)] = pow2ih * (pow2ih*(b*b + alpha) - 2*b*c - 2*d*(b*(a+b)+alpha))
##            else:
##                pow2jh=h/(2**(j-1))
##                if k==m:
##                    if k==0:
##                        Q[(el,p)] = pow2ih*pow2jh*(a*a+alpha)
##                    if k==1:
##                        Q[(el,p)] = pow2ih*pow2jh*(b*b+alpha)
##                else:
##                    Q[(el,p)] = pow2ih*pow2jh*a*b
##    
##    sampler_auto = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))
##    sampleset = sampler_auto.sample_qubo(Q, num_reads = N)
##    sol_array = sampleset.first[0]
##    sol = np.empty([2], dtype=float)   # здесь компоненты решения
##    for i in range(2):      # преобразуем в десятичное каждую компоненту
##        sol[i] = 0
##        for s in range(R):
##            sol[i] = sol[i] + sol_array[i*R+s] / (2**s)
##        sol[i] = h * sol[i] - d   # масштабируем и смещаем
##    
##    return sol

##    Left = min(x[n-3:n])
##    Right = max(x[n-3:n])
##    lag=0.01
##    xi=np.arange(Left, Right+lag, lag)
##    yi=np.empty(len(xi), dtype=float)
##    for i in range(len(xi)):
##        yi[i]=(c-a*xi[i])/b
##    
##    fig = pl.figure(figsize=(5,5))
##    pl.plot(xi,yi,color=(0,0,1))
##    pl.plot(x[n-3:n], y[n-3:n],color=(0,0,0))
##    pl.plot(x[n-3:n], y[n-3:n], 'o',markersize=3,color=(0,0,0), label='Улучшения')
##    pl.plot(x[n-1], y[n-1], 'o',color=(1,0,0),label='Последнее улучшение')
##    pl.legend(loc='best')
##    pl.grid(True)
##    name = 'improve12_'+str(a1)+'x'+('+' if b>0 else '')+str(b1)+'y='+str(c1)+'_n='+str(n)+'_R='+str(R)+'_N='+str(N)+'.png'
##    fig.savefig(name)
