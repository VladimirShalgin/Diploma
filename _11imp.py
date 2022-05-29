from dwave.system import DWaveSampler, EmbeddingComposite
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
def to_binary_string(x,r): # десятичное число в двоичное число-строку
    binary = '' if x > 0 else '-'
    x = math.fabs(x)
    binary += str(bin(math.floor(x)))[2:]+'.'
    x -= math.floor(x)
    for i in range(r):
        x *= 2
        binary += str(math.floor(x))
        x -= math.floor(x)
        if x == 0:
            break
    
    return binary

def to_decimal(x,r,c,d): # x - двоичный массив числа из [0,2)
    num = 0
    for i in range(r):
        num = num + x[i]/2**i
    
    return c * num - d

def find_l(x):  # ищем l такое, что 1/4<(2^l)*x<=1/2
    l = -20
    z = 2**l * math.fabs(x)
    while z<=1/4:
        z *= 2
        l += 1
    
    return l

def call_DWave(a,b,c,d,R):  # вызов D-Wave, выдаёт двоичный массив числа из [0,2) 
    Q = {(0, 0): 0}
    for i in range(R):
        temp1 = a*c/(2**i)
        for j in range(i,R):
            temp2 = a*c/(2**j)
            if j==i:
                Q[(i, j)] = temp1*(temp1 - 2*(a*d+b))
            else:
                Q[(i, j)] = 2*temp1*temp2
    
    sampler_auto = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))
    sampleset = sampler_auto.sample_qubo(Q, num_reads = 1)
    solution = sampleset.first[0]
    sol = []
    for i in range(R):
        sol.append(solution[i])
    return sol

########################################
def solve(a,b,R=5,n=10):
    assert math.fabs(b) < math.fabs(a), "|b| должно быть меньше |a|"
    a1=a
    b1=b
    if a<0:
        a*=-1
        b*=-1
    k = find_l(a)
    a*=2**k
    b*=2**k
    c=d=1
    x_accurate = b / a
    x = []
    x.append(x_accurate)
    #print('Приближения')
    xk = 0
    lk = 0
    for k in range(1, n+1):
        solution = call_DWave(a, 2**lk * (b-a*xk), c, d, R)
        Deltak = 2**(-lk) * to_decimal(solution,R,c,d)
        xk += Deltak
        x.append(xk)
        lk = find_l(b-a*xk)
        print("%d: %s, l_k = %d" % (k, xk, lk))
    
    #print('')
    #print('Точное решение')
    #print('0:',x[0])
    #print('Приближения')
    #for i in range(1,n+1):
    #    print('{}: {}'.format(i, x[i]))
    log_diff = np.empty([n], dtype = float)
    for i in range(n):
        log_diff[i]=math.log(math.fabs(x[i+1]-x[0]))
    
    linear = LinearRegression()
    steps = np.arange(n).reshape((-1,1))
    linear.fit(steps,log_diff)
    shift = linear.intercept_
    coef = linear.coef_[0]
    #print("coef=%g\nshift=%g" % (coef,shift))
    beta = 0.25 * np.e**(-2*coef-np.euler_gamma)
    #print("R=%d\nbeta=%g" % (R,beta))
    #if shift<0:
    #    text = ""+str(round(coef,3))+"k"+str(round(shift,3))
    #else:
    #    text = ""+str(round(coef,3))+"k+"+str(round(shift,3))
    fig = plt.figure(figsize=(5,5))
    plt.plot(np.arange(n),log_diff,'or',label="Логарифмы ошибок")
    plt.plot([0,n-1],[shift, coef*(n-1) + shift],'b',label="Линейная регрессия")
    plt.grid(True)
    plt.legend(loc='best')
    name = 'improve_'+str(a1)+'x='+str(b1)+'_R='+str(R)+'_n='+str(n)+'.png'
    fig.savefig(name)
    fig = plt.figure(figsize=(5,5))
    plt.plot(np.arange(n),log_diff,'or',label="Error logarithms")
    plt.plot([0,n-1],[shift, coef*(n-1) + shift],'b',label="Linear regression")
    plt.grid(True)
    plt.legend(loc='best')
    name = 'improve_'+str(a1)+'x='+str(b1)+'_R='+str(R)+'_n='+str(n)+'_en.png'
    fig.savefig(name)

for i in [5,10,15,20,25]:
    solve(3,1,R=i,n=20)
