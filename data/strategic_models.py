import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def binomial_coefficient(n_, k_):
    """
    Computes the coefficient of the x k-term in the polynomial expansion of the binomial power (1 + x) n
    Appropriately corrects for special cases, unlike the numpy implementation
    Parameters:
    ----------
    n: int
    k:int
    Returns:
    -------
    Binomial Coefficient.
    """
    n = float(n_)
    k = float(k_)
    if (n == -1 and k > 0):
        return -1.0 ** k
    if (n == -1 and k < 0):
        return (-1.0 ** k) * -1.0
    if (n == -1 and k == 0):
        return 1.0
    if (n == k):
        return 1.0
    if n >= k:
        if k > n - k:  # take advantage of symmetry
            k = n - k
        c = 1
        for i in range(int(k)):
            c = c * (n - i)
            c = c // (i + 1)
        return float(c)
    else:
        return 0.0

def theta(x):
    if x < 0.0:
        return 0.0
    else:
        return 1.0

def PiD(k, c, F, N, M):
    return ((k * F * c) / float(N)) * theta(k - M)

def PiC(k, c, F, N, M):
    return PiD(k, c, F, N, M) - c

def Fc(x, c, F, N, M):
    total = 0.0
    for k in range(0, N):
        binomial = binomial_coefficient(N - 1, k)
        probability = (x ** k) * ((1.0 - x) ** (N - 1.0 - k))
        total += binomial * probability * PiC(k + 1, c, F, N, M)
    return total

def Fd(x, c, F, N, M):
    total = 0.0
    for k in range(0, N):
        binomial = binomial_coefficient(N - 1, k)
        probability = (x ** k) * ((1.0 - x) ** (N - 1.0 - k))
        total += binomial * probability * PiD(k, c, F, N, M)
    return total

def replicator(x, c, F, N, M):
    average = x*Fc(x, c, F, N, M) + (1.0-x)*Fd(x, c, F, N, M)
    return x*(Fc(x, c, F, N, M) - average)

def payoff_difference(x, c, F, N, M):
    return Fc(x, c, F, N, M) - Fd(x, c, F, N, M)

def find_roots(x, y,c, F, N, M):
    l = []
    s = np.abs(np.diff(np.sign(y))).astype(bool)

    for i in range(len(s)):
        if s[i] == True:
            l.append([i, np.gradient(replicator(x, c, F, N, M))[i]])
    return l


def find_probability(c,F,N,M):


        # return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)

    xx = list()
    xy = list()
    line_0 = list()

    x_line = np.linspace(0.0, 1.0, 100)
    y_line = []
    for x in x_line:
        y = replicator(x, c, F, N, M)
        #y = payoff_difference(x, c, F, N, M)
        y_line.append(y)
    #plt.plot(x_line, y_line)
    #plt.plot(x_line, np.gradient(y_line))
    #plt.grid(True)
    #plt.ylim(-1.0, 1.0)

    z = find_roots(x_line, y_line,c, F, N, M)

    sink_list = [[0, 0]]

    for i in range(1, len(z)):
        sink_list.append(z[i])
        sink_list[-1][1] = 0

    if sink_list[-1][0] != 100:
        sink_list.append([100, 0])



    index_sink = 0
    neg = True
    for i in range(1, 101):
        i = i / 100
        temp = replicator(i, c, F, N, M)

        if temp > 0:
            neg = False

        if temp < 0:
            if neg == False:
                index_sink += 2
                neg = True

        if neg:
            sink_list[index_sink][1] += 1 / 100
        else:
            sink_list[index_sink + 2][1] += 1 / 100


    max_num = [0, 0]
    for i in range(len(sink_list)):
        if sink_list[i][1] > max_num[1]:
            max_num = sink_list[i]
    return(max_num[0]/100)

    # plt.plot(x,y)
    # plt.plot(z, np.zeros(len(z)), marker="o", ls="", ms=4)