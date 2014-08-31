# -*- coding: cp936 -*-
import sqlite3
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from pandas.tools.plotting import scatter_matrix
from pandas import DataFrame
import scipy.stats as stats
from math import *
import copy
from statsmodels.tsa.stattools import *

def seqnum(year,month,day,hour,minute):
    num = 0
    days = [31,28,31,30,31,30,31,31,30,31,30,31]
    for y in range(2013,year+1):
        for m in range(1,13):
            for d in range(1,days[m-1]+1):
                if (y==2013 and m==1 and d<26):
                    continue
                num = num + 48
                if (y==year and m==month and d==day):
                    num+=hour*2
                    if minute == 30:
                        num+=1
                    return num

np.set_printoptions(precision = 5, suppress = True)

# get data from database
conn = sqlite3.connect('airweather.db')
c = conn.cursor()
c.execute("select PM25, temperature, humidity, airpressure, wind_speed, wind_direction, dew, rain from PM25")
res = c.fetchall()
numtotal = seqnum(2014,4,20,23,0)
numtotal = seqnum(2014,4,16,23,0)
dataset = []
num=0
for row in res:
    if num%2==0:
        dataset.append(list(row))
    if num == numtotal:
        break
    num+=1
conn.close()
dataset = np.array(dataset,dtype = np.str)
idx = np.where(dataset=='None')
for i in np.unique(idx[0]):
    dataset[i,1:] = dataset[i-1,1:]
idx = np.where(dataset=='')
dataset[idx] = np.nan
pm25 = np.double(dataset[:,0])
ok = -np.isnan(pm25)
xp = ok.nonzero()[0]
fp = pm25[ok]
x = np.isnan(pm25).nonzero()[0]
pm25[np.isnan(pm25)] = np.interp(x,xp,fp)
dataset = np.hstack((pm25[:,np.newaxis],dataset[:,1:]))
wind_dir = dataset[:,5]
wind_dir[wind_dir=='None'] = 'Variable'
wind_dir[np.array(wind_dir=='ENE') + np.array(wind_dir=='NNE')] = 'NE'
wind_dir[np.array(wind_dir=='ESE') + np.array(wind_dir=='SSE')] = 'SE'
wind_dir[np.array(wind_dir=='NNW') + np.array(wind_dir=='WNW')] = 'NW'
wind_dir[np.array(wind_dir=='SSW') + np.array(wind_dir=='WSW')] = 'SW'
N = np.array(wind_dir=='North', dtype = np.double)
S = np.array(wind_dir=='South', dtype = np.double)
E = np.array(wind_dir=='East', dtype = np.double)
W = np.array(wind_dir=='West', dtype = np.double)
NE = np.array(wind_dir=='NE', dtype = np.double)
SE = np.array(wind_dir=='SE', dtype = np.double)
NW = np.array(wind_dir=='NW', dtype = np.double)
SW = np.array(wind_dir=='SW', dtype = np.double)
V = np.array(wind_dir=='Variable', dtype = np.double)
dataset = np.hstack((dataset[:,(0,1,2,3,4,6,7)],N[:,np.newaxis],S[:,np.newaxis],E[:,np.newaxis],W[:,np.newaxis],
                NE[:,np.newaxis],SE[:,np.newaxis],NW[:,np.newaxis],SW[:,np.newaxis],V[:,np.newaxis]))
dataset = np.double(dataset)
print "-"*99
print "Raw dataset or raw features: "
print dataset
print "-"*99


def lassopre(p = 24,q = 24):
    pm25 = dataset[range(0,len(dataset)),0]     # pm25 don't share memory with dataset
    feas = dataset[:,1:]
    testnum = 1000
    hournum,feanum = feas.shape
    feature = np.zeros((hournum-p,p+feanum*(q+1)))
    for i in range(0,hournum-p):
        a = pm25[i:i+p]
        b = np.hstack(feas[i+p-q:i+p+1,:])
        feature[i] = np.hstack((a,b))
    response = pm25[range(p,len(pm25))]
    fea1 = feature[:-testnum]
    res1 = response[:-testnum]
    
    ls = lm.LassoCV(alphas=None, fit_intercept = True, verbose=True)
    ls.fit(fea1,res1)
    
    print "-"*99
    print "Lasso regression ..."
    print "(p,q) = (%d, %d)" % (p,q)
    print "chosen alpha is:"
    print ls.alpha_
    print "intercept is:"
    print ls.intercept_
    print "R^2 score is:"
    print ls.score(fea1,res1)
    
    plt.figure()
    plt.plot(range(0,p),ls.coef_[:p],'k-')
    plt.plot(range(p,p+feanum*(q+1)),ls.coef_[p:],'r-')
    plt.xlabel('feature')
    plt.ylabel('coeffecient')
    plt.legend(('previous pm2.5 concentration', 'weather features in the window'))
    plt.suptitle('Coefficients of Lasso')

    
    pre = ls.predict(feature)
    
    plt.figure()
    x = range(0,i+1)
    plt.plot(x,response)
    plt.plot(x[:-testnum],pre[:-testnum],'g-')
    plt.plot(x[-testnum:],pre[-testnum:],'y-')
    plt.xlabel('#hour')
    plt.ylabel('pm25(ug/m^3)')
    plt.legend((('original data','lasso regression','one hour prediction')))

    # confidence interval of the prediction
##    diff = response[:-testnum].T - pre[:-testnum]
    diff = response.T - pre
    square = [x**2 for x in diff]
    SSR = sum(square)
    var0 = (SSR/len(diff))
    print "variation is:"
    print var0
    coef = ls.coef_[:p]
    var = np.zeros((48,1))
    var[0] = var0
    for i in range(1,48):
        for j in range(0,i):
            step = i-j
            if step <= p:
                var[i] += var[i-step]*(coef[p-step])**2
        var[i] += var0
    std = [sqrt(temp) for temp in var]

    print "J_train:"
    print np.average([x**2 for x in diff[:-testnum]])
    print "J_test:"
    print np.average([x**2 for x in diff[-testnum:]])
    
    plt.figure(figsize=(10,8))
    pre = np.zeros(48)
    pre[0] = ls.predict(feature[-48])
    pm25[-48] = pre[0]
    for i in range(1,48):
        a = pm25[-p-48+i:-48+i]
        if i<47:
            b = np.hstack(feas[-q-48+i:-48+i+1,:])
        else:
            b = np.hstack(feas[-q-48+i:,:])
        feature[-48+i] = np.hstack((a,b))
        pre[i] = ls.predict(feature[-48+i])
        pm25[-48+i] = pre[i]# use predicted pm25 to predict next hour's pm25

    day1 = np.average(response[-48:-24])
    day2 = np.average(response[-24:])
    day1new = np.average(pre[-48:-24])
    day2new = np.average(pre[-24:])
    print "J_h:"
    print np.average([x**2 for x in pre-response[-48:]])
    print "J_d:"
    print ((day1-day1new)**2+(day2-day2new)**2)/2
    plt.plot(response[-48:],'o-')
    plt.plot(pre,'r*-')
    plt.xlabel('#hour of prediction')
    plt.ylabel('pm25(ug/m^3)')
    plt.legend(('48 hour ground truth', '48 hour prediction'),loc = 'best')
    plt.title('48 hour prediction via Lasso regression')
    plt.show()
    return 0

def loglassopre(p=24,q=24,alpha=0.014):
    pm25 = dataset[range(0,len(dataset)),0]     # pm25 don't share memory with dataset
    logpm25 = np.array([log(x) for x in pm25])
    feas = dataset[:,1:]
    hournum,feanum = feas.shape
    testnum = 1000
    feature = np.zeros((hournum-p,p+feanum*(q+1)))
    for i in range(0,hournum-p):
        a = logpm25[i:i+p]
        b = np.hstack(feas[i+p-q:i+p+1,:])
        feature[i] = np.hstack((a,b))
    response = logpm25[range(p,len(pm25))]
    print "-"*99
    print "Lasso regression of log(Pm2.5)..."
    print "(p,q): (%d,%d)"%(p,q)
    
    fea1 = feature[:-testnum]
    res1 = response[:-testnum]
    ls = lm.LassoCV(alphas=[alpha], fit_intercept = True, verbose=True)
    ls.fit(fea1,res1)
    
    print "chosen alpha is:"
    print ls.alpha_
    print "intercept is:"
    print ls.intercept_
    print "R^2 score is:"
    print ls.score(fea1,res1)
    
    plt.figure()
    plt.plot(range(0,p),ls.coef_[:p],'k-')
    plt.plot(range(p,p+feanum*(q+1)),ls.coef_[p:],'r-')
    plt.xlabel('feature')
    plt.ylabel('coeffecient')
    plt.legend(('previous pm2.5 concentration', 'weather features in the window'))
    plt.suptitle('Coefficients of log Lasso')
    prelog = ls.predict(feature)
    pre = np.array([exp(x) for x in prelog])

    truth = pm25[p:]
    diff = truth - pre
    print "J_train:"
    Jtrain = np.average([x**2 for x in diff[:-testnum]])
    print np.average([x**2 for x in diff[:-testnum]])
    print "J_test:"
    Jtest = np.average([x**2 for x in diff[-testnum:]])
    print np.average([x**2 for x in diff[-testnum:]])
    
    plt.figure()
    t = range(0,hournum-p)
    plt.plot(t,np.array([exp(x) for x in response]))
    plt.plot(t[:-testnum],pre[:-testnum],'g-')
    plt.plot(t[-testnum:],pre[-testnum:],'y-')
    plt.xlabel('#hour')
    plt.ylabel('pm25(ug/m^3)')
    plt.legend((('original data','log lasso regression','one hour prediction')))

    # confidence interval of the prediction
    diff = response.T - prelog
    square = [x**2 for x in diff]
    SSR = sum(square)
    var0 = (SSR/len(diff))
    print "variation is:"
    print var0
    coef = ls.coef_[:p]
    var = np.zeros((48,1))
    var[0] = var0
    for i in range(1,48):
        for j in range(0,i):
            step = i-j
            if step <= p:
                var[i] += var[i-step]*(coef[p-step])**2
        var[i] += var0
    std = [sqrt(temp) for temp in var]      

    plt.figure(figsize=(10,8))
    logpre = np.zeros(48)
    logpre[0] = ls.predict(feature[-48])
    logpm25[-48] = logpre[0]
    for i in range(1,48):
        a = logpm25[-p-48+i:-48+i]
        if i<47:
            b = np.hstack(feas[-q-48+i:-48+i+1,:])
        else:
            b = np.hstack(feas[-q-48+i:,:])
        feature[-48+i] = np.hstack((a,b))
        logpre[i] = ls.predict(feature[-48+i])
        logpm25[-48+i] = logpre[i]# use predicted pm25 to predict next hour's pm25
    plt.plot(np.array([exp(x) for x in response[-48:]]),'o-')
    expectation = np.array([exp(x) for x in logpre])
    # bound w.p. 95%
    upper = np.array([exp(x) for x in logpre+std])
    lower = np.array([exp(x) for x in logpre-std])
    meanint = np.mean(upper-lower)
    meanvar = sqrt(np.mean(np.array([x**2 for x in pm25[-48:]-expectation])))
    day1pre = np.mean(expectation[:-48:-24])
    day2pre = np.mean(expectation[-24:])
    day1 = np.mean(pm25[-48:-24])
    day2 = np.mean(pm25[-24:])
    print "L_h:"
    print meanint
    print "J_h:"
    Jh = meanvar**2
    print meanvar**2
    print "J_d:"
    Jd = ((day1-day1pre)**2+(day2-day2pre)**2)/2
    print ((day1-day1pre)**2+(day2-day2pre)**2)/2
    
    plt.plot(np.array([exp(x) for x in logpre]),'r*-')
##    plt.plot(np.array([exp(x) for x in logpre+std]))
##    plt.plot(np.array([exp(x) for x in logpre-std]))
    plt.xlabel('#hour of prediction')
    plt.ylabel('pm25(ug/m^3)')
    plt.legend(('48 hour ground truth', '48 hour prediction'),loc = 'best')
    plt.title('48 hour prediction via Lasso regression of log(PM2.5)')
    plt.show()
    return ls.coef_,ls.score(fea1,res1),Jtrain,Jtest,Jh,Jd,meanint

def mulloglasso():
    # generative model 48 hour multi Gaussian distribution
    print "-"*99
    print """
    Prediction by Multivariate Regression Model ...
    Sample:
    (y0,y1,...,y47,w0,w1,...,w47)
    where yi means pm2.5 concentration, while wi means vector of meteorological measuements
    Learn:
    A multivariate Gaussian distribution of (y24,y25,...,y47|y0,t1,...,y23,w0,w1,...,w47)"""
    daynum = (dataset.shape[0])/24
    pm = dataset[range(0,len(dataset)),0]
    logpm = np.array([log(x) for x in pm])
##    logpm = pm
    Y = []
    X = []
    for i in range(0,len(dataset)-48-24):
        Y.append(logpm[i+24:i+48+24])
        X.append(np.hstack((np.hstack(logpm[i:i+24]),np.hstack(dataset[i:i+48+24,1:]))))

    Y = np.array(Y)
    X = np.array(X)
    print "X:"
    print X,X.shape
    print "Y:"
    print Y,Y.shape
    
##    ls = lm.Lasso(alpha = 0.005, fit_intercept = True, warm_start = True, max_iter = 2000)
##    ls.fit(X,Y)
##    with open('multi.pickle','w') as f:
##        pickle.dump(ls,f)
    with open('multi.pickle') as f:
        ls = pickle.load(f)
    
    print "R^2 is:"
    print ls.score(X,Y)
    print "ls.coef_:"
    print ls.coef_,ls.coef_.shape

##    Y_pre = ls.predict(X)
##    y1 = ls.predict(X[-2])
##    X[-1,range(0,24)] = y1
##    y2 = ls.predict(X[-1])

    # variation
    Y_hat = ls.predict(X)
    diff = Y-Y_hat
    sigma = np.dot(diff.T,diff)/len(Y)
    print "sigma is: "
    print sigma

    sigmas = np.zeros((48))
    for i in range(0,48):
        sigmas[i] = sigma[i,i]
    print sigmas

    f = open('store.pickle','w')
    pickle.dump(sigma,f)
    f.close()
    
    plt.figure()
    plt.imshow(ls.coef_, interpolation='none', cmap = cm.coolwarm,
               origin = 'lower', aspect = 'auto',
               vmax = abs(ls.coef_).max(), vmin = -abs(ls.coef_).max())
    cbar = plt.colorbar()
    cbar.set_label('Coeffecient value')
    plt.xlabel('Feature')
    plt.ylabel('Time')
    plt.suptitle('Coeffecients Matrix')
##    plt.savefig('./plots/multiceoff.eps')
##    plt.savefig('./plots/multiceoff.pdf')
    
    plt.figure(figsize=(10,8))
    plt.plot([exp(x) for x in np.hstack(logpm[-48:])],'o-')
##    plt.plot([exp(x) for x in np.hstack((y1,y2))],'r*-')
    ypre = ls.predict(X[-1])
    y_low = ypre - sigmas
    y_up = ypre + sigmas
    plt.plot([exp(x) for x in ypre],'r*-')
    plt.plot([exp(x) for x in y_low])
    plt.plot([exp(x) for x in y_up])
    plt.xlabel('#hour of prediction')
    plt.ylabel('pm25(ug/m^3)')
    plt.legend(('48 hour ground truth', '48 hour prediction'), loc = 'best')
    plt.title('48 hours prediction via multivariate regression')
##    plt.savefig('./plots/multipre.eps')
##    plt.savefig('./plots/multipre.pdf')
    plt.show()

    # J,L
    truth = pm[-48:]
    pre = [exp(x) for x in ypre]
    diff = pre - truth
    day1 = np.mean(truth[-48:-24])
    day2 = np.mean(truth[-24:])
    day1pre = np.mean(pre[-48:-24])
    day2pre = np.mean(pre[-24:])
    print "J_h:"
    print np.mean([x**2 for x in diff])
    print "J_d:"
    print ((day1-day1pre)**2+(day2-day2pre)**2)/2
    ylow = np.array([exp(x) for x in y_low])
    yup = np.array([exp(x) for x in y_up])
    print "L_h:"
    print np.mean(-ylow+yup)
    
def main():
    #lassopre(4,2)
##    for p in range(0,30,5):
##        for q in range(0,p,5):
##            print p,q
##            loglassopre(p,q)
    lassopre(10,10)
    loglassopre(10,10,0.0144)
##    mulloglasso()
    anss = []
    #ls.coef_,ls.score(fea1,res1),Jtrain,Jtest,Jh,Jd,meanint
##    for alpha in np.linspace(0.010,0.019,num=10):
##        ans = loglassopre(24,24,alpha)
##        anss.append(list(ans))
##    with open('anss.pickle','w') as f:
##        pickle.dump(anss,f)
##    for p in range(0,40,5):
##        ans = loglassopre(p,p)
##        anss.append(list(ans))
##    with open('pq.pickle','w') as f:
##        pickle.dump(anss,f)
##    ans = loglassopre(24,24,0.1)
##    print ans[0][22:24],ans[0][-15:]
        
if __name__ == '__main__':
    main()

