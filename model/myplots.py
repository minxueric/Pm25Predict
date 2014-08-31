# -*- coding: cp936 -*-
import sqlite3
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

print len(dataset)

plt.figure()
plt.plot(dataset[:,0])
plt.xlim((0,len(dataset[:,0]-1)))
plt.xlabel('#hour')
plt.ylabel('PM2.5(ug/m^3)')
plt.suptitle('PM2.5 concentration historical data')
##plt.savefig('./plots/trend.pdf')
plt.show()
### autocorrelation for pm25
##temp = copy.copy(dataset[:,0])
##autocor, confint = acf(temp,unbiased=False, nlags=96, confint=None, qstat=False, fft=True, alpha=0.95)
##plt.figure()
##ax1 = plt.subplot(211)
##ax2 = plt.subplot(212)
##plt.sca(ax1)
##plt.plot(temp)
##plt.xlabel('#hour')
##plt.ylabel('pm2.5(ug/m^3)')
##plt.sca(ax2)
##plt.plot(autocor)
##plt.xlabel('k')
##plt.ylabel('correlation')
##plt.suptitle('Autocorrelation of PM2.5 concentration sequence')
##plt.savefig('./plots/autocorr.eps')
##plt.savefig('./plots/autocorr.pdf')

### cross correlation of PM2.5 vs other meteorological features
##t = copy.copy(dataset[:,1])
##h = copy.copy(dataset[:,2])
##ws = copy.copy(dataset[:,4])
##d = copy.copy(dataset[:,5])
##crosscor = ccf(temp,t,unbiased=False)
##plt.figure()
##plt.plot(crosscor)
##plt.show()

### relationship between features
##df = DataFrame(dataset[:1000,0:6],columns = ['pm25','temperature','humidity','pressure','wind_speed','dewpoint'])
##axes = scatter_matrix(df,alpha=0.2,figsize=(8,8),ax=None,marker='.',range_padding=0,diagonal='kde')
##for i in [0,2,4]:
##    ax = axes[i,0]
##    ax.yaxis.set_visible(False)
##for i in [1,3,5]:
##    ax = axes[5,i]
##    ax.xaxis.set_visible(False)
##plt.suptitle('Correlation scatterplot matrix',fontsize=15)
##plt.savefig('./plots/corr.eps')
##
##logset = copy.copy(dataset[:1000,0:6])
##logset[:,0] = [log(x) for x in logset[:,0]]
##df = DataFrame(logset,columns = ['log(pm25)','temperature','humidity','pressure','wind_speed','dewpoint'])
##axes = scatter_matrix(df,alpha=0.2,figsize=(8,8),ax=None,marker='.',range_padding=0,diagonal='kde')
##for i in [0,2,4]:
##    ax = axes[i,0]
##    ax.yaxis.set_visible(False)
##for i in [1,3,5]:
##    ax = axes[5,i]
##    ax.xaxis.set_visible(False)
##plt.suptitle('Correlation scatterplot matrix of logfeatures',fontsize=15)
##plt.savefig('./plots/corrlog.pdf')
##
##plt.show()
##testnum = 2000
##
### naive linear regression
##trainset = dataset[:-testnum,:]
##testset = dataset[-testnum:,:]
##lr = lm.LinearRegression(fit_intercept=True, normalize=True)
##lr.fit(trainset[:,1:],trainset[:,0])
##print "-"*99
##print "Naive linear regression ..."
##print "Features: temperature|humidity|airpressure|wind_speed|dew|rain|wind_speed(categorized into 9)"
##print "Response: hourly pm25"
##print "coef: "
##print lr.coef_
##print "intercept: "
##print lr.intercept_
##print "R^2: "
##print lr.score(trainset[:,1:],trainset[:,0])
##print "-"*99
##y_pre = lr.predict(dataset[:,1:])
##plt.figure()
##y = dataset[:,0]
##x = range(0,len(y))
##plt.plot(x,y)
##plt.xlabel('#hour')
##plt.ylabel('PM25(ug/m^3)')
##plt.plot(x[:-testnum],y_pre[:-testnum],'g-')
##plt.plot(x[-testnum:],y_pre[-testnum:],'y-')
##plt.legend(('original data','basic linear regression','one hour prediction'))
##plt.title('Naive linear regression')
##plt.savefig('./plots/naive.eps')
##plt.savefig('./plots/naive.pdf')
##
### advanced linear regression
##pm = dataset[:,0]
##newset = np.hstack((dataset[1:,:],pm[0:-1,np.newaxis]))
##print "-"*99
##print "Advanced linear regression ..."
##print "Features: Original features plus last pm25 value"
##print "Response: hourly pm25"
##print "New dataset is:"
##print newset
##trainset = newset[:-testnum,:]
##testset = newset[-testnum:,:]
##lr = lm.LinearRegression(fit_intercept=True, normalize=True)
##lr.fit(trainset[:,1:],trainset[:,0])
##print "coef_:"
##print lr.coef_
##print "intercept: "
##print lr.intercept_
##print "R^2: "
##print lr.score(trainset[:,1:],trainset[:,0])
##print "-"*99
##plt.figure()
##y = newset[:,0]
##x = range(0,len(y))
##plt.plot(x,y)
##plt.xlabel('#hour')
##plt.ylabel('PM25(ug/m^3)')
##y_pre = lr.predict(newset[:,1:])
##plt.plot(x[:-testnum],y_pre[:-testnum],'g-')
##plt.plot(x[-testnum:],y_pre[-testnum:],'y-')
##plt.legend(('original data','advanced linear regression','one hour prediction'))
##plt.title('Advanced linear regression')
##plt.savefig('./plots/adlinear.eps')
##plt.savefig('./plots/adlinear.pdf')
##
### predict 48 hours
##pre = np.zeros(48)
##pre[0] = lr.predict(newset[-48,1:])
##for i in range(1,48):
##    x_vector = newset[-48+i,1:]
##    x_vector[-1] = pre[i-1]
##    pre[i] = lr.predict(x_vector)
##plt.figure(figsize=(10,8))
##plt.plot(range(0,48),newset[-48:,0],'o-')
##plt.plot(range(0,48),pre,'r*-')
##plt.xlabel('#hour of prediction')
##plt.ylabel('pm25(ug/m^3)')
##plt.legend(('48 hour ground truth', '48 hour prediction'), loc = 'best')
##plt.title('48 hour prediction via linear regression')
##plt.savefig('./plots/linearpre.eps')
##plt.savefig('./plots/linearpre.pdf')
##
### Lasso regression for more complicated model
##pm25 = dataset[range(0,len(dataset)),0]     # pm25 don't share memory with dataset
##feas = dataset[:,1:]
##hournum,feanum = feas.shape
##delay = 24
##feature = np.zeros((hournum-delay,delay+feanum*(delay+1)))
##for i in range(0,hournum-delay):
##    a = pm25[i:i+delay]
##    b = np.hstack(feas[i:i+delay+1,:])
##    feature[i] = np.hstack((a,b))
##response = pm25[range(delay,len(pm25))]
##print "-"*99
##print "Lasso regression ..."
##print "Features: combined feature vectors in a time window with lenth of delay %d"%delay
##print feature
##print "Response: pm25 concentration per hour"
##print response
##fea1 = feature[:-testnum]
##res1 = response[:-testnum]
##ls = lm.LassoCV(alphas=None, fit_intercept = True, verbose=True)
##ls.fit(fea1,res1)
##print "chosen alpha is:"
##print ls.alpha_
##print "intercept is:"
##print ls.intercept_
##print "R^2 score is:"
##print ls.score(fea1,res1)
##plt.figure()
##plt.plot(range(0,delay),ls.coef_[:delay],'k-')
##plt.plot(range(delay,delay+feanum*(delay+1)),ls.coef_[delay:],'r-')
##plt.xlabel('feature')
##plt.ylabel('coeffecient')
##plt.legend(('previous pm2.5 concentration', 'weather features in the window'))
##plt.suptitle('Coefficients of Lasso')
##plt.savefig('./plots/coeflasso.eps')
##plt.savefig('./plots/coeflasso.pdf')
##pre = ls.predict(feature)
##plt.figure()
##x = range(0,i+1)
##plt.plot(x,response)
##plt.plot(x[:-testnum],pre[:-testnum],'g-')
##plt.plot(x[-testnum:],pre[-testnum:],'y-')
##plt.xlabel('#hour')
##plt.ylabel('pm25(ug/m^3)')
##plt.legend((('original data','lasso regression','one hour prediction')))
##plt.savefig('./plots/lasso.eps')
##plt.savefig('./plots/lasso.pdf')
##plt.figure(figsize=(10,8))
##pre = np.zeros(48)
##pre[0] = ls.predict(feature[-48])
##pm25[-48] = pre[0]
##for i in range(1,48):
##    a = pm25[-delay-48+i:-48+i]
##    if i<47:
##        b = np.hstack(feas[-delay-48+i:-48+i+1,:])
##    else:
##        b = np.hstack(feas[-delay-48+i:,:])
##    feature[-48+i] = np.hstack((a,b))
##    pre[i] = ls.predict(feature[-48+i])
##    pm25[-48+i] = pre[i]# use predicted pm25 to predict next hour's pm25
##plt.plot(response[-48:],'o-')
##plt.plot(pre,'r*-')
##plt.ylim((0,160))
##plt.xlabel('#hour of prediction')
##plt.ylabel('pm25(ug/m^3)')
##plt.legend(('48 hour ground truth', '48 hour prediction'),loc = 'best')
##plt.title('48 hour prediction via Lasso regression')
##plt.savefig('./plots/lassopre.eps')
##plt.savefig('./plots/lassopre.pdf')


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
Y = []
X = []
for i in range(1,daynum):
    Y.append(dataset[i*24:(i+1)*24,0])
    X.append(np.hstack((np.hstack(dataset[(i-1)*24:i*24,0]),
                      np.hstack(dataset[(i-1)*24:(i+1)*24,1:]))))

Y = np.array(Y)
X = np.array(X)
print "X:"
print X,X.shape
print "Y:"
print Y,Y.shape

ls = lm.Lasso(alpha = 20, fit_intercept = True, warm_start = True, max_iter = 2000)
ls.fit(X,Y)

print "R^2 is:"
print ls.score(X,Y)
print "ls.coef_:"
print ls.coef_.shape,'\n',ls.coef_

Y_pre = ls.predict(X)
y1 = ls.predict(X[-2])
X[-1,range(0,24)] = y1
y2 = ls.predict(X[-1])

plt.figure()
plt.imshow(ls.coef_, interpolation='none', cmap = cm.coolwarm,
           origin = 'lower', aspect = 'auto',
           vmax = abs(ls.coef_).max(), vmin = -abs(ls.coef_).max())
cbar = plt.colorbar()
##cbar.set_label('Coeffecient value')
plt.xlabel('Feature')
plt.ylabel('Time')
plt.title('Coeffecients Matrix')
plt.savefig('./plots/multiceoff.eps')
plt.savefig('./plots/multiceoff.pdf')

plt.figure(figsize=(10,8))
plt.plot(np.hstack(Y[-2:]),'o-')
plt.plot(np.hstack((y1,y2)),'r*-')
plt.xlabel('#hour of prediction')
plt.ylabel('pm25(ug/m^3)')
plt.legend(('48 hour ground truth', '48 hour prediction'), loc = 'best')
plt.title('48 hours prediction via multivariate regression')
plt.savefig('./plots/multipre.eps')
plt.savefig('./plots/multipre.pdf')

### TIME SERIES ARMAX
##import pickle
##from statsmodels.tsa.stattools import *
##from statsmodels.tsa.arima_model import *
##print "-"*99
##print """
##Prediction by Time Series ...
##Model: ARMA(p,q)
##"""
##y = copy.copy(dataset[:,0])
##logy = np.array([log(item) for item in y])
##X = copy.copy(dataset[:,1:7])
##
####acf_y = acf(y, nlags = 96, fft = True)
####pacf_y = pacf(y, nlags = 96)
######ccf_yt = ccf(y,X[:,1], unbiased = True)
######peri = periodogram(X)
####plt.figure()
####plt.plot(acf_y)
####plt.plot(pacf_y)
##
##hournum = len(y)
##p = 48
##q = 2
##
##arma = ARMA(endog = logy[:-48], order = (p,q), exog = X[:-48])
##result = arma.fit(trend = 'nc', method = 'css', full_output = True, disp = 5)
##with open('arma_res.pkl','wb') as output:
##    pickle.dump(result,output,pickle.HIGHEST_PROTOCOL)
##result = pickle.load(open('arma_res.pkl','rb'))
##print result.summary()
####pre = result.predict(start = p, end = hournum-1, exog = X, dynamic = False)
##
##pre = result.forecast(steps = 48, exog = X[-48:], alpha = 0.05)
##print pre
##
##plt.figure()
##plt.plot(y[-48:])
##plt.plot(np.array([exp(item) for item in pre[0]]))
##plt.show()
##ARMAX model is similar to naive linear model which have y(t)~X(t)*b 


# Time Series Filter
print "-"*99
print """
Time Series Filtering
"""





### Lasso regression of log(PM2.5)
##pm25 = dataset[range(0,len(dataset)),0]     # pm25 don't share memory with dataset
##logpm25 = np.array([log(x) for x in pm25])
##feas = dataset[:,1:]
##hournum,feanum = feas.shape
##delay = 24
##testnum = 2000
##feature = np.zeros((hournum-delay,delay+feanum*(delay+1)))
##for i in range(0,hournum-delay):
##    a = logpm25[i:i+delay]
##    b = np.hstack(feas[i:i+delay+1,:])
##    feature[i] = np.hstack((a,b))
##response = logpm25[range(delay,len(pm25))]
##print "-"*99
##print "Lasso regression of log(Pm2.5)..."
##print "Features: combined feature vectors in a time window with lenth of delay %d"%delay
##print feature
##print "Response: log of pm25 concentration per hour"
##print response
##fea1 = feature[:-testnum]
##res1 = response[:-testnum]
##ls = lm.LassoCV(alphas=None, fit_intercept = True, verbose=True)
##ls.fit(fea1,res1)
##print "chosen alpha is:"
##print ls.alpha_
##print "intercept is:"
##print ls.intercept_
##print "R^2 score is:"
##print ls.score(fea1,res1)
##plt.figure()
##plt.plot(range(0,delay),ls.coef_[:delay],'k-')
##plt.plot(range(delay,delay+feanum*(delay+1)),ls.coef_[delay:],'r-')
##plt.xlabel('feature')
##plt.ylabel('coeffecient')
##plt.legend(('previous pm2.5 concentration', 'weather features in the window'))
##plt.suptitle('Coefficients of log Lasso')
##plt.savefig('./plots/coeflassolog.eps')
##plt.savefig('./plots/coeflassolog.pdf')
##prelog = ls.predict(feature)
##pre = np.array([exp(x) for x in prelog])
##plt.figure()
##t = range(0,hournum-delay)
##plt.plot(t,np.array([exp(x) for x in response]))
##plt.plot(t[:-testnum],pre[:-testnum],'g-')
##plt.plot(t[-testnum:],pre[-testnum:],'y-')
##plt.xlabel('#hour')
##plt.ylabel('pm25(ug/m^3)')
##plt.legend((('original data','log lasso regression','one hour prediction')))
##plt.savefig('./plots/lassolog.eps')
##plt.savefig('./plots/lassolog.pdf')
##plt.figure(figsize=(10,8))
##logpre = np.zeros(48)
##logpre[0] = ls.predict(feature[-48])
##logpm25[-48] = logpre[0]
##for i in range(1,48):
##    a = logpm25[-delay-48+i:-48+i]
##    if i<47:
##        b = np.hstack(feas[-delay-48+i:-48+i+1,:])
##    else:
##        b = np.hstack(feas[-delay-48+i:,:])
##    feature[-48+i] = np.hstack((a,b))
##    logpre[i] = ls.predict(feature[-48+i])
##    logpm25[-48+i] = logpre[i]# use predicted pm25 to predict next hour's pm25
##plt.plot(np.array([exp(x) for x in response[-48:]]),'o-')
##plt.plot(np.array([exp(x) for x in logpre]),'r*-')
##plt.ylim((0,160))
##plt.xlabel('#hour of prediction')
##plt.ylabel('pm25(ug/m^3)')
##plt.legend(('48 hour ground truth', '48 hour prediction'),loc = 'best')
##plt.title('48 hour prediction via Lasso regression of log(PM2.5)')
##plt.savefig('./plots/lassoprelog.eps')
##plt.savefig('./plots/lassoprelog.pdf')


### Kalman filtering or continuous HMM
##print "-"*99
##print """
##Prediction by Kalman filtering or Continuous HMM ...
##Hidden state:   pm25 concentration
##Observed data:  meteorological observations
##Given:          full data (hidden and observed data)
##Learn:          state transition pdf & ommision pdf
##"""
### 
##daynum = dataset.shape[0]/24
##U = []
##Y = []
##for i in range(0,daynum):
##    Y.append(dataset[i*24:(i+1)*24,0])
##    U.append(dataset[i*24:(i+1)*24,1:])
##A = np.eye(24)


### Artificial Neural Network (ANN)
##from pybrain.structure import *
##from pybrain.structure.modules import *
##from pybrain.datasets import SupervisedDataSet
##from pybrain.supervised.trainers import BackpropTrainer
##from pybrain.tools.neuralnets import NNregression, Trainer
##from pybrain.tools.shortcuts import buildNetwork
##from pybrain.utilities import percentError
##print "-"*99
##print """
##Prediction via Aritificial Neural Network (ANN) ...
##
##"""
##print dataset
##pm25 = dataset[range(0,len(dataset)),0]     # pm25 don't share memory with dataset
##feas = dataset[:,1:]
##hournum,feanum = feas.shape
##delay = 1
##testnum = 2000
##feature = np.zeros((hournum-delay,delay+feanum*(delay+1)))
##for i in range(0,hournum-delay-7000):
##    a = pm25[i:i+delay]
##    b = np.hstack(feas[i:i+delay+1,:])
##    feature[i] = np.hstack((a,b))
##response = pm25[range(delay,len(pm25))]
### build dataset
##DS = SupervisedDataSet(delay+feanum*(delay+1),1)
##for i in range(0,hournum-delay):
##    DS.addSample(feature[i],response[i])
##tstdata, trndata = DS.splitWithProportion(0.25)
##print "Number of training patterns: ", len(trndata)
##print "Input and output dimensions: ", trndata.indim, trndata.outdim
##print DS['target']
##print DS['input']
##### build network
####n = buildNetwork(trndata.indim, 10, trndata.outdim, bias=True, hiddenclass=TanhLayer, outclass = LinearLayer)
##### train network
####trainer = BackpropTrainer(n, dataset=tstdata, verbose=True, batchlearning=False, weightdecay=0.001)
######trndata.randomBatches( 'input', n = 50)
####trainer.trainEpochs(epochs=1)
######trainresult = percentError(trainer.testOnClassData(dataset = trndata), trndata['target'])
######predictresult = percentError(trainer.testOnClassData(dataset = tstdata), tstdata['target'])
######print trainresult, predictresult
##
##n = NNregression(trndata)
##n.setupNN(hidden = 10)
##n.runTraining(convergence = 1)
##n.saveTrainingCurve('tst.csv')
##n.saveNetwork('net')
##
### test network
##res_fit = n.activateOnDataset(DS)
##plt.figure()
##plt.plot(response)
##plt.plot(res_fit[:-testnum])
##plt.plot(res_fit[-testnum:])
##plt.show()
##

