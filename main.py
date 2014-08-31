#-*- coding:utf-8 -*-

__author__ = 'xumin'

from flask import Flask, render_template
import sqlite3
import utils
import pickle
import time

path = "/home/xumin/Work/website"

def average(inlist):
    total = 0
    num = 0
    for x in inlist:
        total = total + x
        num += 1
    return int(total/num)

app = Flask(__name__)
app.debug = True

@app.route('/')
def welcome():
    with open('%s/data/known.pickle' % path) as f:
        known = pickle.load(f)
    with open("%s/data/predict.pickle" % path) as f:
        predict = pickle.load(f)
    with open("%s/data/forecast.pickle" % path) as f:
        forecasts = pickle.load(f) 

    forinfo = forecasts['info']
    preyear = forecasts['year']
    premonth = forecasts['month']
    preday = forecasts['day']
    prehour = forecasts['hour']
    preminute = forecasts['minute']

    values0 = [int(x) for x in known + predict[:24-len(known)]]
    values1 = [int(x) for x in predict[24-len(known):48-len(known)]]
    values2 = [int(x) for x in predict[48-len(known):72-len(known)]]
    day0 = [average(values0),20,80,1000,2,u"东南风",0,u"晴"]
    day1 = [average(values1),20,80,1000,2,u"东南风",0,u"晴"]
    day2 = [average(values2),20,80,1000,2,u"东南风",0,u"晴"]

    # date info
    year = []
    month = []
    day = []
    for offset in [0,24*60*60,24*60*60*2]:
        year.append(time.localtime(time.time()+offset).tm_year)
        month.append(time.localtime(time.time()+offset).tm_mon)
        day.append(time.localtime(time.time()+offset).tm_mday)

    # polt label
    labels = range(0,24)

    return render_template('index.html', day0=day0, day1=day1, day2=day2, 
        values0=values0, values1=values1, values2=values2,
        year=year, month=month, day=day,
        preyear=preyear, premonth=premonth, preday=preday,
        prehour=prehour, preminute=preminute,
        forinfo=forinfo, split=len(known), labels=labels)

@app.route('/about')
def about():
    with open(('%s/data/known.pickle' % path)) as f:
        known = pickle.load(f)
    with open(("%s/data/predict.pickle" % path)) as f:
        predict = pickle.load(f)
    with open("%s/data/forecast.pickle" % path) as f:
        forecasts = pickle.load(f) 

    preyear = forecasts['year']
    premonth = forecasts['month']
    preday = forecasts['day']
    prehour = forecasts['hour']
    preminute = forecasts['minute']

    values0 = [int(x) for x in known + predict[:24-len(known)]]
    values1 = [int(x) for x in predict[24-len(known):48-len(known)]]
    values2 = [int(x) for x in predict[48-len(known):72-len(known)]]
    day0 = [average(values0),20,80,1000,2,u"东南风",0,u"晴"]
    day1 = [average(values1),20,80,1000,2,u"东南风",0,u"晴"]
    day2 = [average(values2),20,80,1000,2,u"东南风",0,u"晴"]
    return render_template('about.html', day0=day0, day1=day1, day2=day2, 
        values0=values0, values1=values1, values2=values2,
        preyear=preyear, premonth=premonth, preday=preday,
        prehour=prehour, preminute=preminute)

if __name__ == '__main__':
    app.run(host='0.0.0.0')