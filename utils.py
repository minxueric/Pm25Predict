# -*- coding: utf-8 -*-
import numpy as np
from pyquery import PyQuery as pq
import urllib2
import json
import time
import string
import re
import sqlite3
import pickle
from math import log,exp
import sklearn.linear_model as lm

# def path for crontab task
path = '/home/xumin/Work/website'

# current time
year = time.localtime(time.time()).tm_year
month = time.localtime(time.time()).tm_mon
day = time.localtime(time.time()).tm_mday
hour = time.localtime(time.time()).tm_hour
minute = time.localtime(time.time()).tm_min

# def parameters for loglasso model
p = 24
q = 24

days = [31,28,31,30,31,30,31,31,30,31,30,31]

location = 'beijing'

np.set_printoptions(precision = 5, suppress = True)

class FoundException(Exception):
	pass

def seqnum(year,month,day,hour):
    num = 0
    for y in range(2013,year+1):
        for m in range(1,13):
            for d in range(1,days[m-1]+1):
                if (y==2013 and m==1 and d<26):
                    continue
                num = num + 24
                if (y==year and m==month and d==day):
                    num+=hour
                    return num

def invseqnum(num):
    for y in range(2014,10000):
        for m in range(1,13):
            for d in range(1,days[m-1]+1):
                for h in range(0,24):
                    if seqnum(y,m,d,h)==num:
                        return (y,m,d,h)

def timestamp(year,month,day,hour):
    timestamp = ('%04d-%02d-%02d-%02d' % (year,month,day,hour))
    return timestamp

def timeformatter(time):
    pos1 = time.find(":")
    pos2 = time.find(" ")
    minute = int(time[pos1+1:pos2])
    hour = int(time[:pos1])%12
    amorpm = time[pos2+1:]
    if amorpm == "PM":
        hour_24 = hour + 12
    else:
        hour_24 = hour
    return hour_24, minute

def tempformatter(temp):
    pos = temp.find(" ")
    try:
        temperature = string.atof(temp[:pos])
    except:
        temperature = None
    return temperature

def speedformatter(speed):
    if speed == "Calm":
        return 0
    else:
        pos = speed.find(" ")
        return string.atof(speed[0:pos])

def rainformatter(event):
    if "ain" in event:
        return 1
    else:
        return 0

# ******************* below are crawlers ********************
def weather(year,month,day):
    # crawl weather observation in beijing from source:
    # http://www.wunderground.com/history/airport/ZBAA/%04d/%02d/%02d/DailyHistory.html?req_city=Beijing&req_state=BJ&req_statename=Beijing
    url = 'http://www.wunderground.com/history/airport/ZBAA/%04d/%02d/%02d/DailyHistory.html?req_city=Beijing&req_state=BJ&req_statename=Beijing' % (year,month,day)
    f = urllib2.urlopen(url)
    doc = f.read()
    f.close()
    query = pq(doc)
    table = query("#observations_details")
    thead = table("thead")("th")
    trows = table("tbody")("tr")
    #check out the heads
    idxs = [0,0,0,0,0,0,0,0,0]
    idx = 0
    for th in thead.items():
        if "Time" in th.text():
            idxs[0] = idx
        if "Temp" in th.text():
            idxs[1] = idx
        if "Humidity" in th.text():
            idxs[2] = idx
        if "Pressure" in th.text():
            idxs[3] = idx
        if "Wind Speed" in th.text():
            idxs[4] = idx
        if "Wind Dir" in th.text():
            idxs[5] = idx
        if "Dew" in th.text():
            idxs[6] = idx
        if "Conditions" in th.text():
            idxs[7] = idx
        if "Events" in th.text():
            idxs[8] = idx
        idx = idx + 1

    # record each hour's observation in a list object
    weather = []
    for row in trows.items():
        tds = row("td")
        hour, minute = timeformatter(tds.eq(idxs[0]).text())
        if minute == 30:
            continue
        temperature = tempformatter(tds.eq(idxs[1]).text())
        humidity = int(tds.eq(idxs[2]).text()[:-1])
        pressure = tempformatter(tds.eq(idxs[3]).text())
        speed = speedformatter(tds.eq(idxs[4]).text())
        direction = tds.eq(idxs[5]).text()
        dew = tempformatter(tds.eq(idxs[6]).text())
        condition = tds.eq(idxs[7]).text()
        rain = rainformatter(tds.eq(idxs[8]).text())
        ob = dict(year=year, month=month, day=day, hour=hour, temperature=temperature, humidity=humidity,
            pressure=pressure, speed=speed, direction=direction, dew=dew, condition=condition, rain=rain)
        weather.append(ob)
        # delete redundant last observation
        if (len(weather)>=2 and ob['hour']==weather[-2]['hour']):
            del weather[-2]

    # check if there are 24 hours
    if (len(weather)<24):
        has = []
        for i in range(0,len(weather)):
            wi = weather[i]
            has.append(wi['hour'])
        hasnot = list(set(range(0,24))-set(has))
        for i in hasnot:
            nul = dict(year=year, month=month, day=day, hour=i, temperature=None, humidity=None,
                pressure=None, speed=None, direction=None, dew=None, condition=None, rain=None)
            weather.insert(i,nul)

    return weather

def forecast():
    # get forecast weather from source: 
    # http://www.wunderground.com/cgi-bin/findweather/getForecast?query=40.06999969,116.58999634&sp=ZBAA#forecast-table
    url = "http://www.wunderground.com/cgi-bin/findweather/getForecast?query=40.06999969,116.58999634&sp=ZBAA"
    f = urllib2.urlopen(url)
    doc = f.read()
    f.close()
    # find the API json from the doc
    json_str = re.findall(r"wui.bootstrapped.API =\n(.*)\n;\n</script>",doc,re.S)
    parsed_json = json.loads(json_str[0])
    # format for print
    format_json = json.dumps(parsed_json,indent=2,ensure_ascii=False)

    # record each hour's observation in a list object
    forecasts = []
    dayforecasts = []
    days = parsed_json['forecast']['days']
    for day_ in days:
        hours = day_['hours']
        onedayforecasts = []
        for hourly in hours:
            now = hourly['date']['iso8601']
            year = int(now[0:4])
            month = int(now[5:7])
            day = int(now[8:10])
            hour = int(now[11:13])
            temperature = hourly['temperature']
            humidity = hourly['humidity']
            pressure = hourly['pressure']
            speed = hourly['wind_speed']
            direction = hourly['wind_dir']
            dew = hourly['dewpoint']
            condition = hourly['condition']
            rain = rainformatter(hourly['condition'])
            hourly = dict(year=year, month=month, day=day, hour=hour, temperature=temperature, humidity=humidity, 
                pressure=pressure, speed=speed, direction=direction, dew=dew, condition=condition, rain=rain)
            forecasts.append(hourly)
            onedayforecasts.append(hourly)
        # summary information i.e., day average 
        summary = day_['summary']
        high = summary['high']
        low = summary['low']
        humidity_l = summary['humidity_max']
        humidity_h = summary['humidity_min']
        wind_avg_dir = summary['wind_avg_dir']
        wind_avg_speed = summary['wind_avg_speed']
        dew = 0
        pressure = 0
        event = u"晴"
        for h in onedayforecasts:
            dew += h['dew']
            pressure += h['pressure']
            if h['rain'] == 1:
                event = u"雨"
        dew = dew/len(onedayforecasts)
        pressure = pressure/len(onedayforecasts)
        dayforecast = dict(high=high, low=low, humidity_l=humidity_l, humidity_h=humidity_h,
            wind_avg_dir=wind_avg_dir, wind_avg_speed=wind_avg_speed, 
            dew=dew, pressure=pressure, event=event)
        dayforecasts.append(dayforecast)

    # record day summary forecasts and corresponding time
    year = time.localtime(time.time()).tm_year
    month = time.localtime(time.time()).tm_mon
    day = time.localtime(time.time()).tm_mday
    hour = time.localtime(time.time()).tm_hour
    minute = time.localtime(time.time()).tm_min 
    with open(("%s/data/forecast.pickle" % path), 'w') as f:
        pickle.dump(dict(info=dayforecasts[:3],year=year,
            month=month,day=day,hour=hour,minute=minute),f)
    return forecasts

def airnow():
    # crawl current air quality in beijing from web source: pm25.in
    # return PM2.5 and timestamp
    url = 'http://www.pm25.in/api/querys/aqi_details.json?city=beijing&token=nU4hUZ6UwzUbtsCSWzPn'
    f = urllib2.urlopen(url)
    json_string = f.read()
    f.close()
    parsed_json = json.loads(json_string)
    # select average observation to dump
    info = parsed_json[-1]
    air = json.dumps(info,ensure_ascii=False,indent=2)
    # read pm2.5 value and timestamp
    ob = json.loads(air)
    PM25 = ob['pm2_5']
    timeArray = time.strptime(ob['time_point'],"%Y-%m-%dT%H:%M:%SZ")
    hour = timeArray.tm_hour
    timestamp = ("%04d-%02d-%02d-%02d" % (year, month, day, hour))
    return PM25, timestamp, hour

def pmbj(year,month,day,hour):
    # get historical beijing pm25 from source: young-0.com
    url = ("http://www.young-0.com/airquality/index.php?number=1&unit=0&enddate=1&year=%d&month=%d&day=%d&hour=%d&city=0&cn=1&action=2" % (year,month,day,hour))
    req = urllib2.Request(url, headers={'User-Agent':'Magic Browser'})
    f = urllib2.urlopen(req)
    doc = f.read()
    f.close()
    query = pq(doc)
    table = query("table")
    trs = table("tr")
    tr = trs.eq(hour+1)
    tds = tr("td")
    return tds.eq(3).text()

# ******************* database constructors *****************
def build():
	# build the database from 2013-01-25-00 to 2014-04-25-14
    conn = sqlite3.connect('%s/data/update.db' % path)
    c = conn.cursor()

    conn2 = sqlite3.connect('%s/data/airweather.db' % path)
    c2 = conn2.cursor()
    c2.execute("select * from PM25")

    c.execute("""
            create table if not exists airweather(
                num integer primary key,
                timestamp text,
                location text,
                pm real,
                temperature real,
                humidity real,
                pressure real,
                speed real,
                direction text,
                dew real,
                condition text,
                rain integer
            )
            """)

    for x in range(0,10935):
        row = c2.fetchone()
        row = list(row)
        row[0] = x
        row[1] = row[1][:-3]
        for i in range(3,len(row)):
            if row[i] == '':
                row[i] = None
        c.execute("""
            insert into airweather 
            (num, timestamp, location, pm, temperature, humidity, pressure, speed,
                direction, dew, condition, rain) values (?,?,?,?,?,?,?,?,?,?,?,?)""",
        row)
        # drop the half hour observation
        c2.fetchone()

    # commit the change
    conn.commit()
    conn.close()

def complete():
    # complete the database from 2014-04-25-15 to 2014-07-02-23
    # new complete from 2014-07 to 2014-08 

    try:
        for month in [8]:
            for day in range(1,days[month-1]+1):
                if (month==8 and day<26):
                    continue
                print 2014, month, day
                w = weather(2014,month,day)
                # # beginning day not all hours inserted
                # if (month==4 and day==25):
                #     for hour in range(15,24):
                #         whour = w[hour]
                #         c.execute("""
                #             insert or replace into airweather
                #             (num, timestamp, location, pm, temperature, humidity, pressure, speed,
                #                 direction, dew, condition, rain) values (?,?,?,?,?,?,?,?,?,?,?,?) """,
                #         (seqnum(2014,4,25,hour), timestamp(2014,4,25,hour), location, pmbj(2014,4,25,hour), 
                #         whour['temperature'], whour['humidity'], whour['pressure'], whour['speed'], 
                #         whour['direction'],whour['dew'], whour['condition'],whour['rain']))
                # len(w) hours inserted per day
                for hour in range(0,len(w)):
                    whour = w[hour]
                    conn = sqlite3.connect('%s/data/update.db' % path)
                    c = conn.cursor()
                    print whour
                    c.execute("""
                        insert or replace into airweather
                        (num, timestamp, location, pm, temperature, humidity, pressure, speed,
                            direction, dew, condition, rain) values (?,?,?,?,?,?,?,?,?,?,?,?) """,
                    (seqnum(2014,month,day,hour), timestamp(2014,month,day,hour), location, pmbj(2014,month,day,hour), 
                    whour['temperature'], whour['humidity'], whour['pressure'], whour['speed'], 
                    whour['direction'],whour['dew'], whour['condition'],whour['rain']))
                    # commit the change
                    conn.commit()
                    conn.close()
                if (month==8 and day==30):
                    raise FoundException()
    except FoundException:
        print "database update.db completed now~"


def update():
    # update the update.db dynamically
    conn = sqlite3.connect('%s/data/update.db' % path)
    c = conn.cursor()
    # get weather information from crawlers
    w = weather(year,month,day)
    # insert weather and pm25 into table airweather
    for wi in w:
        c.execute("""
            insert or replace into airweather 
            (num, timestamp, location, pm, temperature, humidity, pressure, speed,
                direction, dew, condition, rain) values (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (seqnum(year,month,day,wi['hour']), timestamp(year,month,day,wi['hour']), 
                location, pmbj(year,month,day,wi['hour']), wi['temperature'], 
                wi['humidity'], wi['pressure'],wi['speed'],
                wi['direction'], wi['dew'], wi['condition'], wi['rain'])
            )
    print "updated at %04d-%02d-%02d-%02d-%02d" % (year,month,day,hour,minute)
    conn.commit()
    conn.close()

# ********************* model training **********************
def retrieve(year,month,day,hour):
    # retrive training raw data from the database
    conn = sqlite3.connect('%s/data/update.db' % path)
    c = conn.cursor()
    c.execute("""
        select pm, temperature, humidity, pressure, 
        speed, direction, dew, rain from airweather
        """)
    res = c.fetchall()
    numtotal = seqnum(year,month,day,hour)
    raw = []
    num = 0
    for row in res:
        raw.append(list(row))
        if num == numtotal:
            break
        num+=1
    conn.close()
    return raw
    
def raw2dataset(raw):
    # fill missed meterological observations
    i = 0
    for row in raw:
        if (row[2]==None):
            raw[i] = raw[i-1]
        i += 1
    # interpolation for pm25
    dataset = np.array(raw)     # use array for easier operation on dataset
    i = 0
    for x in dataset[:,0]:      # None or ''(string) converted into np.nan
        if (x=='' or x==None):  
            dataset[i,0] = np.nan
        i += 1
    pm25 = np.double(dataset[:,0])
    ok = -np.isnan(pm25)
    xp = ok.nonzero()[0]
    fp = pm25[ok]
    x = np.isnan(pm25).nonzero()[0]
    pm25[np.isnan(pm25)] = np.interp(x,xp,fp)
    dataset = np.hstack((pm25[:,np.newaxis],dataset[:,1:]))
    # discretize direction into 0-1 variables
    direction = dataset[:,5]
    direction[direction=='None'] = 'Variable'
    direction[np.array(direction=='ENE') + np.array(direction=='NNE')] = 'NE'
    direction[np.array(direction=='ESE') + np.array(direction=='SSE')] = 'SE'
    direction[np.array(direction=='NNW') + np.array(direction=='WNW')] = 'NW'
    direction[np.array(direction=='SSW') + np.array(direction=='WSW')] = 'SW'
    N = np.array(direction=='North', dtype = np.double)
    S = np.array(direction=='South', dtype = np.double)
    E = np.array(direction=='East', dtype = np.double)
    W = np.array(direction=='West', dtype = np.double)
    NE = np.array(direction=='NE', dtype = np.double)
    SE = np.array(direction=='SE', dtype = np.double)
    NW = np.array(direction=='NW', dtype = np.double)
    SW = np.array(direction=='SW', dtype = np.double)
    V = np.array(direction=='Variable', dtype = np.double)
    dataset = np.hstack((dataset[:,(0,1,2,3,4,6,7)],N[:,np.newaxis],S[:,np.newaxis],E[:,np.newaxis],W[:,np.newaxis],
                    NE[:,np.newaxis],SE[:,np.newaxis],NW[:,np.newaxis],SW[:,np.newaxis],V[:,np.newaxis]))
    dataset = np.double(dataset)
    return dataset

def extract(dataset):
    pm25 = dataset[range(0,len(dataset)),0]
    logpm25 = np.array([log(x) for x in pm25])
    feas = dataset[:,1:]
    hournum, feanum = feas.shape
    feature = np.zeros((hournum-p,p+feanum*(q+1)))
    for i in range(0,hournum-p):
        a = logpm25[i:i+p]                  # p terms for pm2.5
        b = np.hstack(feas[i+p-q:i+p+1,:])  # q+1 terms for weather
        feature[i] = np.hstack((a,b))
    response = logpm25[range(p,len(pm25))]
    return feature,response

def train():
    # retrieve training raw data until yesterday 23:00
    raw = retrieve(year,month,day-1,23)
    dataset = raw2dataset(raw)
    
    # convert dataset into loglasso feature vs. response
    feature,response = extract(dataset)

    # training loglasso model
    ls = lm.LassoCV(alphas=None, fit_intercept=True, verbose=True)
    ls.fit(feature,response)
    # save loglasso model in pickle file
    with open(('%s/model/loglasso.pickle' % path), 'w') as f:
        pickle.dump(ls,f)

def predict():
    # predict future pm25 per hour
    # load model
    with open('%s/model/loglasso.pickle' % path) as f:
        ls = pickle.load(f)
    
    # load dataset from database
    raw = retrieve(year,month,day,hour)
    raw = raw[-p:]  # reserve last p hours of pm2.5 and weather
    # download forecast information
    forecasts = forecast()
    # avoid website crashing
    if forecasts == []:
        print "no forecasts"
        return
    # concatenate history and forecast
    for x in forecasts[:24*3-hour]:
        raw.append([None,x['temperature'],x['humidity'],x['pressure'],
            x['speed'],x['direction'],x['dew'],x['rain']])
    
    # convert raw into dataset
    dataset = raw2dataset(raw)

    # extract feature from dataset
    feature, response = extract(dataset)

    # recursive prediction
    predict = response
    for i in range(0,24*3-hour):
        response[i] = ls.predict(feature[i])
        predict[i] = exp(response[i])
        dataset[p+i,0] = predict[i]
        feature, response = extract(dataset)

    # save prediction result for website
    known = dataset[p-hour:p,0]
    with open(('%s/data/known.pickle' % path),'w') as f:
        pickle.dump(list(known),f)
    with open(('%s/data/predict.pickle' % path),'w') as f:
        pickle.dump(list(predict),f)
    print "forecasted at %04d-%02d-%02d-%02d-%02d" % (year,month,day,hour,minute)
    return predict 

if __name__ == "__main__":
    complete()