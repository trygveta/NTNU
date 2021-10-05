import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statistics import mean


def renamecolumns(year):
    data = pd.read_csv("data.UF.A-F2."+str(year)+".csv", index_col=0, parse_dates=True)
    data = data.rename(columns={'pi:37628|average':'A-Inlet flow','pi:37634|average':'B-Inlet flow',
                                'pi:37640|average':'C-Inlet flow','pi:37646|average':'D-Inlet flow',
                                'pi:37652|average':'E-Inlet flow','pi:37658|average':'F-Inlet flow',
                                'pi:38942|average':'A-Inlet header pressure','pi:38948|average':'B-Inlet header pressure',
                                'pi:38954|average':'C-Inlet header pressure','pi:38960|average':'D-Inlet header pressure',
                                'pi:38966|average':'E-Inlet header pressure','pi:38972|average':'F-Inlet header pressure',
                                'pi:37946|average':'A-Inlet vessel level','pi:37952|average':'B-Inlet vessel level',
                                'pi:37958|average':'C-Inlet vessel level','pi:37964|average':'D-Inlet vessel level',
                                'pi:37970|average':'E-Inlet vessel level','pi:37976|average':'F-Inlet vessel level',
                                'pi:38344|average':'A-TMP','pi:38350|average':'B-TMP','pi:38356|average':'C-TMP',
                                'pi:38362|average':'D-TMP','pi:38368|average':'E-TMP','pi:38374|average':'F-TMP',
                                'pi:37424|average':'A-R factor','pi:37436|average':'B-R factor',
                                'pi:37448|average':'C-R factor','pi:37460|average':'D-R factor',
                                'pi:37472|average':'E-R factor','pi:37484|average':'F-R factor',
                                'pi:39667|average':'Sea temp','pi:38840|average':'Common Inlet pressure'})
    data.to_csv('data.UF.A-F2.'+str(year)+'_new_col.csv')


def combine_years():
    data18 = pd.read_csv("data.UF.A-F2.2018_new_col.csv", index_col=0, parse_dates=True)
    data19 = pd.read_csv("data.UF.A-F2.2019_new_col.csv", index_col=0, parse_dates=True)
    data20 = pd.read_csv("data.UF.A-F2.2020_new_col.csv", index_col=0, parse_dates=True)

    frames = [data18, data19, data20]
    result = pd.concat(frames)
    result.to_csv('data18-20.csv')


def getvesseldata(vessel):
    data = pd.read_csv('data18-20.csv', index_col=0, parse_dates=True)
    vesseldata = data[[vessel+'-Inlet flow',vessel+'-Inlet header pressure',vessel+'-Inlet vessel level',
                       vessel+'-TMP',vessel+'-R factor','Sea temp','Common Inlet pressure']]
    return vesseldata


def createvesseldata():
    vessels = ['A', 'B']
    for letter in vessels:
        data = getvesseldata(letter)
        data.dropna(subset=[letter+'-R factor'], inplace=True)
        data.to_csv('data_'+letter)
        print('Ferdig med '+letter)


def countNan(vessel):
    print('Begynner med '+vessel)
    vesseldata = pd.read_csv('data_'+vessel)
    vesseldata['isna'] = vesseldata[vessel+'-R factor'].isna()
    vesseldata['NaN_count'] = ""
    vesseldata.loc[0, 'NaN_count'] = 1
    mod = round(len(vesseldata))
    for i in range(1,len(vesseldata)):
        if vesseldata.loc[i, 'isna'] and vesseldata.loc[i-1, 'isna']:
            vesseldata.loc[i,'NaN_count'] = vesseldata.loc[i-1,'NaN_count'] + 1
        elif vesseldata.loc[i, 'isna']:
            vesseldata.loc[i,'NaN_count'] = 1
        if i % mod == 0:
            print(round((i/mod)*10),'%')
    vesseldata.to_csv('data_'+vessel)
    print('Ferdig med '+vessel)


def create_intervals_nancount(vessel):
    vesseldata = pd.read_csv('data_'+vessel,index_col=0,parse_dates=True)
    diff = vesseldata[vessel+'-R factor'].diff()
    intervals = []
    limit = 50
    mod = round(len(vesseldata)/10)
    start = 0
    stop = 0
    i = 0
    while i < len(vesseldata):
        if vesseldata.loc[i, 'NaN_count'] == limit:
            if start == 0:
                while i < len(vesseldata) and vesseldata.loc[i, 'isna']:
                    i = i + 1
                start = i
            else:
                stop = i - (limit - 1)
                intervals.append(vesseldata[vessel+'-R factor'][start:stop].interpolate(
                    method='linear').to_numpy())
                start = 0
                stop = 0
                i = i - 1
        if i % mod == 0:
            print(round(i / mod) * 10, '%')
        i = i + 1
    print('Antall intervaller: ', len(intervals))
    for i in range(len(intervals)-1,0,-1):
        if len(intervals[i]) < 50:
            del intervals[i]
    print('Antall intervaller etter rydding: ', len(intervals))
    f = open('intervals_'+vessel,'w')
    for interval in intervals:
        interval.tofile(f,sep=",")
        f.write('\n')
    f.close()


def create_intervals_diff(vessel):
    vesseldata = pd.read_csv('data_' + vessel)
    intervals = []
    limit = 1.5
    step = 10
    mod = round(len(vesseldata)/10)
    start = 1
    stop = 0
    i = 0
    while i < len(vesseldata)-step:
        if vesseldata.loc[i, vessel+'-R factor'] - vesseldata.loc[i+step, vessel+'-R factor'] > limit:
            if start == 0:
                start = i+step
                i = i + step - 1
            else:
                stop = i
                intervals.append(vesseldata[vessel + '-R factor'][start:stop].to_numpy())
                start = stop + step
        if i % mod == 0:
            print(round(i/mod)*10, '%')
        i += 1
    print('Antall intervaller: ', len(intervals))
    f = open('intervals_' + vessel, 'w')
    for interval in intervals:
        interval.tofile(f, sep=",")
        f.write('\n')
    f.close()


def plot_intervals(vessel):
    intervals = []
    f = open('intervals_'+vessel, 'r')
    for line in f:
        try:
            intervals.append(list(map(float, line.replace(' ', '').split(','))))
        except:
            continue
    f.close()
    print('Antall intervaller: ', len(intervals))
    last_f = 0
    for interval in intervals:
        x, y = np.array(range(last_f, last_f + len(interval))).reshape(-1, 1), np.array(interval)
        plt.plot(x, y)
        last_f = last_f + len(y)
    plt.show()


def clean_intervals(vessel):
    intervals = []
    f = open('intervals_'+vessel, 'r')
    for line in f:
        try:
            intervals.append(list(map(float, line.replace(" ", "").split(','))))
        except:
            continue
    f.close()
    counter = 0
    for i in range(len(intervals)-1, 0, -1):
        if len(intervals[i]) > 5000:
            del intervals[i]
            counter += 1
        elif len(intervals[i]) < 500:
            del intervals[i]
            counter += 1
        elif mean(intervals[i]) < 0:
            del intervals[i]
            counter += 1
    print('Slettet ', counter, ' intervaller')
    f = open('intervals_' + vessel, 'w')
    for interval in intervals:
        np.array(interval).tofile(f, sep=',')
        f.write('\n')
    f.close()


def regression(vessel):
    intervals = []
    coef = []
    counter = 0
    f = open('intervals_'+vessel, 'r')
    for line in f:
        counter += 1
        try:
            intervals.append(list(map(float, line.replace(' ', '').split(','))))
        except:
            continue
    f.close()
    print(counter)
    print(len(intervals))
    for i in range(len(intervals)):
        y = np.array(intervals[i])
        x = np.array(range(0, len(intervals[i]))).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        coef.append(float(model.coef_))
        # if model.coef_*1000 < 20:
        #     coef.append(model.coef_*1000)
    cf_mean = mean(coef)
    for cf in coef:
        if cf > 5 * cf_mean:
            coef.remove(cf)
            print('!')
    #range(1, len(coef) + 1)
    return coef


def results():
    figure, axes = plt.subplots(nrows=3, ncols=2, sharey=True)

    axes[0, 0].plot(regression('A'), '.')
    axes[0, 0].set_title('Vessel A')
    axes[0, 1].plot(regression('B'), '.')
    axes[0, 1].set_title('Vessel B')
    axes[1, 0].plot(regression('C'), '.')
    axes[1, 0].set_title('Vessel C')
    axes[1, 1].plot(regression('D'), '.')
    axes[1, 1].set_title('Vessel D')
    axes[2, 0].plot(regression('E'), '.')
    axes[2, 0].set_title('Vessel E')
    axes[2, 1].plot(regression('F'), '.')
    axes[2, 1].set_title('Vessel F')

    figure.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)

    plt.xlabel("Number of intervals")
    plt.ylabel("Regression coefficient")

    figure.tight_layout()
    plt.show()


def compare():
    dataC = pd.read_csv('data_C', index_col=0, parse_dates=True)
    dataC = dataC[dataC['C-R factor'] < 12]
    dataF = pd.read_csv('data_F', index_col=0, parse_dates=True)

    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)

    dataC.plot(ax=axes[0], y = 'C-R factor', use_index=True)
    dataF.plot(ax=axes[1], y = 'F-R factor', use_index=True)

    plt.ylabel('R factor')
    plt.show()


def regressionOnIntervals():
    intervals = []
    f = open('intervals_A', 'r')
    for line in f:
        try:
            intervals.append(list(map(float, line.replace(' ', '').split(','))))
        except:
            continue
    f.close()
    intervals = intervals[3:7]

    fig, axes = plt.subplots(nrows=1, ncols=4, sharey=True)
    counter = 0
    for interval in intervals:
        y = np.array(interval)
        axes[counter].plot(y, '.')
        x = np.array(range(0, len(interval))).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        y_reg = model.predict(x)
        coef = model.coef_
        print(coef[0])
        axes[counter].plot(y_reg, color='red')
        text = 'Coefficient= '+str(round(coef[0],6)) + '\n Coefficient of determination: ' + str(round(r2_score(y, y_reg), 2))
        axes[counter].text(0.5, 0.95, text, horizontalalignment='center', transform=axes[counter].transAxes, color='red')
        counter += 1
    plt.show()




