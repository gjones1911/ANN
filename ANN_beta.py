from DataProcessor import *
import numpy as np
from ML_Visualizations import *
from performance_tests import *


class n_node:

    def __init__(self, x, w):
        self.x = x
        self.w = w

    def get_x(self):
        return self.x

    def get_w(self):
        return self.w


def sigmoid(w,x):
    s = 1/(1 + (np.exp(-1*np.dot(w.T, x), dtype=np.float64)))
    return s


def init_wij(k_size, d_size):
    ch = [-.01, .01]
    wij = list()
    for i in range(k_size):
        wi = list([])
        for j in range(d_size):
            wi.append(np.random.choice(ch, 1, replace=False)[0])
        wij.append(wi)

    return wij


def add_bias(data):
    d = data.tolist()
    ret_l = []
    for row in d:
        ret_l.append([1]+row)
    return np.array(ret_l, dtype=np.float64)


def get_yi(vh, zh):
    yt = list()
    sum = vh[1] * zh[1]

    vh = vh.T

    for t in range(len(zh)):
        for h in range(0, len(vh)):
            #sum += np.dot(vh[h], zh[t])
            #yt.append(sigmoid(vh[h], zh[t]))
            yt.append(1/(1+(-1*np.exp(np.dot(vh[h], zh[t])))))

    return yt


def get_zh(w, x):
    zh = list()
    for h in range(len(x)):
        zh.append(sigmoid(w[h], x[h]))
    return zh


def get_delvih(nu, r, y, z):
    delvih = []
    sum = np.array([0], dtype=np.float64)

    for h in range(len(z)):
            sum += (r-y[h])* z

    return delvih


def get_delwhj(nu, r, y, v, z, x):
    delwhj = []

    sum = 0

    return delwhj


def train_mlp(train_x, train_r, nu, hval, epoch=None, test_limit=5):

    # initalize whj, vih

    #whj = np.array(init_wij(len(train_x[0]), len(train_x[0])), dtype=np.float64)
    #vih = np.array(init_wij(len(train_x), len(train_x[0])), dtype=np.float64)
    whj = np.array(init_wij(hval, len(train_x[0])), dtype=np.float64)
    vih = np.array(init_wij(len(train_x), hval), dtype=np.float64)

    whj_list = list()
    vih_list = {}

    #for i in range(len(train_x)):
    for i in range(test_limit):
        #whj_list.append(np.array(init_wij(len(train_x[0]), len(train_x[0])), dtype=np.float64) )
        whj_list.append(whj)

    whj_list = np.array(whj_list, dtype=np.float64)

    # print('whj')
    # print(len(whj_list))
    # print(whj_list)
    # print('---------------------------------')
    # print('---------------------------------')
    # print('---------------------------------')

    #r_c = np.random.choice(len(train_x), len(train_x), replace=False)

    dif_wh, dif_vj = 100, 100

    good_wh = []
    # good_vj = []

    r_c = np.random.choice(len(train_x), len(train_x), replace=False)
    print('r_c')
    print(len(r_c))


    print(r_c[0:test_limit])
    r_c = r_c[0:test_limit]

    big_dif = 100

    z_h_l = {}
    yi_l = {}
    vih_l = {}


    e_cnt = 0
    # while dif_wh > 0 and dif_vj > 0:
    # while abs(dif_wh) > 0:
    while big_dif > 0:
        # while len(good_wh) <= 0 and len(good_vj) <= 0:
        orig_whj_list = np.array(whj_list.tolist(), dtype=np.float64)
        print()
        print()
        print('-----------------------------------------------------------------> epoch: ', e_cnt)

        for i in good_wh:
            print(r_c[i], "is done")
        print()
        print()
        # for t in range(len(train_x)):
        for t in range(len(r_c)):
            if t in good_wh:
                continue
            print()
            print()
            print('--------------------------------------------------------------> processing {:d}'.format(r_c[t]))
            print()
            xt = train_x[r_c[t]]
            rt = train_r[r_c[t]]
            #whj = whj_list[r_c[t]]
            whj = np.array(whj_list[t].tolist(), dtype=np.float64)

            zh = list([1])
            #for h in range(len(train_x[0])-1):
            # Make the hidden layer inputs (zh's)
            for h in range(hval-1):
                #zh.append(1/(1 + (np.exp(-1 * np.dot(whj[h], xt), dtype=np.float64))))
                zh.append(sigmoid(whj[h], xt))

            z_h_l[r_c[t]] = zh

            # calculate new predictions of class value (>0 == 1, <0 == 0)
            yi = list()
            yi.append(sigmoid(vih[r_c[t]], zh))
            print('sigmoid yio: ', yi)
            #if yi[0] > 0 or yi[0] == 0:
            if yi[0] > 0:
                yi[0] = 1
            else:
                yi[0] = 0
            yi = np.array(yi, dtype=np.float64)

            yi_l[r_c[t]] = yi[0]

            print('--------------------------------> predicted yi: ', yi[0])
            print('--------------------------------> actual y(rt): ', rt)

            # calculate the new delta vih
            # print('calculated del vih')
            del_vih = nu * (rt - yi[0])
            #print('nu * (rt - y): ', del_vih)
            del_vih = np.dot(del_vih, zh)
            # print('nu * (rt - yi) zh: ', del_vih)

            to_see = 1

            # calculate and add new adjustment(d_wh) to whj
            print()
            print('--------------------------------------------------------------------------------')
            print('--------------------------------------------------------------------------------')
            print('calculated del wh')
            d_wh = list()
            for h in range(hval):
                # val = np.dot((rt - yi[0]), vih[r_c[t], h])
                #val = np.dot((rt - yi[0]), vih[r_c[t],h])
                val = (rt - yi[0]) *  vih[r_c[t],h]
                # if h == to_see:
                    # print('(rt - yi)*vih', val)
                val = nu * val
                # if h == to_see:
                    # print('nu * ((rt-yi)*vih): ', val )
                #val = np.dot(val, zh[h])
                val = val * zh[h]
                # if h == to_see:
                    # print('nu * ((rt-yi)*vih)*zh: ', val, 'zh', zh[h])
                # diff = 1 - np.array(zh[h])
                diff = 1 - zh[h]
                #val = np.dot(val, diff)
                val = val * diff
                # if h == to_see:
                    # print('nu * ((rt-yi)*vih)*zh(1-zh): ', val)
                val = np.dot(val, xt)
                # if h == to_see:
                    # print('d_wh (val)')
                #print('nu * ((rt-yi)*vih)*zh(1-zh)*xt: \n', val)
                    #print(val)
                d_wh.append(val)

            # make adjustments to weights
            print('get the adjust by adding')
            vih[r_c[t]] = np.add(del_vih, vih[r_c[t]])
            #whj_list[r_c[t]] = (whj + d_wh)
            #whj_list[t] = (whj + d_wh)
            whj_list[t] = np.add(whj, d_wh)

            #dif_wh = np.around(np.mean(whj - whj_list[r_c[t]], dtype=np.float64), 3)
            #dif_wh = np.around(np.mean(whj - whj_list[t], dtype=np.float64), 7)
            dif_wh = np.mean(whj - whj_list[t], dtype=np.float64)

            #dif_vj = np.around(np.mean(vh - vih[r_c[t]], dtype=np.float64), 3)

            print('new wh diff')
            print(dif_wh)
            # print('new vj diff')
            # print(dif_vj)
            #if np.around(np.mean(dif_wh, dtype=np.float64), 9) == 0:
            if np.around(np.mean(dif_wh, dtype=np.float64), 9) == 0:
                print('diff wh for observation {:d} is done'.format(r_c[t]))
                good_wh.append(t)
            # if np.mean(dif_vj, dtype=np.float64) == 0:
              #   print('diff vj is done')
               #  good_vj.append(t)


            # dif_wh = np.around(np.mean(old_vh - whj_list, dtype=np.float64), 3)
            # dif_vj = np.around(np.mean(old_vh - vih, dtype=np.float64), 3)

            # print('dif_wh')
            # print(dif_wh)
            # print('dif_vi')
            # print(dif_vj)

            # dif_wh = 0
            # dif_vj = 0
            print()
            print()
            print()

        if epoch is not None:
            e_cnt += 1
            if e_cnt == epoch:
                print('breaking')
                break
        print()
        print()
        print()
        big_dif = abs(np.around(np.mean((orig_whj_list - whj_list), dtype=np.float64), 9))
        print('big dif', big_dif)

        '''
        print()
        print()
        print()
        print('--------------------new d_wh')
        print(len(d_wh))
        print(d_wh)
        print()
        print()
        print()
        print('whj + d_wh')
        print(whj + d_wh)
        print('------------------------------------------------')
        print('------------------------------------------------')
        del_wh = np.dot((rt-yi[0]), del_vih)
        print('(rt-yi) * delvih: ', del_wh)
        del_wh = nu * del_wh
        print('nu * ((rt-yi)*del_vih): ', del_wh)
        del_wh = np.dot(del_wh, zh)
        print('(nu * ((rt-yi) * delvih) ) * zh: ', del_wh)
        diff = 1 - np.array(zh)
        print('1 - zh')
        print(diff)
        del_wh = np.dot(del_wh, diff)
        print('(nu * (rt-yi) * delvih) * zh (1 - zh): ', del_wh)
        print('len of xt: ', len(xt))
        del_wh = np.dot(del_wh, xt)
        print('')
        '''
    return z_h_l, yi_l, rt, whj_list, vih, r_c


def test_mlp(d_set, d_rt, z_h_l, yi_l, whj_list, vih, r_c):

    yi_l = {}
    d_rt
    hval = len(vih[0])

    sum = 0

    cnt = 0

    #for idx in r_c:
    for t in range(len(r_c)):
        idx = r_c[t]
        xt = d_set[idx]
        rt = d_rt[idx]
        whj = whj_list[idx]

        zh = list([1])

        for h in range(hval - 1):
            zh.append(1 / (1 + (-1 * np.exp(np.dot(whj[h], xt), dtype=np.float64))))

        yi = yi = list()
        yi.append(sigmoid(vih[idx], zh))
        if yi[0] > 0:
            yi[0] = 1
        else:
            yi[0] = 0

        yi = np.array(yi, dtype=np.float64)

        yi_l[idx] = yi[0]

        sum += -1 * (rt * np.log(yi[0]) + (1 - rt) * np.log(1-yi[0]))

        cnt += 1

    mse = (sum**2)/2

    return mse
