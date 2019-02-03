from DataProcessor import *
import numpy as np
from ML_Visualizations import *
from performance_tests import *
from DisplayMethods import *


def sigmoid(w,x):
    """
        will perform a basic sigmoid function
    :param w: numpy array
    :param x: numpy array
    :return:
    """
    s = 1/(1 + (np.exp(np.dot((-1*w.T), x), dtype=np.float64)))
    return s


# def init_w(hval, d_size, ch=[-.01, .01]):
def init_w(hval, d_size, ch=None):
    whj = list()
    if ch is None:
        ch = np.linspace(-.01, .01, d_size)

    for h in range(hval):
        whj.append(np.random.choice(ch, d_size, replace=True))
    return whj


# def init_vih(hval, ch=[-.01, .01]):
def init_vih(hval, ch=None):
    if ch is None:
        ch = np.linspace(-.01, .01, hval)
    return np.random.choice(ch, hval, replace=True)


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


# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------


def back_propagation(train_x, train_r, val_set, whj, vih, r_c, hval, eta, epoch, mse_limit, del_mse, ep_ret=False):
    tr_mse = {}
    vl_mse = {}
    best_mse = 100
    best_msetr = 100
    best_whj = np.array([], dtype=np.float64)
    best_vih = np.array([], dtype=np.float64)

    w_dict = {}
    v_dict = {}

    best_tr_ps = []
    best_vl_ps = []

    tr_per_avg, vl_per_avg = {}, {}

    e_cntr = 0
    e_cnt = 0
    old_mse = 10

    msev, mset = 0, 0

    while True:
        e_cntr += 1
        r_c = np.random.choice(r_c, len(r_c), replace=False)
        # print('-------------------------------------------------length of good list', len(good_wh))
        print('------------------------------------------------------epoch count', e_cntr)
        for idx in r_c:
            xt = train_x[idx]  # grab observation at idx
            rt = train_r[idx]  # grab classification at idx for this observation

            zh = list([1])
            for h in range(hval - 1):
                zh.append(sigmoid(whj[h], xt))

            # calculate new predictions of class value (>0 == 1, <=0 == 0)
            yi = list()
            yi.append(sigmoid(vih, zh))
            if yi[0] > 0:
                yi[0] = 1
            else:
                yi[0] = 0
            yi = np.array(yi, dtype=np.float64)

            # calculate the new delta vih
            del_vih = np.dot((eta * (rt - yi[0])), zh)

            # calculate the delta whj
            d_wh = list()
            for h in range(hval):
                val = (rt - yi[0]) * vih[h]
                val = eta * val
                val = val * zh[h]
                diff = 1 - zh[h]
                val = val * diff
                val = np.dot(val, xt)
                d_wh.append(val)

            # make adjustments to weights
            old_vih = np.array(vih.tolist(), dtype=np.float64)
            vih = vih + del_vih

            old_whj = np.array(whj.tolist(), dtype=np.float64)
            whj = np.add(whj, d_wh)

        # once all observations have been processed and used to make adjustments
        # test and calculate the error of the modeling of the training data
        # with the current model
        trps = test_mlp3(train_x, whj, vih)
        mset = mse_ann(trps, train_r)
        print('Training MSE: ', mset)

        # if want to keep track of which epoch
        # lead to what version of trained mlp
        if ep_ret:
            w_dict[e_cntr] = whj
            tr_mse[e_cntr] = mset
            cm, p_a = confusion_matrix(trps, train_r)
            tr_per_avg[e_cntr] = np.mean(p_a, dtype=np.float64)

        if val_set is not None:
            vlps = test_mlp3(val_set[0], whj, vih)
            msev = mse_ann(vlps, val_set[1])
            print('val mse: ', msev)
            if ep_ret:
                vl_mse[e_cntr] = msev
                v_dict[e_cntr] = vih
                cm, p_a2 = confusion_matrix(vlps, val_set[1])
                vl_per_avg[e_cntr] = np.mean(p_a2, dtype=np.float64)
            if msev < best_mse:
                print('------------------------------------------------------->best validation mse is now: ', msev)
                print('------------------------------------------------------->best epoch is ', e_cntr)
                best_mse = msev
                best_msetr = mset
                best_whj = whj
                best_vih = vih
                best_tr_ps = trps
                best_vl_ps = vlps

        if mse_limit is not None and mse_limit > msev:
            if val_set is not None:
                print('val mse break: ', msev)
            else:
                print('mse break: ', mset)
            break
        elif del_mse is not None and e_cntr > 1 and (old_mse - msev) < del_mse:
            # elif del_mse is not None and e_cntr > 1 and np.around((old_mse - mse), 2) <= del_mse or (old_mse - mse) < .0001:
            # elif del_mse is not None and abs(old_mse - mse) <= del_mse:
            print('----------------------------------------------------change in mse break: ', (old_mse - msev))
            mse = old_mse
            whj = old_whj
            vih = old_vih
            break
        else:
            old_mse = msev

        if epoch is not None:
            e_cnt += 1
            if e_cnt == epoch:
                print('------------------------------------------------------broke due to epoch')
                break

    if ep_ret:
        ep_l = list([tr_mse, vl_mse, tr_per_avg, vl_per_avg])
        w_v = list([w_dict, v_dict])
        return best_whj, best_vih, best_mse, best_msetr, best_tr_ps, best_vl_ps, ep_l, w_v
    return best_whj, best_vih, best_mse, best_msetr, best_tr_ps, best_vl_ps

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------


def mlp_trainer2(train_x, train_r, eta, hval, epoch=None, test_limit=None, whj=None, vih=None, r_c=None, verbose=False,
                 mse_limit=None, del_mse=None, val_set=None, ep_ret=False):
    e_cntr = 0

    if r_c is None:
        r_c = np.random.choice(len(train_x), len(train_x), replace=False)

    if whj is None:
        whj = init_w(hval, len(train_x[0]))

    if vih is None:
        vih = init_vih(hval)

    '''
    if test_limit is None:
        test_limit = len(train_x)

    good_wh = []
    z_h_l = {}
    yi_l = {}
    best_mse = 100
    best_msetr = 100
    best_whj = np.array([], dtype=np.float64)
    best_vih = np.array([], dtype=np.float64)
    best_tr_ps = []
    best_vl_ps = []
    e_cnt = 0
    old_mse = 10
    f_cnt = 0
    '''

    if ep_ret:
        best_whj, best_vih, best_mse, best_msetr, best_tr_ps, best_vl_ps, ep_l, w_v = back_propagation(train_x, train_r,
                                                                                                  val_set, whj, vih,
                                                                                                  r_c, hval, eta,
                                                                                                  epoch, mse_limit,
                                                                                                  del_mse, ep_ret=ep_ret)
    else:
        best_whj, best_vih, best_mse, best_msetr, best_tr_ps, best_vl_ps = back_propagation(train_x, train_r, val_set,
                                                                                            whj, vih, r_c, hval, eta,
                                                                                            epoch, mse_limit, del_mse,
                                                                                            ep_ret=ep_ret)

    '''
    # print('-----------------------------------------------------------> test limit: ', test_limit)
    #while len(good_wh) < test_limit:
    while True:

        e_cntr += 1
        r_c = np.random.choice(r_c, len(r_c), replace=False)
        print('-------------------------------------------------length of good list', len(good_wh))
        print('------------------------------------------------------epoch count', e_cntr)
        for idx in r_c:
            #if idx in good_wh:
            #    continue
            f_cnt += 1
            xt = train_x[idx]           # grab observation at idx
            rt = train_r[idx]           # grab classification at idx for this observation

            zh = list([1])
            for h in range(hval-1):
                zh.append(sigmoid(whj[h], xt))

            z_h_l[idx] = zh

            # calculate new predictions of class value (>0 == 1, <0 == 0)
            yi = list()
            yi.append(sigmoid(vih, zh))
            if yi[0] > 0:
                yi[0] = 1
            else:
                yi[0] = 0
            yi = np.array(yi, dtype=np.float64)

            yi_l[idx] = yi[0]

            # calculate the new delta vih
            # print('calculated del vih')
            #del_vih = nu * (rt - yi[0])
            #del_vih = np.dot(del_vih, zh)
            del_vih = np.dot((eta * (rt - yi[0])), zh)

            d_wh = list()
            for h in range(hval):
                val = (rt - yi[0]) * vih[h]
                val = eta * val
                val = val * zh[h]
                diff = 1 - zh[h]
                val = val * diff
                val = np.dot(val, xt)
                d_wh.append(val)

            # make adjustments to weights
            old_vih = np.array(vih.tolist(), dtype=np.float64)
            #vih = np.add(del_vih, vih)
            vih = vih + del_vih
            old_whj = np.array(whj.tolist(), dtype=np.float64)
            whj = np.add(whj, d_wh)

            #dif_wh = np.around(np.mean(old_whj - whj, dtype=np.float64), 2)
            #dif_wh = np.around(np.mean(old_whj - whj, dtype=np.float64), 3)
            dif_wh = np.around(np.mean(old_whj - whj, dtype=np.float64), 9)
            #dif_wh = np.around(np.mean(old_whj - whj, dtype=np.float64), 5)
            #dif_wh = np.around(np.mean(old_whj - whj, dtype=np.float64), 9)

            #if np.around(np.mean(dif_wh, dtype=np.float64), 9) == 0:
            if len(good_wh) < len(r_c) and abs(dif_wh) == 0:
                good_wh.append(idx)

        #if len(good_wh) == len(r_c):
        #    break

        trps = test_mlp3(train_x, whj, vih)
        mset = mse_ann(trps, train_r)
        print('Train mse: ', mset)

        if val_set is not None:
            vlps = test_mlp3(val_set[0], whj, vih)
            msev = mse_ann(vlps, val_set[1])
            print('val mse: ', msev)
            if msev < best_mse:
                print('------------------------------------------------------->best validation mse is now: ', msev)
                print('------------------------------------------------------->best epoch is ', e_cntr)
                best_mse = msev
                best_msetr = mset
                best_whj = whj
                best_vih = vih
                best_tr_ps = trps
                best_vl_ps = vlps

        if mse_limit is not None and mse_limit > msev:
            if val_set is not None:
                print('val mse break: ', msev)
            else:
                print('mse break: ', mset)
            break
        elif del_mse is not None and e_cntr > 1 and (old_mse - msev) < del_mse:
        #elif del_mse is not None and e_cntr > 1 and np.around((old_mse - mse), 2) <= del_mse or (old_mse - mse) < .0001:
        # elif del_mse is not None and abs(old_mse - mse) <= del_mse:
            print('----------------------------------------------------change in mse break: ', (old_mse-msev))
            mse = old_mse
            whj = old_whj
            vih = old_vih
            break
        else:
            old_mse = msev

        if epoch is not None:
            e_cnt += 1
            if e_cnt == epoch:
                print('------------------------------------------------------broke due to epoch')
                break
    '''
    whj = best_whj
    vih = best_vih
    #ps = test_mlp3(train_x, whj, vih)
    cm, p_a = confusion_matrix(best_tr_ps, train_r)
    dis_conf_perform(cm, p_a)

    cm2, p_a2 = confusion_matrix(best_vl_ps, val_set[1])
    dis_conf_perform(cm2, p_a2)

    tr_pr = [cm, p_a]
    vl_pr = [cm2, p_a2]

    pr_list = list([tr_pr, vl_pr])

    #mse = mse_ann(best_tr_ps, train_r)
    #print('mse: ', mse)
    print('best mse for training: ', best_msetr)
    if verbose:
        print("Went through {:d} epochs ".format(e_cntr))
        # print(' f_cnt', f_cnt)

    tr_avg = np.mean(p_a, dtype=np.float64)
    vl_avg = np.mean(p_a2, dtype=np.float64)

    if ep_ret:
        return whj, vih, best_msetr, tr_avg, vl_avg, best_mse, pr_list, ep_l, w_v
    return whj, vih, best_msetr, tr_avg, vl_avg, best_mse, pr_list


def mlp_trainer3(train_x, train_r, eta, hval, epoch=None, test_limit=None, whj=None, vih=None, r_c=None, verbose=False,
                 mse_limit=None, del_mse=None, val_set=None):
    e_cntr = 0

    if r_c is None:
        r_c = np.random.choice(len(train_x), len(train_x), replace=False)

    if whj is None:
        whj = init_w(hval, len(train_x[0]))

    if vih is None:
        vih = init_vih(hval)

    if test_limit is None:
        test_limit = len(train_x)

    good_wh = []

    z_h_l = {}
    yi_l = {}

    tr_mse = {}
    vl_mse = {}

    whj_dict = {}
    vih_dict = {}

    best_mse = 100
    best_epoch = 0
    best_whj = np.array([], dtype=np.float64)
    best_vih = np.array([], dtype=np.float64)

    e_cnt = 0
    old_mse = 10
    f_cnt = 0

    best_whj, best_vih, best_mse, best_msetr, best_tr_ps, best_vl_ps, ep_l, w_v = back_propagation(train_x, train_r,
                                                                                                   val_set, whj, vih,
                                                                                                   r_c, hval, eta,
                                                                                                   epoch, mse_limit,
                                                                                                   del_mse,
                                                                                                   ep_ret=True)
    '''
    # print('-----------------------------------------------------------> test limit: ', test_limit)
    #while len(good_wh) < test_limit:
    while True:

        e_cntr += 1
        r_c = np.random.choice(r_c, len(r_c), replace=False)
        print('-------------------------------------------------length of good list', len(good_wh))
        print('------------------------------------------------------epoch count', e_cntr)
        for idx in r_c:
            #if idx in good_wh:
            #    continue
            f_cnt += 1
            xt = train_x[idx]           # grab observation at idx
            rt = train_r[idx]           # grab classification at idx for this observation

            zh = list([1])
            for h in range(hval-1):
                zh.append(sigmoid(whj[h], xt))

            z_h_l[idx] = zh

            # calculate new predictions of class value (>0 == 1, <0 == 0)
            yi = list()
            yi.append(sigmoid(vih, zh))
            if yi[0] > 0:
                yi[0] = 1
            else:
                yi[0] = 0
            yi = np.array(yi, dtype=np.float64)

            yi_l[idx] = yi[0]

            # calculate the new delta vih
            # print('calculated del vih')
            #del_vih = nu * (rt - yi[0])
            #del_vih = np.dot(del_vih, zh)
            del_vih = np.dot((eta * (rt - yi[0])), zh)

            d_wh = list()
            for h in range(hval):
                val = (rt - yi[0]) * vih[h]
                val = eta * val
                val = val * zh[h]
                diff = 1 - zh[h]
                val = val * diff
                val = np.dot(val, xt)
                d_wh.append(val)

            # make adjustments to weights
            old_vih = np.array(vih.tolist(), dtype=np.float64)
            #vih = np.add(del_vih, vih)
            vih = vih + del_vih
            old_whj = np.array(whj.tolist(), dtype=np.float64)
            whj = np.add(whj, d_wh)

            whj_dict[e_cntr] = whj
            vih_dict[e_cntr] = vih


            #dif_wh = np.around(np.mean(old_whj - whj, dtype=np.float64), 2)
            #dif_wh = np.around(np.mean(old_whj - whj, dtype=np.float64), 3)
            dif_wh = np.around(np.mean(old_whj - whj, dtype=np.float64), 9)
            #dif_wh = np.around(np.mean(old_whj - whj, dtype=np.float64), 5)
            #dif_wh = np.around(np.mean(old_whj - whj, dtype=np.float64), 9)

            #if np.around(np.mean(dif_wh, dtype=np.float64), 9) == 0:
            if len(good_wh) < len(r_c) and abs(dif_wh) == 0:
                good_wh.append(idx)

        #if len(good_wh) == len(r_c):
        #    break

        ys, ps = test_mlp3(train_x, train_r, whj, vih)
        mse = mse_ann(ps, ys)
        print('Train mse: ', mse)

        tr_mse[e_cntr] = mse

        if val_set is not None:
            ys, ps = test_mlp3(val_set[0], val_set[1], whj, vih)
            mse = mse_ann(ps, ys)
            print('Validation mse: ', mse)
            if mse < best_mse:
                print('------------------------------------------------------->best mse is now: ', mse)
                print('------------------------------------------------------->best epoch is ', e_cntr)
                best_mse = mse
                best_whj = whj
                best_vih = vih
                best_epoch = e_cntr

        vl_mse[e_cntr] = mse

        if mse_limit is not None and mse_limit >= mse:
            if val_set is not None:
                print('val mse break: ', mse)
            else:
                print('mse break: ', mse)
            break
        elif del_mse is not None and e_cntr > 1 and (np.around((old_mse - mse), 4) <= del_mse or (old_mse - mse) < .0001):
        #elif del_mse is not None and e_cntr > 1 and np.around((old_mse - mse), 2) <= del_mse or (old_mse - mse) < .0001:
        # elif del_mse is not None and abs(old_mse - mse) <= del_mse:
            print('----------------------------------------------------change in mse break: ', np.around(old_mse-mse))
            mse = old_mse
            whj = old_whj
            vih = old_vih
            break
        else:
            old_mse = mse

        if epoch is not None:
            e_cnt += 1
            if e_cnt == epoch:
                print('------------------------------------------------------broke due to epoch')
                break
    '''

    whj = best_whj
    vih = best_vih

    #print('Training Results:')
    #ys, ps = test_mlp3(train_x, train_r, whj, vih)

    cmt, p_a = confusion_matrix(best_tr_ps, train_r)
    #dis_conf_perorm(cm, p_a)

    #mse = mse_ann(ps, ys)
    #print('mse: ', mse)

    #print('Validation Results:')
    #ys, ps = test_mlp3(val_set[0], val_set[1], whj, vih)
    cmv, p_a2 = confusion_matrix(best_vl_ps, val_set[1])
    #dis_conf_perorm(cm, p_a)

    pa = list([p_a, p_a2, cmt, cmv])

    #mse_ls = list([tr_mse, vl_mse])
    mse_ls = list([ep_l[0], ep_l[1], ep_l[2], ep_l[3]])
    #whj_vih = list([whj_dict, vih_dict])
    whj_vih = list([whj_dict, vih_dict])

    if verbose:
        print("Went through {:d} epochs ".format(e_cntr))
        # print(' f_cnt', f_cnt)

    return whj, vih, pa, mse_ls, w_v


# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

# def test_mlp3(d_set, d_rt, whj, vih)
def test_mlp3(d_set, whj, vih):

    hval = len(vih)

    p_l = []

    #for idx in r_c:
    #for t in range(len(r_c)):
    for idx in range(len(d_set)):
        xt = d_set[idx]
        zh = list([1])
        for h in range(hval - 1):
            zh.append(sigmoid(whj[h], xt))

        yi = (sigmoid(vih, zh))
        if (sigmoid(vih, zh)) > 0:
            p_l.append(1.0)
        else:
            p_l.append(0.0)

    return p_l

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------


def perform_test_run(trn_set, val_set, eta, epoch, h_val, verbose, mse_limit, del_mse, whj, vih, r_c, ep_ret=False):
    tr_x = trn_set[0]
    tr_r = trn_set[1]

    # whj, vih, best_msetr, np.mean(p_a, dtype=np.float64), np.mean(p_a2, dtype=np.float64), best_mse, pr_list
    if ep_ret:
        whj_list, vih_l, mse2, tr_avg, val_avg, msevl, pr_list, ep_l = mlp_trainer2(tr_x, tr_r, eta, h_val, epoch=epoch,
                                                                                    val_set=val_set, whj=whj, vih=vih,
                                                                                    r_c=r_c, mse_limit=mse_limit,
                                                                                    del_mse=del_mse, ep_ret=ep_ret)
    else:
        whj_list, vih_l, mse2, tr_avg, val_avg, msevl, pr_list = mlp_trainer2(tr_x, tr_r, eta, h_val, epoch=epoch,
                                                                              val_set=val_set, whj=whj, vih=vih,
                                                                              r_c=r_c, mse_limit=mse_limit,
                                                                              del_mse=del_mse, ep_ret=ep_ret)
    if verbose:
        print('Training MSE2 for best validation: \n', mse2)
        print('average performance: ', tr_avg)
        print('-------------------------------')
    if verbose:
        print('MSE for best validation: \n', msevl)
        print('average performance: ', val_avg)
        print('-------------------------------')

    if ep_ret:
        return mse2, msevl, whj_list, vih_l, tr_avg, val_avg, pr_list, ep_l
    return mse2, msevl, whj_list, vih_l, tr_avg, val_avg, pr_list

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------


def perform_eta_test(trn_set, val_set, tst_set, eta_a=None, epochl=300, h_val=5, runs=1, verbose=False, mse_limit=None,
                     del_mse=None):
    tr_x = trn_set[0]
    tr_r = trn_set[1]
    val_x = val_set[0]
    val_r = val_set[1]
    ts_x = tst_set[0]
    ts_r = tst_set[1]
    ds = len(trn_set[0])

    if eta_a is None:
        # nu_a = list([.2, .1, .09, .08, .07, .06, .05, .04, .03, .02])
        # eta_a = list(map(float, np.linspace(.09, .0001, 10)))
        # eta_a = list(map(float, np.linspace(.0001, .01, 10)))
        # eta_a = list(map(float, np.linspace(.001, .009, 20)))
        # eta_a = list(map(float, np.linspace(.004, .007, 10)))
        # eta_a = list(map(float, np.linspace(.004, .007, 20)))
        # eta_a = list([.2, .1, .09, .08, .07, .06, .05, .04, .03, .01, .006, .005, .004 ])
        eta_a = list([.2, .1, .09, .05, .02, .008, .006, .005, .004])
        #eta_a = list([.1, .004])
        if verbose:
            print('eta\'s:')
            print(eta_a)

    mse_list = []
    mseval_list = []

    per_list = []
    per_val_list = []

    mse_dict = {}
    mseval_dict = {}

    tr_per_dict = {}
    val_per_dict = {}

    whj_dict = {}
    vih_dict = {}

    orig_whj = np.array(init_w(h_val, len(tr_x[0]), ch=None), dtype=np.float64)
    #orig_whj = np.array(init_w(h_val, len(tr_x[0])), dtype=np.float64)
    orig_vih = init_vih(h_val, ch=None)
    #orig_vih = init_vih(h_val)

    for n in eta_a:
        eta = n
        whj = np.array(orig_whj.tolist(), dtype=np.float64)
        vih = np.array(orig_vih.tolist(), dtype=np.float64)
        r_c = np.random.choice(len(tr_x), len(tr_x), replace=False)
        if verbose:
            print('-------------------------------------------------------------------------> testing eta: ', eta)
        best_mse_v = 0

        '''
        for t in range(runs):
            # whj_list, vih, mse2 = mlp_trainer2(tr_x, tr_r, nu, hval, test_limit=test_limit, epoch=epoch,
            whj_list, vih, mse2, tr_avg, msevl = mlp_trainer2(tr_x, tr_r, eta, h_val, epoch=epochl, val_set=val_set,
                                                           whj=whj, vih=vih, r_c=r_c, mse_limit=.05, del_mse=None)
        '''

        # perform_test_run(trn_set, val_set, eta, epoch, h_val, verbose, mse_limit, del_mse, whj, vih, r_c ):

        mse2, mseval, whj_list, vih, tr_avg, val_avg, pr_list = perform_test_run(trn_set, val_set, eta, epochl, h_val,
                                                                                 verbose, mse_limit, del_mse, whj,
                                                                                 vih, r_c)

        mse_dict[n] = mse2
        mseval_dict[n] = mseval

        whj_dict[n] = whj_list
        vih_dict[n] = vih

        tr_per_dict[n] = tr_avg
        val_per_dict[n] = val_avg

    mse_list = [mse_dict, mseval_dict]
    performance_list = [tr_per_dict, val_per_dict]

    mse_results, performance_results = dis_training_results(mse_list, performance_list, runs, hval=h_val, epoch=epochl,
                                                            keytype='Eta')

    top_tup = mse_results[0]
    train_pts = mse_results[1]
    top_tupval = mse_results[2]
    val_pts = mse_results[3]

    top_tup1 = performance_results[0]
    train_pts1 = performance_results[1]
    top_tupval1 = performance_results[2]
    val_pts1 = performance_results[3]

    '''
    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('Runs: ', runs)
    print('hvals: ', h_val)
    print('epoch: ', epochl)
    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('----------------------------------   mse results  ----------------------------------------------------')
    top_tup, train_pts = process_mlp_testing_results(mse_dict, title='Training Result', key_type='Eta',
                                                     val_type='Average MSE', verbose=True, reverse=False)
    top_tupval, val_pts = process_mlp_testing_results(mseval_dict, title='Validation Result', key_type='Eta',
                                                      val_type='Average MSE', verbose=True, reverse=False)
    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('----------------------------------   performance  results  -------------------------------------------')
    top_tup1, train_pts_per = process_mlp_testing_results(tr_per_dict, title='Training Result', key_type='Eta',
                                                          val_type='Average Performance', verbose=True, reverse=True)
    top_tupval1, val_pts_per = process_mlp_testing_results(val_per_dict, title='Validation Result', key_type='Eta',
                                                           val_type='Average Performance', verbose=True, reverse=True)
    '''

    '''
    best_eta = top_tupval[0]
    print('Using MSE: best Eta {:f}'.format(best_eta))
    best_whj = whj_dict[best_eta]
    best_vih = vih_dict[best_eta]
    ys, ps = test_mlp3(ts_x, ts_r, best_whj, best_vih)
    msetst = mse_ann(ps, ys)
    print('Mse for test data: ', msetst)
    print('-------------------------------------------------')
    print('-------------------------------------------------')
    cm, p_a = confusion_matrix(ps, ys)
    dis_conf_perorm(cm, p_a)
    print('--------------------------------')
    print('--------------------------------')
    '''
    best_eta = display_trained_mlp(top_tupval, tst_set, whj_dict, vih_dict, title='Using MSE: best eta {:f}',
                                   data_title='Mse for test data: ')

    '''
    best_eta1 = top_tupval1[0]
    print('Using Averaged Performance: best Eta {:f}'.format(best_eta1))
    best_whj = whj_dict[best_eta1]
    best_vih = vih_dict[best_eta1]
    ys, ps = test_mlp3(ts_x, ts_r, best_whj, best_vih)
    msetst = mse_ann(ps, ys)
    print('Mse for test data: ', msetst)
    print('-------------------------------------------------')
    print('-------------------------------------------------')
    cm, p_a = confusion_matrix(ps, ys)
    dis_conf_perorm(cm, p_a)
    '''

    #best_eta1 = display_trained_mlp(top_tupval1, tst_set, whj_dict, vih_dict,
    #                                title='Using Average Performance: best eta {:f}',
    #                                data_title='Mse for test data: ')

    y_a = list()
    y_a.append(train_pts)
    y_a.append(val_pts)

    title = 'Learning Rate vs. MSE: hval = {:d}, epoch limit {:d}'.format(h_val, epochl)
    multi_y_plotter(eta_a, y_a, title=title, leg_a=['Training', 'Validation', 'purple'],
                    x_label='Learning Rate(eta)', y_label='MSE', show_it=True)

    '''
    y_a2 = list()
    y_a2.append(train_pts1)
    y_a2.append(val_pts1)

    title = 'Learning Rate vs. average performance: hval = {:d}, epoch limit {:d}'.format(h_val, epochl)
    multi_y_plotter(eta_a, y_a2, title=title, leg_a=['Training', 'Validation', 'purple'],
                    x_label='Learning Rate(eta)', y_label='Average Performance', show_it=True)
    '''

    return best_eta


def perform_epoch_test(trn_set, val_set, tst_set, eta_a=.1, epochl=None, h_val=5, runs=1, verbose=False,
                       mse_limit=None, del_mse=None):

    ds = len(trn_set[0])

    if epochl is None:
        #epochl = list(map(int, np.linspace(5, 300, 10)))
        #epochl = list(map(int, np.linspace(2, 80, 20)))
        #epochl = list(map(int, np.linspace(10, 60, 10)))
        epochl = range(1, 190)
        #epochl.insert(0,2)
        print('epoch')
        print(epochl)

    '''
    mse_list = []
    per_list = []

    mseval_list = []
    per_val_list = []

    mse_dict = {}
    mseval_dict = {}

    tr_per_dict = {}
    val_per_dict = {}

    whj_dict = {}
    vih_dict = {}
    '''

    orig_whj = np.array(init_w(h_val, len(trn_set[0][0]), ch=None), dtype=np.float64)
    orig_vih = init_vih(h_val, ch=None)

    epoch_dict = {}

    whj = np.array(orig_whj.tolist(), dtype=np.float64)
    vih = np.array(orig_vih.tolist(), dtype=np.float64)

    r_c = np.random.choice(len(trn_set[0]), len(trn_set[0]), replace=False)

    whj_list = dict()

    best_mse_v = 0
    best_tr = 0

    eta = eta_a
    epoch = epochl

    mse2, mseval, whj_list, vih, tr_avg, val_avg, pr_list, ep_l = perform_test_run(trn_set, val_set, eta, epoch,
                                                                                   h_val, verbose, mse_limit,
                                                                                   del_mse, whj, vih, r_c,
                                                                                   ep_ret=True)

    mse_dict = ep_l[0]
    mseval_dict = ep_l[1]

    whj_dict = whj_list
    vih_dict = vih

    tr_per_dict = ep_l[2]
    val_per_dict = ep_l[3]

    mse_list = [mse_dict, mseval_dict]
    performance_list = [tr_per_dict, val_per_dict]

    mse_results, performance_results = dis_training_results(mse_list, performance_list, runs, hval=h_val, eta=eta_a)

    top_tup = mse_results[0]
    train_pts = mse_results[1]
    top_tupval = mse_results[2]
    val_pts = mse_results[3]

    top_tup1 = performance_results[0]
    train_pts1 = performance_results[1]
    top_tupval1 = performance_results[2]
    val_pts1 = performance_results[3]

    '''
    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('Runs: ',runs )
    print('hvals: ', h_val)
    print('eta: ', eta_a)
    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('----------------------------------   mse results  ----------------------------------------------------')
    top_tup, train_pts = process_mlp_testing_results(mse_dict, title='Training Result', key_type='Epoch',
                                                     val_type='Average MSE', verbose=True, reverse=False)
    top_tupval, val_pts = process_mlp_testing_results(mseval_dict, title='Validation Result', key_type='Epoch',
                                                      val_type='Average MSE', verbose=True, reverse=False)

    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('----------------------------------   performance results  --------------------------------------------')
    top_tup1, train_pts1 = process_mlp_testing_results(tr_per_dict, title='Training Result', key_type='Epoch',
                                                       val_type='Average Performance', verbose=True, reverse=True, )
    top_tupval1, val_pts1 = process_mlp_testing_results(val_per_dict, title='Validation Result', key_type='Epoch',
                                                        val_type='Average Performance', verbose=True, reverse=True)
    '''

    '''
    best_epoch = top_tupval[0]
    print('Using MSE: best epoch {:d}'.format(best_epoch))
    best_whj = whj_dict[best_epoch]
    best_vih = vih_dict[best_epoch]
    #ys, ps = test_mlp3(ts_x, ts_r, best_whj, best_vih)
    ys, ps = test_mlp3(tst_set[0], tst_set[1], best_whj, best_vih)
    msetst = mse_ann(ps, ys)
    print('Mse for test data: ', msetst)
    print('--------------------------------')
    print('--------------------------------')
    cm, p_a = confusion_matrix(ps, ys)
    dis_conf_perorm(cm, p_a)
    print('-------------------------------')
    print('-------------------------------')
    '''

    best_epoch = display_trained_mlp(top_tupval, tst_set, whj_dict, vih_dict, title='Using MSE: best epoch {:d}',
                                     data_title='Mse for test data: ')

    '''
    best_epoch1 = top_tupval1[0]
    print('Using Average Performance: best epoch {:d}'.format(best_epoch1))
    best_whj = whj_dict[best_epoch1]
    best_vih = vih_dict[best_epoch1]
    ys, ps = test_mlp3(tst_set[0], tst_set[1], best_whj, best_vih)
    msetst = mse_ann(ps, ys)
    print('Mse for test data: ', msetst)
    print('-------------------------------------------------')
    print('-------------------------------------------------')
    cm, p_a = confusion_matrix(ps, ys)
    dis_conf_perorm(cm, p_a)
    '''

    #best_epoch1 = display_trained_mlp(top_tupval1, tst_set, whj_dict, vih_dict,
    #                                  title='Using Average Performance: best epoch {:d}',
    #                                  data_title='Mse for test data: ')

    y_a = list()
    y_a.append(train_pts)
    y_a.append(val_pts)
    title = 'Epoch vs. MSE: hval={:d}, nu={:f}'.format(h_val, eta_a)
    multi_y_plotter(epochl, y_a, title=title, leg_a=['Training', 'Validation', 'purple'], x_label='Epoch',
                    y_label='MSE', show_it=True)
    '''
    y_a1 = list()
    y_a1.append(train_pts1)
    y_a1.append(val_pts1)
    title = 'Epoch vs. Performance: hval={:d}, nu={:f}'.format(h_val, eta_a)
    multi_y_plotter(epochl, y_a1, title=title, leg_a=['Training', 'Validation', 'purple'], x_label='Epoch',
                    y_label='Average Performance', show_it=True)
    '''
    return best_epoch


def perform_hval_test(trn_set, val_set, tst_set, eta_a=.1, epochl=300, h_val=None, runs=1, verbose=True,
                      mse_limit=None, del_mse=None):
    '''
    tr_x = trn_set[0]
    tr_r = trn_set[1]

    val_x = val_set[0]
    val_r = val_set[1]

    ts_x = tst_set[0]
    ts_r = tst_set[1]
    '''

    #ds = len(trn_set[0])

    # print('val_set: \n',val_set[0][0])
    # print('test set: \n',tst_set[0][0])

    if h_val is None:
        #h_val = list(map(int, np.linspace(2, len(trn_set[0][0]), 10)))
        #h_val = list([230])
        #h_val = list([230])
        #h_val = list([0])
        #h_val = list(map(int, np.linspace(150, 250, 20)))
        #h_val = list(map(int, np.linspace(50, 100, 6)))
        h_val = list([10, 30, 50,   60,   70,   80,   90,   100, 120, 130, 140, 150, 170, 190, 200])
        #h_val = list(map(int, np.linspace(200, 400, 10)))
        #h_val = list(map(int, np.linspace(220, 240, 4)))
        #h_val = list(map(int, np.linspace(170, 230, 7)))
        #h_val = list(map(int, np.linspace(210, 270, 7)))
        if verbose:
            print('hvals:')
            print(h_val)

    mse_list = []
    mseval_list = []

    per_list = []
    per_val_list = []

    mse_dict = {}
    mseval_dict = {}

    tr_per_dict = {}
    val_per_dict = {}

    whj_dict = {}
    vih_dict = {}

    #orig_whj = np.array(init_w(h_val, len(tr_x[0])), dtype=np.float64)
    #orig_whj = np.array(init_w(h_val, len(trn_set[0][0])), dtype=np.float64)
    #orig_vih = init_vih(h_val)

    for n in h_val:

        eta = eta_a
        #epoch = epochl
        #test_limit = len(tr_x)
        test_limit = len(trn_set[0])
        hval = n

        #whj = np.array(init_w(hval, len(trn_set[0][0])), dtype=np.float64)

        #whj = np.array(init_w(hval, len(trn_set[0][0]), ch=None), dtype=np.float64)
        #vih = init_vih(hval, ch=None)

        whj = np.array(init_w(hval, len(trn_set[0][0])), dtype=np.float64)
        vih = init_vih(hval)

        #whj = np.array(orig_whj.tolist(), dtype=np.float64)
        #vih = np.array(orig_vih.tolist(), dtype=np.float64)

        r_c = np.random.choice(len(trn_set[0]), len(trn_set[0]), replace=False)

        whj_list = dict()

        if verbose:
            print('-------------------------------------------------------------------------> testing hval: ', hval)

        best_mse_v = 0
        best_tr = 0

        '''
        for t in range(runs):
            whj_list, vih, mse2, tr_avg = mlp_trainer2(trn_set[0], trn_set[1], eta_a, hval, test_limit=test_limit,
                                                       epoch=epochl, whj=whj, vih=vih, r_c=r_c.copy(),
                                                       val_set=val_set, mse_limit=mse_limit)
            mse_list.append(mse2)
            per_list.append(tr_avg)
            # epoch_list.append(mse2)

            if verbose:
                print('Run: ', t + 1)
                print('MSE2: \n', mse2)
                print('-------------------------------')
                print('-------------------------------')

            ys, ps = test_mlp3(val_set[0], val_set[1], whj_list, vih)

            mseval = mse_ann(ps, ys)

            cm, p_a = confusion_matrix(ps, ys)

            avg_per = np.mean(p_a, dtype=np.float64)
            per_val_list.append(avg_per)

            mseval_list.append(mseval)

            if avg_per > best_mse_v:
                best_mse_v = avg_per
                whj_dict[n] = whj_list
                vih_dict[n] = vih
            if verbose:
                print('MSE for validation: \n', mseval)
                print('-------------------------------')
                print('-------------------------------')

            dis_conf_perform(cm, p_a)
        '''

        mse2, mseval, whj_list, vih, tr_avg, val_avg, pr_list = perform_test_run(trn_set, val_set, eta, epochl, h_val,
                                                                                 verbose, mse_limit, del_mse, whj,
                                                                                 vih, r_c)

        mse_dict[n] = mse2
        mseval_dict[n] = mseval

        whj_dict[n] = whj_list
        vih_dict[n] = vih

        tr_per_dict[n] = tr_avg
        val_per_dict[n] = val_avg

        '''
        mse_dict[n] = np.mean(np.array(mse_list, dtype=np.float64), dtype=np.float64)
        mseval_dict[n] = np.mean(np.array(mseval_list, dtype=np.float64), dtype=np.float64)
        tr_per_dict[n] = np.mean(np.array(per_list, dtype=np.float64), dtype=np.float64)
        val_per_dict[n] = np.mean(np.array(per_val_list, dtype=np.float64), dtype=np.float64)
        mse_list.clear()
        mseval_list.clear()
        per_list.clear()
        per_val_list.clear()
        '''

    train_pts = []
    val_pts = []

    mse_list = [mse_dict, mseval_dict]
    performance_list = [tr_per_dict, val_per_dict]

    mse_results, performance_results = dis_training_results(mse_list, performance_list, runs, epoch=epochl, eta=eta_a)

    top_tup = mse_results[0]
    train_pts = mse_results[1]
    top_tupval = mse_results[2]
    val_pts = mse_results[3]

    top_tup1 = performance_results[0]
    train_pts1 = performance_results[1]
    top_tupval1 = performance_results[2]
    val_pts1 = performance_results[3]

    '''
    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('runs: ', runs)
    print('epoch: ', epochl)
    print('eta: ', eta_a)

    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('---------------------------------    Mse result    ---------------------------------------------------')
    top_tup, train_pts = process_mlp_testing_results(mse_dict, title='Training Result', key_type='h value',
                                                     val_type='Average MSE',verbose=True, reverse=False)

    top_tupval, val_pts = process_mlp_testing_results(mseval_dict, title='Validation Result', key_type='h value',
                                                      val_type='Average MSE', verbose=True, reverse=False)

    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('---------------------------------    Performance result    -------------------------------------------')
    top_tup1, train_pts1 = process_mlp_testing_results(tr_per_dict, title='Training Result', key_type='h value',
                                                       val_type='Average Performance', verbose=True, reverse=True)

    top_tupval1, val_pts1 = process_mlp_testing_results(val_per_dict, title='Validation Result', key_type='h value',
                                                        val_type='Average Performance', verbose=True, reverse=True)

    '''

    best_hval = display_trained_mlp(top_tupval, tst_set, whj_dict, vih_dict, title='Using MSE: best hval {:d}',
                                    data_title='Mse for test data: ')

    '''
    best_hval = top_tupval[0]
    print('Using MSE: best hval {:d}'.format(best_hval))
    best_whj = whj_dict[best_hval]
    best_vih = vih_dict[best_hval]
    ys, ps = test_mlp3(tst_set[0], tst_set[1], best_whj, best_vih)
    msetst = mse_ann(ps, ys)
    print('Mse for test data: ', msetst)
    print('--------------------------------')
    print('--------------------------------')
    cm, p_a = confusion_matrix(ps, ys)
    dis_conf_perorm(cm, p_a)
    print('-------------------------------')
    print('-------------------------------')
    '''

    best_hval1 = display_trained_mlp(top_tupval1, tst_set, whj_dict, vih_dict,
                                     title='Using Average Performance: best hval {:d}',
                                     data_title='Mse for test data: ')

    '''
    best_hval1 = top_tupval1[0]
    print('Using Average Performance: best hval {:d}'.format(best_hval1))
    best_whj = whj_dict[best_hval1]
    best_vih = vih_dict[best_hval1]
    ys, ps = test_mlp3(tst_set[0], tst_set[1], best_whj, best_vih)
    msetst = mse_ann(ps, ys)
    print('Mse for test data: ', msetst)
    print('--------------------------------')
    print('--------------------------------')
    cm, p_a = confusion_matrix(ps, ys)
    dis_conf_perorm(cm, p_a)
    '''
    y_a = list()
    y_a.append(train_pts)
    y_a.append(val_pts)
    title = 'H value vs. MSE: epoch: {:d}, eta: {:f}'.format(epochl, eta_a)
    multi_y_plotter(h_val, y_a, title=title, leg_a=['training', 'validation', 'purple'], x_label='h value',
                    y_label='MSE', show_it=True)

    '''
    y_a1 = list()
    y_a1.append(train_pts1)
    y_a1.append(val_pts1)
    title = 'H value vs. Performance: epoch: {:d}, eta: {:f}'.format(epochl, eta_a)
    multi_y_plotter(h_val, y_a1, title=title, leg_a=['training', 'validation', 'purple'],
                    x_label='h value', y_label='Average Performance', show_it=True)
    '''

    return best_hval, best_hval1


def perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=None, h_val=5, runs=1, verbose=False,
                       mse_limit=None, del_mse=None, graph=False):

    if epoch is None:
        epoch = 75
        print('epoch')
        print(epoch)


    orig_whj = np.array(init_w(h_val, len(trn_set[0][0]), ch=None), dtype=np.float64)
    orig_vih = init_vih(h_val, ch=None)
    #orig_whj = np.array(init_w(h_val, len(trn_set[0][0])), dtype=np.float64)
    #orig_vih = init_vih(h_val)

    r_c = np.random.choice(len(trn_set[0]), len(trn_set[0]), replace=False)

    best_mse_v = 0
    best_tr = 0
    whj_vih = []
    mse_ls = []

    for r in range(runs):
        whj = np.array(orig_whj.tolist(), dtype=np.float64)
        vih = np.array(orig_vih.tolist(), dtype=np.float64)

        whj, vih, pa, mse_ls, whj_vih = mlp_trainer3(trn_set[0], trn_set[1], eta_a, h_val, test_limit=len(trn_set[0]),
                                                     epoch=epoch, whj=whj, vih=vih, r_c=r_c, verbose=verbose,
                                                     mse_limit=mse_limit, del_mse=del_mse, val_set=val_set)

    mse_dict = mse_ls[0]
    mseval_dict = mse_ls[1]

    whj_dict = whj_vih[0]
    vih_dict = whj_vih[1]

    mse_list = [mse_dict, mseval_dict]
    mse_results = dis_training_results1(mse_list, runs, hval=h_val, eta=eta_a, keytype='Epoch')

    patrn = pa[0]
    cmtrn = pa[2]

    paval = pa[1]
    cmval = pa[3]

    print('------------------------')
    print('------------------------')
    print('------------------------')
    print('------------------------')
    print('Training Result:')
    pstr = test_mlp3(trn_set[0], whj, vih)
    msetr = mse_ann(pstr, trn_set[1])
    print("Training mse: ", msetr)
    dis_conf_perform(cmtrn, patrn)

    print('Validation Result:')
    psval = test_mlp3(val_set[0], whj, vih)
    mseval = mse_ann(psval, val_set[1])
    print("Validation mse: ", mseval)
    dis_conf_perform(cmval, paval)

    print('Testing Result:')
    pstst = test_mlp3(tst_set[0], whj, vih)
    mseval = mse_ann(pstst, tst_set[1])
    cmtst, patst = confusion_matrix(pstst, tst_set[1])
    print("Testing mse: ", mseval)
    dis_conf_perform(cmtst, patst)


    top_tup = mse_results[0]
    train_pts = mse_results[1]
    top_tupval = mse_results[2]
    val_pts = mse_results[3]

    print('Training Result:')
    best_epochtrn, r_whj, r_vih = display_trained_mlp(top_tupval, trn_set, whj_dict, vih_dict, title='Using MSE: best epoch {:d}',
                                                      data_title='Mse for test data: ')
    print('validation result Result:')
    best_epochval, r_whj, r_vih = display_trained_mlp(top_tupval, val_set, whj_dict, vih_dict, title='Using MSE: best epoch {:d}',
                                                      data_title='Mse for test data: ')

    print('Testing Result:')
    best_epochtst, r_whj, r_vih = display_trained_mlp(top_tupval, tst_set, whj_dict, vih_dict, title='Using MSE: best epoch {:d}',
                                                      data_title='Mse for test data: ')

    if graph:
        y_a = list([train_pts, val_pts])
        title = 'Epoch vs. MSE: hval={:d}, eta={:f}'.format(h_val, eta_a)
        multi_y_plotter(np.linspace(1, epoch, len(train_pts)), y_a, title=title, leg_a=['Training', 'Validation', 'purple'], x_label='Epoch',
                        y_label='MSE', show_it=True)

    return best_epochval, r_whj, r_vih


def ann_mlp(data, eta=.2, hval=190, epoch=110, mse_limit=None, del_mse=None, trn_pct=.8, val_pct=.1, tst_pct=.1,
            verbose=False, ret_epoch=False):
    trn_set, val_set, tst_set, ksds = generate_data_sets_xy(data, ptrn=trn_pct, pval=val_pct, ptst=tst_pct,
                                                            verbose=False, rand=True, seed=True, normalize='z')

    best_epoch, b_whj, b_vih = perform_epoch_test2(trn_set, val_set, tst_set, eta_a=eta, epoch=epoch, h_val=hval,
                                                   verbose=verbose, mse_limit=mse_limit, del_mse=del_mse)

    if ret_epoch:
        return best_epoch, b_whj, b_vih
    else:
        return b_whj, b_vih