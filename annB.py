from ML_Visualizations import *
from performance_tests import *
from DisplayMethods import *


def sigmoid(w, x):
    """
        will perform a basic sigmoid function
    :param w: numpy array
    :param x: numpy array
    :return:
    """
    s = 1/(1 + (np.exp(np.dot((-1*w.T), x), dtype=np.float64)))
    return s


# creates an initialized set of input weights
# for a perceptron
def init_w(hval, d_size, ch=[-.01, .01]):
    whj = list()

    if ch is None:
        ch = np.linspace(-.01, .01, d_size)

    for h in range(hval):
        whj.append(np.random.choice(ch, d_size, replace=True))
    return whj


# creates an initialized set of hidden layers weights
# for a perceptron
def init_vih(hval, ch=[-.01, .01]):
    if ch is None:
        ch = np.linspace(-.01, .01, hval)
    return np.random.choice(ch, hval, replace=True)


# adds a bias column to an array
# takss multi dimensional array
def add_bias(data):
    d = data.tolist()
    ret_l = []
    for row in d:
        ret_l.append([1]+row)
    return np.array(ret_l, dtype=np.float64)


# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------   Training methods      -----------------------------------------------

def mlp_trainer(train_set, val_set, eta, hval, epoch=None, whj=None, vih=None, r_c=None, verbose=False,
                 mse_limit=None, del_mse=None):
    train_x = train_set[0]
    train_r = train_set[1]

    e_cntr = 0

    if r_c is None:
        r_c = np.random.choice(len(train_x), len(train_x), replace=False)

    if whj is None:
        whj = init_w(hval, len(train_x[0]))

    if vih is None:
        vih = init_vih(hval)

    if epoch is None:
        epoch = 75

    z_h_l = {}
    yi_l = {}

    tr_mse = {}
    vl_mse = {}

    whj_dict = {}
    vih_dict = {}

    best_mse = 100
    best_whj = np.array([], dtype=np.float64)
    best_vih = np.array([], dtype=np.float64)
    best_epoch = 0

    e_cnt = 0
    old_mse = 10
    # print('-----------------------------------------------------------> test limit: ', test_limit)
    # while len(good_wh) < test_limit:
    while True:

        e_cntr += 1
        r_c = np.random.choice(r_c, len(r_c), replace=False)
        # print('-------------------------------------------------length of good list', len(good_wh))
        print('--------------------------------------------------------->epoch count', e_cntr)
        for idx in r_c:
            xt = train_x[idx]  # grab observation at idx or inputs for input layer
            rt = train_r[idx]  # grab classification at idx for this observation

            zh = list([1])  # add bias to hidden layer set

            # use weights and inputs to calculate hidden layer neurons
            for h in range(hval - 1):
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

            # ---- make adjustments to weights ----

            # update weights of hidden layers
            vih = vih + del_vih

            # update weights of input layer
            whj = np.add(whj, d_wh)

            # store new weight array
            whj_dict[e_cntr] = whj
            vih_dict[e_cntr] = vih

        # ------ test the current mlp against training and validation sets ------
        # test the current perceptron using the training data and display
        # the mean square error for the result of the predictions of the test

        ps = test_mlp(train_x, whj, vih)
        mse = mse_ann(ps, train_r.tolist())
        print('Train mse : ', mse)

        tr_mse[e_cntr] = mse

        # test the current mlp using the validation data and display
        # the mean square error for the result of the predictions of the test

        ps = test_mlp(val_set[0], whj, vih)
        mse = mse_ann(ps, val_set[1].tolist())
        print('Validation mse : ', mse)
        # check mean square error of the validation test, if is
        # better than the current stored best (lowest) mse store it and the input
        # and hidden layer weights that produced it
        if mse < best_mse:
            print('------------------------------------------------------->best mse is now: ', mse)
            print('------------------------------------------------------->best epoch is ', e_cntr)
            best_mse = mse
            best_whj = whj
            best_vih = vih
            best_epoch = e_cntr

        vl_mse[e_cntr] = mse

        # check the mse of validation test against the mse threshold (mse_limit)
        # and change in mse (del_mse) threshold if either is reached stop training (break while)
        if mse_limit is not None and mse_limit >= mse:
            print('val mse break: ', mse)
            break
        elif del_mse is not None and e_cntr > 1 and abs(old_mse - mse) <= del_mse:
            print('----------------------------------------------------change in mse break: ', np.around(old_mse - mse))
            break
        else:
            old_mse = mse

        # check for epoch limit
        if epoch is not None:
            e_cnt += 1
            if e_cnt == epoch:
                print('------------------------------------------------------broke due to epoch')
                break

    whj = best_whj
    vih = best_vih

    # print('Training Results:')
    ys, ps = test_mlp3(train_x, train_r, whj, vih)
    cm, p_a = confusion_matrix(ps, ys)
    # dis_conf_perorm(cm, p_a)

    # mse = mse_ann(ps, ys)
    # print('mse: ', mse)

    # print('Validation Results:')
    ys, ps = test_mlp3(val_set[0], val_set[1], whj, vih)
    cm, p_a2 = confusion_matrix(ps, ys)
    # dis_conf_perorm(cm, p_a)

    pa = list([p_a, p_a2])

    mse_ls = list([tr_mse, vl_mse])
    whj_vih = list([whj_dict, vih_dict])

    if verbose:
        print("Went through {:d} epochs ".format(e_cntr))

    return whj, vih, pa, mse_ls, whj_vih, best_epoch

# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------


def test_mlp3(d_set, d_rt, whj, vih):

    yi_l = {}
    hval = len(vih)

    rt_l, p_l = [], []
    for idx in range(len(d_set)):
        xt = d_set[idx]
        rt = d_rt[idx]

        rt_l.append(rt)

        zh = list([1])

        for h in range(hval - 1):
            zh.append(sigmoid(whj[h], xt))

        yi = list()
        yi.append(sigmoid(vih, zh))
        if yi[0] > 0:
            yi[0] = 1
        else:
            yi[0] = 0

        yi = np.array(yi, dtype=np.float64)

        p_l.append(yi[0])

        yi_l[idx] = yi[0]

    return rt_l, p_l


# produces a list of predictions using a set of
# input weights and hidden layer weights of a mlp
# and a given data set
def test_mlp(d_set, whj, vih):

    hval = len(vih)

    p_l = []

    for idx in range(len(d_set)):
        xt = d_set[idx]

        zh = list([1])
        for h in range(hval - 1):
            zh.append(sigmoid(whj[h], xt))

        yi = list()
        yi.append(sigmoid(vih, zh))
        if yi[0] > 0:
            yi[0] = 1
        else:
            yi[0] = 0

        p_l.append(yi[0])

    return p_l

# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------


def perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=None, h_val=5, runs=1, verbose=False,
                       mse_limit=None, del_mse=None):
    '''
    tr_x = trn_set[0]
    tr_r = trn_set[1]

    val_x = val_set[0]
    val_r = val_set[1]

    ts_x = tst_set[0]
    ts_r = tst_set[1]
    '''

    ds = len(trn_set[0])

    if epoch is None:
        epoch = 75
        print('epoch')
        print(epoch)

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

        whj, vih, pa, mse_ls, whj_vih, b_ep = mlp_trainer(trn_set, val_set, eta_a, h_val,
                                                          epoch=epoch, whj=whj, vih=vih, r_c=r_c, verbose=verbose,
                                                          mse_limit=mse_limit, del_mse=del_mse)

    mse_dict = mse_ls[0]
    mseval_dict = mse_ls[1]

    whj_dict = whj_vih[0]
    vih_dict = whj_vih[1]

    mse_list = [mse_dict, mseval_dict]
    mse_results = dis_training_results1(mse_list, runs, hval=h_val, eta=eta_a, keytype='Epoch')

    top_tup = mse_results[0]
    train_pts = mse_results[1]
    top_tupval = mse_results[2]
    val_pts = mse_results[3]

    # use the best mse value for the training mlp to model a given set of data and display
    # the confusion matrix and performance metrics
    print('Training Result:')
    best_epochtrn = display_trained_mlp(top_tupval, trn_set, whj_dict, vih_dict, title='Using MSE: best epoch {:d}',
                                        data_title='Mse for test data: ')
    print('validation result Result:')
    best_epochval = display_trained_mlp(top_tupval, val_set, whj_dict, vih_dict, title='Using MSE: best epoch {:d}',
                                        data_title='Mse for test data: ')

    print('Testing Result:')
    best_epochtst = display_trained_mlp(top_tupval, tst_set, whj_dict, vih_dict, title='Using MSE: best epoch {:d}',
                                        data_title='Mse for test data: ')

    y_a = list([train_pts, val_pts])
    title = 'Epoch vs. MSE: hval={:d}, eta={:f}'.format(h_val, eta_a)
    multi_y_plotter(np.linspace(1, epoch, len(train_pts)), y_a, title=title, leg_a=['Training', 'Validation', 'purple'], x_label='Epoch',
                    y_label='MSE', show_it=True)

    return

