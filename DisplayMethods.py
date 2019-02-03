import numpy as np


def print_array(array):
    cnt = 0
    for row in array:
        print(cnt, ': ', row)
        cnt += 1
    return


def array_dim(array):
    print('number of observations: ', len(array))
    print('number of attributes: ', len(array[0]))

    return len(array), len(array[0])


def process_mlp_testing_results(test_dict, title='title', key_type='key', val_type='average MSE', verbose=False,
                                reverse=False):

    train_pts = []

    if verbose:
        print(title)

    for ep in test_dict:
        #if verbose:
        #    print(key_type + ': ', ep, val_type + ': ', test_dict[ep])
        train_pts.append(test_dict[ep])

    # train_pts = sorted(train_pts)

    mse_sorted = sorted(test_dict.items(), key=lambda kv: kv[1], reverse=reverse)
    # grab first tuple from sorted dictionary
    top_tup = mse_sorted[0]

    if verbose:
        if key_type == 'Eta':
            print("Best {:s} is {:f} with {:f}".format(key_type, top_tup[0], top_tup[1]))
        else:
            print("Best {:s} is {:d} with {:f}".format(key_type, top_tup[0], top_tup[1]))
        print('-------------------------------------------------')
        print('-------------------------------------------------')

    return top_tup, train_pts


def display_confusion_matrix(tn,tp, fn, fp, class_names=None):

    if class_names is None:
        class_names = list(['benign', 'malignant'])

    print('           Confusion Matrix:')
    print(' ___________________________________')
    print('|              |   Predicted        |')
    print('|  True Class  |____________________|')
    print('|______________|_{:_>s}_|_{:_>s}_|'.format(class_names[0], class_names[1]))
    print('|____{:_>s}____|__{:_>4d}__|___{:_>4d}____|'.format(class_names[0], int(tn), int(fp)))
    print('|_{:_>s}____|__{:_>4d}__|___{:_>4d}____|'.format(class_names[1], int(fn), int(tp)))
    print('')
    return


def display_performance(p_array, k_val=2):
    print('')
    #print('Averages for k value of {:d}'.format(int(k_val)))
    print('Accuracy : {:f}'.format(p_array[0]))
    print('Sensitivity : {:f}'.format(float(p_array[1])))
    print('Precision : {:f}'.format(float(p_array[2])))
    print('True Negative Rate : {:f}'.format(float(p_array[3])))
    print('F1 score : {:f}'.format(float(p_array[4])))
    print('')
    return


def dis_conf_perform(cm, p_array, cl_names=None):
    display_confusion_matrix(cm[0][0], cm[1][1], cm[1][0], cm[0][1], class_names=cl_names)
    display_performance(p_array)
    return


def dis_training_results1(mse_dicts, runs, epoch=None, eta=None, hval=None, title1='Training Results',
                         title2='Validation Result', keytype='h value', val_type1='Average MSE',
                         verbose1=True, reverse1=False):
    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('runs: ', runs)
    if hval is not None:
        print('hval: ',hval)
    if epoch is not None:
        print('epoch: ', epoch)
    if eta is not None:
        print('eta: ', eta)
    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('---------------------------------    Mse result    ---------------------------------------------------')
    top_tup, train_pts = process_mlp_testing_results(mse_dicts[0], title=title1, key_type=keytype,
                                                     val_type=val_type1, verbose=verbose1, reverse=reverse1)

    top_tupval, val_pts = process_mlp_testing_results(mse_dicts[1], title=title2, key_type=keytype,
                                                      val_type=val_type1, verbose=verbose1, reverse=reverse1)

    mse_results = [top_tup, train_pts, top_tupval, val_pts]

    return mse_results


def dis_training_results(mse_dicts, performance_dicts, runs, epoch=None, eta=None, hval=None, title1='Training Results',
                         title2='Validation Result', keytype='h value', val_type1='Average MSE',
                         val_type2='Performance', verbose1=True, verbose2=True, reverse1=False, reverse2=True):
    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('runs: ', runs)
    if hval is not None:
        print('hval: ',hval)
    if epoch is not None:
        print('epoch: ', epoch)
    if eta is not None:
        print('eta: ', eta)
    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('---------------------------------    Mse result    ---------------------------------------------------')
    top_tup, train_pts = process_mlp_testing_results(mse_dicts[0], title=title1, key_type=keytype,
                                                     val_type=val_type1, verbose=verbose1, reverse=reverse1)

    top_tupval, val_pts = process_mlp_testing_results(mse_dicts[1], title=title2, key_type=keytype,
                                                      val_type=val_type1, verbose=verbose1, reverse=reverse1)

    mse_results = [top_tup, train_pts, top_tupval, val_pts]

    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('------------------------------------------------------------------------------------------------------')
    print('---------------------------------    Performance result    -------------------------------------------')
    top_tup1, train_pts1 = process_mlp_testing_results(performance_dicts[0], title=title1, key_type=keytype,
                                                       val_type=val_type2, verbose=verbose2, reverse=reverse2)

    top_tupval1, val_pts1 = process_mlp_testing_results(performance_dicts[1], title=title2, key_type=keytype,
                                                        val_type=val_type2, verbose=verbose2, reverse=reverse2)

    performance_results = [top_tup1, train_pts1, top_tupval1, val_pts1]

    return mse_results, performance_results


# attempts to predict classes of a set of data using a given mlp(whj, vih)
# then displays the confusion matrix and performance metrics for the modeling
def display_trained_mlp(top_tupval, test_set, whj_dict, vih_dict, title='Using MSE: best hval {:d}',
                        data_title='Mse for test data: '):
    from ANN import test_mlp3
    from ANN import mse_ann
    from performance_tests import confusion_matrix

    # grab top key value
    best_hval = top_tupval[0]
    print(title.format(best_hval))

    # grab the whj, vih that correspond to best key value
    best_whj = whj_dict[best_hval]
    best_vih = vih_dict[best_hval]
    ps = test_mlp3(test_set[0], best_whj, best_vih)
    msetst = mse_ann(ps, test_set[1])
    # print('Mse for test data: ', msetst)
    print(data_title, msetst)
    print('--------------------------------')
    print('--------------------------------')
    cm, p_a = confusion_matrix(ps, test_set[1])
    dis_conf_perform(cm, p_a)
    print('-------------------------------')
    print('-------------------------------')

    return best_hval, best_whj, best_vih
