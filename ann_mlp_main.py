from ANN import *

email_data = load_data_file('spambase.dt')


b_epoch, b_whj, b_vih = ann_mlp(email_data, eta=.2, hval=190, epoch=110, mse_limit=.09, del_mse=None, trn_pct=.8,
                                val_pct=.1, tst_pct=.1, verbose=False, ret_epoch=True)

