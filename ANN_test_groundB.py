from ANN import *
from ML_Visualizations import ani_generic_xy_plot

email_data = load_data_file('spambase.dt')

trn_set, val_set, tst_set, ksds = generate_data_sets_xy(email_data, verbose=False, rand=True, seed=True, normalize='z')

print(trn_set[1])

# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# eta testing
#perform_eta_test(trn_set, val_set, tst_set, eta_a=None, epochl=30, h_val=190, runs=1, verbose=True)




#perform_eta_test(trn_set, val_set, tst_set, eta_a=None, epochl=230, h_val=230, runs=1, verbose=True)
#perform_eta_test(trn_set, val_set, tst_set, eta_a=None, epochl=1, h_val=226, runs=1, verbose=True, mse_limit=.05, del_mse=.001)

#perform_eta_test(trn_set, val_set, tst_set, eta_a=None, epochl=50, h_val=60, runs=1, verbose=True, mse_limit=.05, del_mse=None)

#perform_eta_test(trn_set, val_set, tst_set, eta_a=None, epochl=600, h_val=230, runs=1, verbose=True)
#perform_eta_test(trn_set, val_set, tst_set, eta_a=None, epochl=300, h_val=130, runs=1, verbose=True)


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# epoch testing
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.1, epochl=None, h_val=5, runs=5)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.1, epochl=None, h_val=230, runs=1, verbose=True)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.068966, epochl=None, h_val=232, runs=1, verbose=True)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.068966, epochl=None, h_val=230, runs=1, verbose=True)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.068966, epochl=None, h_val=230, runs=1, verbose=True)

#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.006449,epochl=None, h_val=230, runs=1, verbose=True)

#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.006449, epochl=None, h_val=230, runs=1, verbose=True)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.006449, epochl=None, h_val=230, runs=1, verbose=True)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.006449, epochl=None, h_val=300, runs=1, verbose=True, mse_limit=.05, del_mse=.02)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.006449, epochl=None, h_val=300, runs=1, verbose=True, mse_limit=.05, del_mse=.02)

#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.1, epochl=[100], h_val=60, runs=1, verbose=True, mse_limit=.05, del_mse=.001)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.1, epochl=[100], h_val=60, runs=1, verbose=True, mse_limit=None, del_mse=None)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.1, epochl=None, h_val=60, runs=1, verbose=True, mse_limit=None, del_mse=None)


# 3333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.2, epoch=150, h_val=150, runs=1, verbose=True, mse_limit=None, del_mse=None)
#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=200, h_val=60, runs=1, verbose=True, mse_limit=None, del_mse=None)
#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=30, h_val=130, runs=1, verbose=True, mse_limit=None, del_mse=None)
#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.006449, epoch=30, h_val=60, runs=1, verbose=True, mse_limit=None, del_mse=None)
#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.005122, epoch=30, h_val=60, runs=1, verbose=True, mse_limit=None, del_mse=None)
#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.004611, epoch=500, h_val=60, runs=1, verbose=True, mse_limit=None, del_mse=None)
#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=50, h_val=60, runs=1, verbose=True, mse_limit=None, del_mse=None)
#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=40, h_val=120, runs=1, verbose=True, mse_limit=None, del_mse=None)
#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=500, h_val=60, runs=1, verbose=True, mse_limit=None, del_mse=None)
# 3333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333


#perform_epoch_test(trn_set, val_set, tst_set, eta_a=., epochl=None, h_val=230, runs=1, verbose=True, mse_limit=.05, del_mse=.001)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=., epochl=None, h_val=230, runs=1, verbose=True, mse_limit=.05, del_mse=.001)


#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.09, epochl=None, h_val=100, runs=1, verbose=True, mse_limit=.05)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.09, epochl=None, h_val=200, runs=1, verbose=True, mse_limit=.05)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.09, epochl=None, h_val=250, runs=1, verbose=True, mse_limit=.05)

#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.2, epochl=None, h_val=70, runs=1, verbose=True, mse_limit=.05)

#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.1, epochl=None, h_val=50, runs=1, verbose=True, mse_limit=.05, del_mse=.0001)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.1, epochl=None, h_val=70, runs=1, verbose=True, mse_limit=.05)

#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.1, epochl=None, h_val=100, runs=1, verbose=True, mse_limit=.05)

#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.1, epochl=None, h_val=150, runs=1, verbose=True, mse_limit=.05, del_mse=.0001)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.006449, epochl=None, h_val=230, runs=1, verbose=True, mse_limit=.05, del_mse=.01)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.006449, epochl=None, h_val=230, runs=1, verbose=True, mse_limit=.05)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.1, epochl=None, h_val=5, runs=1, verbose=True, mse_limit=.05)

#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.006449, epochl=None, h_val=250, runs=1, verbose=True, mse_limit=.05)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.006449,epochl=None, h_val=230, runs=1, verbose=True, del_mse=.0001)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.006449, epochl=None, h_val=200, runs=1, verbose=True, mse_limit=.05, del_mse=.0001)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.006449, epochl=None, h_val=200, runs=1, verbose=True, mse_limit=.05, del_mse=.0001)
#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.006449, epochl=None, h_val=150, runs=1, verbose=True, mse_limit=.05)

#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.006449, epochl=None, h_val=200, runs=1, verbose=True, mse_limit=.05)

#perform_epoch_test(trn_set, val_set, tst_set, eta_a=.006449, epochl=None, h_val=70, runs=1, verbose=True, mse_limit=.05)


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# hval testing
#perform_hval_test(trn_set, val_set, tst_set, eta_a=.005564 , epochl=130, h_val=None, runs=1, verbose=True)
#perform_hval_test(trn_set, val_set, tst_set, eta_a=.068966 , epochl=130, h_val=None, runs=1, verbose=True)
#perform_hval_test(trn_set, val_set, tst_set, eta_a=.068966 , epochl=230, h_val=None, runs=1, verbose=True)
#perform_hval_test(trn_set, val_set, tst_set, eta_a=.09, epochl=130, h_val=None, runs=1, verbose=True)

# --------------------------------------------------------------------------------------------------------------------------------------
#perform_hval_test(trn_set, val_set, tst_set, eta_a=.006449, epochl=1, h_val=None, runs=1, verbose=True, mse_limit=.05, del_mse=.001)

#perform_hval_test(trn_set, val_set, tst_set, eta_a=.005122, epochl=1, h_val=None, runs=1, verbose=True, mse_limit=.05, del_mse=.001)

#perform_hval_test(trn_set, val_set, tst_set, eta_a=.004611, epochl=1, h_val=None, runs=1, verbose=True, mse_limit=.05, del_mse=.001)
# --------------------------------------------------------------------------------------------------------------------------------------


#perform_hval_test(trn_set, val_set, tst_set, eta_a=.006449, epochl=1, h_val=None, runs=1, verbose=True, mse_limit=.05, del_mse=.001)

#perform_hval_test(trn_set, val_set, tst_set, eta_a=.1, epochl=180, h_val=None, runs=1, verbose=True, mse_limit=.096, del_mse=.001)
#perform_hval_test(trn_set, val_set, tst_set, eta_a=.004611, epochl=70, h_val=None, runs=1, verbose=True, mse_limit=.05, del_mse=.001)
#perform_hval_test(trn_set, val_set, tst_set, eta_a=.004611, epochl=75, h_val=None, runs=1, verbose=True, mse_limit=.05, del_mse=.001)
#perform_hval_test(trn_set, val_set, tst_set, eta_a=.05, epochl=75, h_val=None, runs=1, verbose=True, mse_limit=.05, del_mse=.001)
#perform_hval_test(trn_set, val_set, tst_set, eta_a=.2, epochl=500, h_val=[250], runs=1, verbose=True, mse_limit=.05, del_mse=.001)

#perform_hval_test(trn_set, val_set, tst_set, eta_a=.1, epochl=30, h_val=[250], runs=1, verbose=True, mse_limit=.05, del_mse=.001)

#perform_hval_test(trn_set, val_set, tst_set, eta_a=.1, epochl=40, h_val=None, runs=1, verbose=True, mse_limit=.05, del_mse=.001)
#perform_hval_test(trn_set, val_set, tst_set, eta_a=.1, epochl=40, h_val=None, runs=1, verbose=True, mse_limit=None, del_mse=None)

#perform_hval_test(trn_set, val_set, tst_set, eta_a=.09, epochl=1, h_val=None, runs=1, verbose=True, mse_limit=.05, del_mse=.001)
#perform_hval_test(trn_set, val_set, tst_set, eta_a=.09, epochl=100, h_val=None, runs=1, verbose=True, mse_limit=.05, del_mse=.001)


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------



