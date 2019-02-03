from annB import *
from DataProcessor import *
from ML_Visualizations import ani_generic_xy_plot

email_data = load_data_file('spambase.dt')

trn_set, val_set, tst_set, ksds = generate_data_sets_xy(email_data, verbose=False, rand=True, seed=True, normalize='z')



# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=200, h_val=10, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=200, h_val=30, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=200, h_val=30, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=200, h_val=40, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)


#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=200, h_val=50, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=200, h_val=60, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)


#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=200, h_val=70, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=200, h_val=80, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=200, h_val=90, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=200, h_val=100, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=200, h_val=120, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=200, h_val=150, runs=1, verbose=True, mse_limit=None,
#                   del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=200, h_val=170, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=200, h_val=190, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=200, h_val=200, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.1, epoch=190, h_val=5, runs=1, verbose=True, mse_limit=.085,
#                    del_mse=None)
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.2, epoch=50, h_val=60, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.2, epoch=50, h_val=70, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.2, epoch=50, h_val=80, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.2, epoch=50, h_val=90, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.2, epoch=50, h_val=100, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.09, epoch=50, h_val=60, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.09, epoch=50, h_val=70, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.09, epoch=50, h_val=80, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.09, epoch=50, h_val=90, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.09, epoch=50, h_val=100, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.06, epoch=50, h_val=60, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.06, epoch=50, h_val=70, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.06, epoch=50, h_val=80, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.06, epoch=50, h_val=90, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)

#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.06, epoch=50, h_val=100, runs=1, verbose=True, mse_limit=None,
#                    del_mse=None)


#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.004, epoch=190, h_val=230, runs=1, verbose=True, mse_limit=.085,
#                    del_mse=None)


#perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.004, epoch=125, h_val=250, runs=1, verbose=True, mse_limit=.085,
#                    del_mse=None)

perform_epoch_test2(trn_set, val_set, tst_set, eta_a=.2, epoch=300, h_val=190, runs=1, verbose=True, mse_limit=None,
                    del_mse=None)


