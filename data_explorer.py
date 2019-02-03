from ANN import *
from ML_Visualizations import ani_generic_xy_plot

email_data = load_data_file('spambase.dt')

rows, cols = email_data.shape

print('rows: ', rows, "cols: ", cols)
