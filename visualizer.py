from annB import *
from ML_Visualizations import *

#eta = [0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.01, 0.006, 0.005, 0.004]
#epoch = [17, 16, 33, 9, 8, 34, 43, 18, 10, 20,36, 42]
#trnmse = [.112,.123, .124, .113, .122, .114, .121, .115, .129,  .119, .112, .128, .130 ]
#valmse = [.1,  .115, .098, .10,  .111, .111, .1,   .109, .113 , .117, .102, .122, .111 ]
'''
eta =    [0.2,  .17, .15,  .13,  .11,  .1 ,  .09,  .08,  .07, .06,  .05,  .04,   .03,   .01,  .006, .005, .004]
epoch =  [17,   41,  2,    5,    28,   42,   33,    9,   8,    34,   43,   18,   10,    20,    36,  42]
trnmse = [.112,.115, .127, .120, .128, .115, .124, .113, .122, .114, .121, .115, .129,  .119, .112, .128, .130 ]
valmse = [.1,  .1,   .117, .107, .107, .093, .098, .10,  .111, .111, .1,   .109, .113 , .117, .102, .122, .111 ]

y_a = list([trnmse, valmse])

title = 'Eta vs. MSE: hval={:d}, epoch={:d}'.format(60, 50)

multi_y_plotter(eta, y_a, title=title, leg_a=['Training', 'Validation', 'purple'], x_label='eta',
                y_label='MSE', show_it=True)
'''

'''
hval =   [5,   10,   30,   50,   60,   70,   80,   90,    100,  120,  150,  170,  190,  200,  230]
trnmse = [.102,  .126, .105, .116, .118, .099, .109, .112,  .104, .094, .090, .105, .101, .099, .098]
valmse = [.122,  .111, .098, .102, .1,   .091, .096, .104,  .096, .087, .091, .091, .087, .080, .091]
tstmse = [.122,  .135, .120, .128, .1,   .123, .111, .133,  .096, .113, .107, .096, .089, .115, .135]
trmsbs = [.102,  .125, .105, .105, .103, .096, .101, .110,  .095, .087, .087, .085, .073, .081, .078]

y_a = list([trnmse, valmse, tstmse, trmsbs])

title = 'Hval vs. MSE: eta={:f}, epoch={:d}'.format(.1, 200)

multi_y_plotter(hval, y_a, title=title, leg_a=['Training', 'Validation', 'Test', 'Best Train'], x_label='hval',
                y_label='MSE', show_it=True)
'''


trnmse = [0.163, .122, .108, .133, .120, .1089, .109, .102, ]
valmse = []

y_a = list([trnmse, valmse])

title = 'Hval vs. MSE: eta={:f}, epoch={:d}'.format(.1, 200)

multi_y_plotter(hval, y_a, title=title, leg_a=['Training', 'Validation', 'Test', 'Best Train'], x_label='hval',
                y_label='MSE', show_it=True)