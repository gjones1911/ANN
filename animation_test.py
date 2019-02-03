from ML_Visualizations import *
import numpy as np


x = np.array(list(np.arange(1, 12, .5)), dtype=np.float64)
y = x**2
y2 = x**3

x_a, y_a = list(), list()
x_a.append(x)
x_a.append(x)

y_a.append(y)
y_a.append(y2)
#print(x)
#print(y)

#print(np.linspace(1, 10, 10))

print(list(map(int,np.linspace(1, (58)+10, 10))))
#ani_generic_xy_plot(x,y,title='Test', x_ax='1-10', y_ax='x^2')
ani_multi_xy_plot(x_a,y_a,title='Test', x_ax='1-10', y_ax='x^2', ylim=y_a[1][len(y_a[1])-1])
