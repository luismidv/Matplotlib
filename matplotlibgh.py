import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def sin_cos_draw():
    """DRAWING SINE AND COSINe
    """
    x1 = np.linspace(0,10,100)
    print(x1)
    fig = plt.figure()

    plt.plot(x1, np.sin(x1),'-')
    plt.plot(x1,np.cos(x1),'--')



    plt.show()

def subplot_try():
    """UNDERSTANDING SUBPLOT
    """
    x1 = np.linspace(0,10,100)
    plt.subplot(2,1,1)
    plt.plot(x1, np.sin(x1))
    plt.subplot(2,1,2)
    plt.plot(x1, np.cos(x1), '--')
    plt.show()

def figure_info():
    """GETTINF FIGURE INFO
    """
    print(plt.gca())
    print(plt.gcf())

def single_list_plot():
    """PLOTTING WITH A SINGLE LIST, USES THAT LIST AS VALUES FOR AXIS Y
    """
    list = [1,2,3,4]
   
    plt.plot(list)
    plt.show()

def state_machine_test():
    """PLOT TESTING LABELS AND LEGENDS
    """
    x = np.linspace(0,2,100)
    print(x)
    plt.plot(x,x, label = 'linear')
    plt.plot(x,x**2, label='quadratic1')
    plt.plot(x,x**3, label = 'cubic')
    plt.xlabel('x label')
    plt.ylabel('x ** number label')
    plt.legend()
    plt.show()

def plot_red_circles():
    """USING RED DOTS/CIRCLES 
    """
    plt.plot([1,2,3,4], [2,4,9,16], 'ro')
    plt.show()

def work_numpy_arrays():
    """WORKING WITH NUMPY ARRAYS AND DIFFERENT DRAWINGS
    """
    array = np.arange(0.,5.,0.2)
    plt.plot(array, array,'r--', array,array**2,'bs',array,array**3,'g^')
    plt.show()

def fig_ax_subplots():
    x1 = np.arange(0,5,1)
    fig,ax = plt.subplots(2)
    ax[0].plot(x1,np.sin(x1), 'r-')
    ax[1].plot(x1,np.cos(x1), 'b-')
    plt.show()

def objects():
    """ADD AXES IS USED WHEN POSITION OF THE FIGURE IS IMPORTANT IN OTHER CASES WE CAN USE SUBPLOT"""
    fig = plt.figure()

    x2 = np.linspace(0,5,10)
    y2 = x2**2    

    axes = fig.add_axes([0.1,0.1,0.8,0.8])
    axes.plot(x2,y2,'r')

    axes.set_xlabel('x2')
    axes.set_ylabel('y2')
    axes.set_title('title')
    plt.show()

def figure_axes():
    """CREATION AND MANAGING FIGURE AND AXES"""

    fig = plt.figure()
    ax = plt.axes()

    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)
    plt.show()

def concise_subplots():
    fig, axes = plt.subplots()
    axes = fig.add_subplot(1,1,1)
    plt.show()

def concise_multiple_subplots():
    """SUBPLTOS WITH ROWS AND COLUMNS DISTRIBUTIONS ("GRID")"""
    fig,axes = plt.subplots(nrows = 2, ncols =2)
    fig,axes = plt.subplots(2,1)
    plt.show()

def plot_with_without_x():
    """EXAMPLE OF PLOT PARAMETERS"""
    plt.plot([1,2,3,4],'r-')
    plt.plot([1,2,3,4],[1,2,3,4],'b-')
    plt.show()

def plot_with_without_x2():
    """MORE EXAMPLES"""
    x = np.arange(0,5,0.5)
    plt.plot(x,x**2,'r-')
    plt.show()

def multiline_plots():
    """MORE EXAMPLES"""
    x4 = np.arange(1,5,1)
    fig = plt.figure()
    plt.plot(x4, x4*1.5,'r-')
    plt.plot(x4, x4*3,'b-')
    plt.plot(x4, x4/3,'g-')
    plt.show()
    fig.savefig('fig.png')

def using_scatter():
    """EXAMPLE USING SCATTERS,OK ALSO DRAW POINTS SO WE DONT NEED THE PLOT BEFORE"""

    x7 = np.linspace(0,10,30)
    y7 = np.sin(x7)

    print(x7)
    plt.plot(x7,np.sin(x7),'o')
    plt.plot(x7,np.sin(x7),'-ok')
    plt.show()

def scatter_plot():
    """THIS IS A LESS EFFICIENT WAY OS USING SCATTERS CAUSE IT CREATES POINTS INDIVIDUALLY, USING 'O' WITH PLOT, EACH POINTS IS A CLONE OF OTHER SO THE 
       CREATING WORK IS DONE"""
    x7 = np.linspace(0,10,30)
    y7 = np.sin(x7)

    plt.scatter(x7,y7)
    plt.show()

def histograms():
    data1 = np.random.randn(1000)
    print(len(data1))
    plt.hist(data1)
    plt.show()

def histograms_parameters():
    """DENSITY PARAMETER IS INTERESTING WHEN YOU HAVE TO INTERPRET IN TERMS OF DENSITY OF PROBABILITY INSTEAD OF RAW COUNTS  ."""
    data1 = np.random.randn(1000)
    plt.hist(data1, bins = 30,density = True,alpha = 0.5,histtype='stepfilled', color = 'steelblue')
    plt.show()

def histogram_distributions():
    """USING KWARGS VARIABLES TO DEFINE PARAMETERS, YOU CAN CHANGE COLORS AS YOU WISH"""
    x1 = np.random.normal(0,4,1000)
    x2 = np.random.normal(-2,2,1000)
    x3 = np.random.normal(1,5,1000)

    kwargs = dict(histtype ='stepfilled', alpha = 0.5, density = True, bins = 40)

    plt.hist(x1, **kwargs)
    plt.hist(x2, **kwargs)
    plt.hist(x3, **kwargs)
    
    plt.show()

def histograms_oo_api():
    data1 = np.random.normal(1,10,10)
    print(data1)
    print(len(data1))
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    ax1 = fig.add_subplot(2,1,2)
    ax.hist(data1, bins = 10)
    plt.show()

def two_dimensiom_histogram():
    """USING TWO DIMENSION HISTOGRAM, MULTIVARIATE NORMAL IS A MATH MODEL USED IN MACHINE LEARNING """
    mean = [0,0]
    cov = [[1,1,],[1,2]]
    x8,y8 = np.random.multivariate_normal(mean, cov, 10000).T

    plt.hist2d(x8,y8,bins = 30, cmap = "Blues")

    cb = plt.colorbar()

    cb.set_label('counts in bin')
    plt.show()

def bar_charts():
    """RE"""
    data2 = [5. ,25. ,50. ,20.]
    plt.bar(range(len(data2)),data2)
    plt.show()

def thickness_bar_charts():
    """CHANGING BAR WIDTH"""
    data2 = [5. ,25. ,50. ,20.]
    plt.bar(range(len(data2)), data2, width=0.8)
    plt.show()

def horizontal_bar_charts():
    """DRAW HORIZONTAL BAR"""
    data2 = [5. ,25. ,50. , 20.]
    plt.barh(range(len(data2)),data2)
    plt.show()

def error_ybar_chart():
    """ERROR BAR"""
    x9 = np.arange(0,4,0.2)
    y9 = np.exp(-x9)
    print(y9)
    e1 = 0.1*np.abs(np.random.rand(len(y9)))
    plt.errorbar(x9,y9, yerr = e1,fmt = '.-')
    plt.show()

def error_Xbar_chart():
    """ERROR BAR"""
    x9 = np.arange(0,4,0.2)
    y9 = np.exp(-x9)
    print(y9)
    e1 = 0.1*np.abs(np.random.randn(len(y9)))
    e2 = 0.1*np.abs(np.random.randn(len(y9)))
    plt.errorbar(x9,y9, yerr = e1, xerr=e2, fmt = '.-')
    plt.show()

def error_asymmetrical_bar():
    """SYMMETRICAL ERROR BARS USE POSITIVE AND NEGATIVE ERROR WITH SAME VALUE
       ASYMMETRICAL BARS LET YOU USE ERROR WITH DIFFERENT VALUES ON POSITIVE AND NEGATIVE"""

    plt.errorbar
    x9 = np.arange(0,4,0.2)
    y9 = np.exp(-x9)
    e1 = 0.1*np.abs(np.random.randn(len(y9)))
    e2 = 0.1*np.abs(np.random.randn(len(y9)))
    plt.show()

def multiple_bar_chart():
    """PRINTING MULTIPLE BARS"""
    data3 = [[15.,25.,40.,30.],
             [11.,23.,51.,17.],
             [16.,22.,52.,19.]]
    
    z1 = np.arange(4)

    plt.bar(z1 + 0.00, data3[0], color = 'r', width = 0.25)
    plt.bar(z1 + 0.25, data3[1], color = 'b', width = 0.25)
    plt.bar(z1 + 0.50, data3[2], color = 'g', width = 0.25)
    plt.show()

def multiple_bar_bottom():
    """PRINTING BAR ON TOP OF ANOTHER"""

    data2 = [15., 30., 45., 22.]
    data3 = [15., 25., 50., 20.]

    z = np.arange(4)
    plt.bar(z , data2, color = 'r', width = 0.25)
    plt.bar(z ,   data3,   color = 'b',   width=0.25, bottom=data2)
    plt.show()

def back_to_back():
    """BAR GROWING ON OPPOSITE DIRECTIONS"""

    u1 = np.array([15.,35.,45.,32.])
    u2 = np.array([12.,30.,50.,25.])

    z1 = np.arange(4)

    plt.barh(z1, u1, color = 'r')
    plt.barh(z1, -u2, color = 'b')
    plt.show()

def pie_charts():
    """PRINTING PIE CHARTS WITH MATPLOTLIB"""
    plt.figure(figsize=(7,7))
    x10 = [35,25,20,20]
    labels = ['Computer', 'Electronics', 'Mechanical', 'Chemical']
    plt.pie(x10, labels = labels)
    plt.show()

def exploded_pie_charts():
    """PRINTING SEPARATED PIE CHARTS WITH MATPLOTLIB"""
    plt.figure(figsize=(7,7))
    x11 = [30,25,20,15,10]
    labels = [0.2,0.1,0.1,0.05,0]
    explode = [0.2,0.1,0.1,0.05,0]
    plt.pie(x11, labels = labels, explode = explode, autopct='%1.1f%%')
    plt.show()    

def boxplot():
    """PRINTING BOXLPOT WITH MATPLOTLIB"""
    data3 = np.random.randn(100)
    plt.boxplot(data3)
    plt.show()

def customized_boxplot():
    """PRINTING CUSTOMIZED BOXPLOT"""
    data4 = np.random.randn(100,5)
    plt.boxplot(data4)
    plt.show()

def area_chart():
    x12 = range(1,6)
    y12 = [1,4,6,8,4]

    plt.fill_between(x12,y12)
    plt.show()

def countour_Plot():
    """IS USED TO DISPLAY 3 DIMENSIONAL DATA IN 2 DIMENSIONS"""
    matrix1 = np.random.rand(10,20)
    cp = plt.contour(matrix1)
    plt.show()

def color_bar():
    matrix1 = np.random.rand(10,20)
    csf = plt.contourf(matrix1)
    plt.show()








"""single_list_plot()
state_machine_test()
plot_red_circles()
work_numpy_arrays()"""
"""fig_ax_subplots()
objects()
figure_axes()
concise_subplots()
concise_multiple_subplots()
plot_with_without_x()
plot_with_without_x2()
multiline_plots()
using_scatter()
scatter_plot()
histograms()
histograms_parameters()
histogram_distributions()
histograms_oo_api()
two_dimensiom_histogram()
bar_charts()
thickness_bar_charts()
error_ybar_chart()
error_Xbar_chart()
multiple_bar_chart()
multiple_bar_bottom()
back_to_back()
pie_charts()
exploded_pie_charts()
boxplot()
customized_boxplot()
area_chart()
countour_Plot()"""
color_bar()
