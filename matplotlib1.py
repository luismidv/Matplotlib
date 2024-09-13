import matplotlib.pyplot as plt
import numpy as np

def first():
    """FIRST EXAMPLE
    """
    x = np.arange(1,10)
    y = np.arange(1,10)
    draw(x,y)

def draw(x,y):
    """SHOWING THE REPRESENTATION

    Args:
        a (_type_): 1st array
        b (_type_): 2nd array
    """
    plt.plot(x,y,color= 'red')
    plt.show()

def draw_bars(x,y):
    plt.xlabel('Coordenada x')
    plt.ylabel('Coordenada x')
    plt.xticks((-300,-200,-100,0,50))
    plt.hist(x)
    plt.show()

def style(a,b):
    """SET STYLE FOR THE REPRESENTATIONS

    Args:
        a (_type_): 1st array
        b (_type_): 2nd array
    """
    plt.title("Triangle")
    plt.xlabel("Array A")
    plt.ylabel("Array B")
    plt.axis([-50,80,2,8])
    plt.xticks((-40,-20,0,20,40,60,80),('L40','u20','i0','s20','m40','i60','x80'))
    draw_bars(a,b)

def second():
    """SECOND EXAMPLE
    """
    a = [0,-100,25,67,-323]
    b = [0,3,7,3,9]
    #style(a,b)
    draw_bars(a,b)

#first()
second()



