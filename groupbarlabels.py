import numpy as np
import matplotlib.pyplot as plt

labels = ['G1', 'G2', 'G3', 'G4 ', 'G5']
menmeans =(20,34,30,35,27)
womenmeans = (25,32,34,20,25)

x=np.arange(len(labels))
width = 0.35

fig,ax = plt.subplots()
rects1 = ax.bar(x-width/2, menmeans, width, label = 'Men')
rects2 = ax.bar(x+width/2, womenmeans, width, label='Women')

ax.set_ylabel('Scores')
ax.set_title =('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
        xy = (rect.get_x() + rect.get_width() /2, height),
        xytext = (0,3),
        textcoords = "offset points",
        ha ='center', va ='bottom')

    

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()
