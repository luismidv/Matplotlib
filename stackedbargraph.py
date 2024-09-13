import numpy as np
import matplotlib.pyplot as plt

N = 5
menMeans =(20,35,30,35,27)
womenmeans = (25,34,34,20,25)
menstd = (2,3,4,1,2)
womenstd = (3,2,3,3)
ind = np.arange(N)



p2 = plt.bar(ind,womenmeans,width=0.35, bottom =menMeans,yerr=menstd)
p1 = plt.bar(ind,menMeans,width=0.35, yerr = menstd)

plt.ylabel('Scores')
plt.title('Scores by group and render')
plt.xticks(ind,('G1', 'G2','G3', 'G4','G5'))
plt.yticks(np.arange(0,91,10))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))
plt.show()