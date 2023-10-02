import matplotlib.pyplot as plt

import os

print(os.getcwd())


colonna1 = []
colonna2 = []

with open('error_log.txt', 'r') as file:
    for line in file:
        dati = line.strip().split(',')
        if len(dati) == 2:
            colonna1.append(float(dati[0]))
            colonna2.append(float(dati[1]))

plt.plot(colonna1, colonna2)
plt.show()