import json
import numpy as np
import matplotlib.pyplot as plt
with open('stats1', 'r') as file:
    dati1 = json.load(file)

coins_array = []
for round_key, round_data in dati1['by_round'].items():
    coins_array.append(round_data['coins'])

array1 = np.array(coins_array)

coins_array = []

with open('stats2', 'r') as file:
    dati2 = json.load(file)
for round_key, round_data in dati2['by_round'].items():
    coins_array.append(round_data['coins'])

array2 = np.array(coins_array)

plt.plot(array1, label='Dati 1', color='blue', marker='o')

plt.plot(array2, label='Dati 2', color='red', marker='x')

plt.xlabel('Asse X')
plt.ylabel('Asse Y')
plt.title('Grafico con Due Set di Dati')
plt.legend()

plt.show()