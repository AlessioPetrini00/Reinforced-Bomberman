import matplotlib.pyplot as plt
import os
import numpy as np

# Print current directory
print(os.getcwd())


with open('q_table_log.txt', 'r') as file:
    # Legge tutte le righe del file e le inserisce in un array
    values = file.readlines()

# Remove white spaces and new lines (\n)
array = [float(valore.strip()) for valore in values]
min_array = float(min(array))
max_array = float(max(array))

values_y = np.linspace(min_array, max_array, 10)
derivative = np.gradient(array, 100)

#plt.bar(range(len(array)), array)
plt.plot(derivative)
plt.ylim(-0.05, 0.05)
#plt.yticks(values_y)
plt.show()