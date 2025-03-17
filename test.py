import matplotlib.pyplot as plt

# Example history dictionary
history = {1: [2, 4], 2: [3, 8], 3: [4, 5]}
keys = history.keys()
v1 = [v[0] for v in history.values()]
v2 = [v[1] for v in history.values()]
# Transpose the history dictionary
plt.plot(keys, v1, color='red', label='dis_los',marker='x')
plt.plot(keys, v2, color='blue', label='gen_los', marker='^')

# Add labels and legend
plt.xlabel('Key')
plt.ylabel('Value')
plt.title('History Visualization')
plt.legend()

# Show plot
plt.show()
