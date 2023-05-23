import matplotlib.pyplot as plt

# Define the data points
points = [
    (6, 1, 'blue'), (12, 1, 'black'), (18, 1, 'red'), (24, 1, 'blue'), (126, 1, 'black'), (132, 1, 'red'), (138, 1, 'blue'), (144, 1, 'black'), (246, 1, 'red'), (252, 1, 'blue'), (258, 1, 'black'), (264, 1, 'red'),
    (6, 2, 'black'), (12, 2, 'black'), (18, 2, 'black'), (24, 2, 'blue'), (126, 2, 'black'), (132, 2, 'black'), (138, 2, 'black'), (144, 2, 'black'), (246, 2, 'black'), (252, 2, 'black'), (258, 2, 'black'), (264, 2, 'black'),
    (6, 3, 'black'), (12, 3, 'black'), (18, 3, 'black'), (24, 3, 'black'), (126, 3, 'black'), (132, 3, 'blue'), (138, 3, 'black'), (144, 3, 'black'), (246, 3, 'black'), (252, 3, 'black'), (258, 3, 'black'), (264, 3, 'black')
]

# Extract x, y, and color values from the points
x_values = [point[0] for point in points]
y_values = [point[1] for point in points]
colors = [point[2] for point in points]

# Create the figure and axis objects
fig, ax = plt.subplots()

# Create five subplots with shared y-axis
f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True, facecolor='w', gridspec_kw={'width_ratios': [3, 1]})

# Set the y-axis labels
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['1', '2', '3'])

# Plot the points
for x, y, color in zip(x_values, y_values, colors):
    ax.scatter(x, y, color=color)

# Set plot title and labels
ax.set_title('Data Points')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')

# Manually set x-axis tick positions and limits
x_ticks = [6, 12, 18, 24, 126, 132, 138, 144, 246, 252, 258, 264]
plt.xticks(x_ticks)
# plt.xlim(x_ticks[0], x_ticks[-2])  # Exclude the last tick position from the limit

# Display the plot
plt.show()
