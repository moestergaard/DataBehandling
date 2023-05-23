import matplotlib.pyplot as plt
import numpy as np

# Define the data points
points1 = [
    (6, 1, 'blue'), (12, 1, 'black'), (18, 1, 'red'), (24, 1, 'blue'),
    (6, 2, 'black'), (12, 2, 'black'), (18, 2, 'black'), (24, 2, 'blue'),
    (6, 3, 'black'), (12, 3, 'black'), (18, 3, 'black'), (24, 3, 'black')
]

points2 = [
    (126, 1, 'black'), (132, 1, 'red'), (138, 1, 'blue'), (144, 1, 'black'), 
    (126, 2, 'black'), (132, 2, 'black'), (138, 2, 'black'), (144, 2, 'black'), 
    (126, 3, 'black'), (132, 3, 'blue'), (138, 3, 'black'), (144, 3, 'black')
]

# Extract x, y, and color values from the points
x_values1 = [point[0] for point in points1]
y_values1 = [point[1] for point in points1]
colors1 = [point[2] for point in points1]

# Extract x, y, and color values from the points
x_values2 = [point[0] for point in points2]
y_values2 = [point[1] for point in points2]
colors2 = [point[2] for point in points2]


# Create two subplots with shared y-axis
f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')

# Set the y-axis labels
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['1', '2', '3'])

# Plot the points
for x, y, color in zip(x_values1, y_values1, colors1):
    ax.scatter(x, y, color=color)
    
for x, y, color in zip(x_values2, y_values2, colors2):
    ax2.scatter(x, y, color=color)

# Set the x-axis limits for each subplot
ax.set_xlim(5, 25)
ax2.set_xlim(125, 145)

# Set the x-axis labels
ax.set_xticks([6, 12, 18, 24])
ax.set_xticklabels(['6', '12', '18', '24'])
ax2.set_xticks([126, 132, 138, 144])
ax2.set_xticklabels(['126', '132', '138', '144'])

# Hide the spines between ax and ax2
ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.yaxis.tick_left()
ax2.yaxis.tick_right()

# Set plot title and labels
plt.suptitle('Wi-Fi Scanninger', fontsize=12)
# Add title for the x-axis in the middle
f.text(0.5, 0.01, 'Sekunder', ha='center', fontsize=10)
# plt.xlabel('Sekunder', fontsize=10, ha='center')
# Add title for the x-axis in the middle
# ax.set_xlabel('Sekunder', ha='center')
# ax.xaxis.set_label_coords(0.5, -0.2)
#set_title('Wi-Fi Scanninger')
# ax.set_xlabel('')
ax.set_ylabel('Antal på hinanden følgende\n detekteringer før skift')

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((1-d, 1+d), (-d, +d), **kwargs)
ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
ax2.plot((-d, +d), (-d, +d), **kwargs)

# Display the plot
plt.show()
