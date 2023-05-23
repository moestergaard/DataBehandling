import matplotlib.pyplot as plt
import numpy as np

# Define the data points
points1 = [
    (6, 1, 'blue'), (12, 1, 'blue'), (18, 1, 'red'), (24, 1, 'blue'),
    (6, 2, 'black'), (12, 2, 'black'), (18, 2, 'black'), (24, 2, 'blue'),
    (6, 3, 'black'), (12, 3, 'black'), (18, 3, 'black'), (24, 3, 'black')
]

points2 = [
    (126, 1, 'blue'), (132, 1, 'red'), (138, 1, 'blue'), (144, 1, 'blue'), 
    (126, 2, 'blue'), (132, 2, 'blue'), (138, 2, 'blue'), (144, 2, 'blue'), 
    (126, 3, 'black'), (132, 3, 'blue'), (138, 3, 'blue'), (144, 3, 'blue')
]

points3 = [
    (246, 1, 'red'), (252, 1, 'blue'), (258, 1, 'blue'), (264, 1, 'red'), 
    (246, 2, 'blue'), (252, 2, 'blue'), (258, 2, 'blue'), (264, 2, 'blue'), 
    (246, 3, 'blue'), (252, 3, 'blue'), (258, 3, 'blue'), (264, 3, 'blue')
]

points4 = [
    (366, 1, 'blue'), (372, 1, 'blue'), (378, 1, 'red'), (384, 1, 'blue'), 
    (366, 2, 'blue'), (372, 2, 'blue'), (378, 2, 'blue'), (384, 2, 'blue'), 
    (366, 3, 'blue'), (372, 3, 'blue'), (378, 3, 'blue'), (384, 3, 'blue')
]

points5 = [
    (486, 1, 'blue'), (492, 1, 'red'), (498, 1, 'blue'), (504, 1, 'blue'), 
    (486, 2, 'blue'), (492, 2, 'blue'), (498, 2, 'blue'), (504, 2, 'blue'), 
    (486, 3, 'blue'), (492, 3, 'blue'), (498, 3, 'blue'), (504, 3, 'blue')
]

points6 = [
    (606, 1, 'red'), (612, 1, 'blue'), (618, 1, 'blue'), (624, 1, 'red'), 
    (606, 2, 'blue'), (612, 2, 'blue'), (618, 2, 'red'), (624, 2, 'red'), 
    (606, 3, 'blue'), (612, 3, 'blue'), (618, 3, 'blue'), (624, 3, 'blue')
]

points7 = [
    (726, 1, 'blue'), (732, 1, 'blue'), (738, 1, 'red'), (744, 1, 'blue'), 
    (726, 2, 'red'), (732, 2, 'blue'), (738, 2, 'blue'), (744, 2, 'blue'), 
    (726, 3, 'blue'), (732, 3, 'blue'), (738, 3, 'blue'), (744, 3, 'blue')
]

# Extract x, y, and color values from the points

# Points 1
x_values1 = [point[0] for point in points1]
y_values1 = [point[1] for point in points1]
colors1 = [point[2] for point in points1]

# Points 2
x_values2 = [point[0] for point in points2]
y_values2 = [point[1] for point in points2]
colors2 = [point[2] for point in points2]

# Points 3
x_values3 = [point[0] for point in points3]
y_values3 = [point[1] for point in points3]
colors3 = [point[2] for point in points3]

# Points 4
x_values4 = [point[0] for point in points4]
y_values4 = [point[1] for point in points4]
colors4 = [point[2] for point in points4]

# Points 5
x_values5 = [point[0] for point in points5]
y_values5 = [point[1] for point in points5]
colors5 = [point[2] for point in points5]

# Points 6
x_values6 = [point[0] for point in points6]
y_values6 = [point[1] for point in points6]
colors6 = [point[2] for point in points6]

# Points 7
x_values7 = [point[0] for point in points7]
y_values7 = [point[1] for point in points7]
colors7 = [point[2] for point in points7]


# Create a figure with seven subplots
fig, axs = plt.subplots(1, 7, sharey=True, facecolor='w', figsize=(16, 6))

# Subplot 1
axs[0].scatter(x_values1, y_values1, c=colors1)
# Subplot 2
axs[1].scatter(x_values2, y_values2, c=colors2)
# Subplot 3
axs[2].scatter(x_values3, y_values3, c=colors3)
# Subplot 4
axs[3].scatter(x_values4, y_values4, c=colors4)
# Subplot 5
axs[4].scatter(x_values5, y_values5, c=colors5)
# Subplot 6
axs[5].scatter(x_values6, y_values6, c=colors6)
# Subplot 7
axs[6].scatter(x_values7, y_values7, c=colors7)

# Create two subplots with shared y-axis
# f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')

# Set the y-axis labels
axs[0].set_yticks([1, 2, 3])
axs[0].set_yticklabels(['1', '2', '3'])

# Plot the points
# for x, y, color in zip(x_values1, y_values1, colors1):
#     ax.scatter(x, y, color=color)
    
# for x, y, color in zip(x_values2, y_values2, colors2):
#     ax2.scatter(x, y, color=color)

# Set the x-axis limits for each subplot
axs[0].set_xlim(5, 25)
axs[1].set_xlim(125, 145)

# Set the x-axis labels
axs[0].set_xticks([6, 12, 18, 24])
axs[0].set_xticklabels(['6', '12', '18', '24'])
axs[1].set_xticks([126, 132, 138, 144])
axs[1].set_xticklabels(['126', '132', '138', '144'])

# Hide the spines between ax and ax2
axs[0].spines['right'].set_visible(False)
axs[1].spines['left'].set_visible(False)
axs[0].yaxis.tick_left()
axs[1].yaxis.tick_right()

# Set plot title and labels
plt.suptitle('Wi-Fi Scanninger', fontsize=12)
fig.text(0.5, 0.01, 'Sekunder', ha='center', fontsize=10)
axs[0].set_ylabel('Antal på hinanden følgende\n detekteringer før skift')

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=axs[0].transAxes, color='k', clip_on=False)
axs[0].plot((1-d, 1+d), (-d, +d), **kwargs)
axs[0].plot((1-d, 1+d), (1-d, 1+d), **kwargs)

kwargs.update(transform=axs[1].transAxes)  # switch to the bottom axes
axs[1].plot((-d, +d), (1-d, 1+d), **kwargs)
axs[1].plot((-d, +d), (-d, +d), **kwargs)

# Display the plot
plt.show()
