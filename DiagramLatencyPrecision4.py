import matplotlib.pyplot as plt

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

# Create the figure and axis objects
# fig, ax = plt.subplots()

# Create two subplots with shared y-axis
f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')

# Set the y-axis labels

ax2.set_yticks([1, 2, 3])
ax2.set_yticklabels([])
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['1', '2', '3'])

# Plot the points
for x, y, color in zip(x_values1, y_values1, colors1):
    ax.scatter(x, y, color=color)
    
for x, y, color in zip(x_values2, y_values2, colors2):
    ax2.scatter(x, y, color=color)

# Set the x-axis limits for each subplot
ax.set_xlim(0, 25)
ax2.set_xlim(125, 144)

# Hide the spines between ax and ax2
ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.yaxis.tick_left()
# ax.tick_params(labelright='off')
ax2.yaxis.tick_right()
# ax2.tick_params(labelright='off')

# Set plot title and labels
ax.set_title('Data Points')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')

# Add broken axes
# ax.set_ylim(-1.2, 1.2)
# ax2.set_ylim(-1.2, 1.2)
# ax.spines['top'].set_visible(False)
# ax2.spines['top'].set_visible(False)

# Create the gap in the x-axis between 24 and 126
# d1 = .02  # how big to make the diagonal lines in axes coordinates
# kwargs1 = dict(transform=ax.transAxes, color='k', clip_on=False)
# ax.plot((24-d1, 24+d1), (-d1, +d1), **kwargs1)
# ax.plot((24-d1, 24+d1), (1-d1, 1+d1), **kwargs1)
# kwargs2 = dict(transform=ax2.transAxes, color='k', clip_on=False)
# ax2.plot((24-d1, 24+d1), (-d1, +d1), **kwargs2)
# ax2.plot((24-d1, 24+d1), (1-d1, 1+d1), **kwargs2)

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((1-d, 1+d), (-d, +d), **kwargs)
ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
ax2.plot((-d, +d), (-d, +d), **kwargs)

# Manually set x-axis tick positions and limits
# x_ticks = [6, 12, 18, 24, 126, 132, 138, 144]
# plt.xticks(x_ticks)
# plt.xlim(x_ticks[0], x_ticks[-2])  # Exclude the last tick position from the limit

# Display the plot
plt.show()
