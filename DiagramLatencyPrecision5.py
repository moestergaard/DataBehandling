import matplotlib.pyplot as plt
import numpy as np

# Define the data points for all subplots
points = [
    [
        (6, 1, 'blue'), (12, 1, 'blue'), (18, 1, 'red'), (24, 1, 'blue'),
        (6, 2, 'black'), (12, 2, 'black'), (18, 2, 'black'), (24, 2, 'blue'),
        (6, 3, 'black'), (12, 3, 'black'), (18, 3, 'black'), (24, 3, 'black')
    ],
    [
        (126, 1, 'blue'), (132, 1, 'red'), (138, 1, 'blue'), (144, 1, 'blue'),
        (126, 2, 'blue'), (132, 2, 'blue'), (138, 2, 'blue'), (144, 2, 'blue'),
        (126, 3, 'black'), (132, 3, 'blue'), (138, 3, 'blue'), (144, 3, 'blue')
    ],
    [
        (246, 1, 'red'), (252, 1, 'blue'), (258, 1, 'blue'), (264, 1, 'red'),
        (246, 2, 'blue'), (252, 2, 'blue'), (258, 2, 'blue'), (264, 2, 'blue'),
        (246, 3, 'blue'), (252, 3, 'blue'), (258, 3, 'blue'), (264, 3, 'blue')
    ],
    [
        (366, 1, 'blue'), (372, 1, 'blue'), (378, 1, 'red'), (384, 1, 'blue'),
        (366, 2, 'blue'), (372, 2, 'blue'), (378, 2, 'blue'), (384, 2, 'blue'),
        (366, 3, 'blue'), (372, 3, 'blue'), (378, 3, 'blue'), (384, 3, 'blue')
    ],
    [
        (486, 1, 'blue'), (492, 1, 'red'), (498, 1, 'blue'), (504, 1, 'blue'),
        (486, 2, 'blue'), (492, 2, 'blue'), (498, 2, 'blue'), (504, 2, 'blue'),
        (486, 3, 'blue'), (492, 3, 'blue'), (498, 3, 'blue'), (504, 3, 'blue')
    ],
    [
        (606, 1, 'red'), (612, 1, 'blue'), (618, 1, 'blue'), (624, 1, 'red'),
        (606, 2, 'blue'), (612, 2, 'blue'), (618, 2, 'red'), (624, 2, 'red'),
        (606, 3, 'blue'), (612, 3, 'blue'), (618, 3, 'blue'), (624, 3, 'blue')
    ],
    [
        (726, 1, 'blue'), (732, 1, 'blue'), (738, 1, 'red'), (744, 1, 'blue'),
        (726, 2, 'red'), (732, 2, 'blue'), (738, 2, 'blue'), (744, 2, 'blue'),
        (726, 3, 'blue'), (732, 3, 'blue'), (738, 3, 'blue'), (744, 3, 'blue')
    ]
]

# Create a figure with seven subplots
fig, axs = plt.subplots(1, 7, sharey=True, facecolor='w', figsize=(16, 6))

# Iterate through each subplot and plot the data
for i in range(7):
    points_i = points[i]
    x_values = [point[0] for point in points_i]
    y_values = [point[1] for point in points_i]
    colors = [point[2] for point in points_i]

    axs[i].scatter(x_values, y_values, c=colors)
    axs[i].set_xlim((i * 120) + 5, (i * 120) + 25)
    axs[i].set_xticks([(i * 120) + 6, (i * 120) + 12, (i * 120) + 18, (i * 120) + 24])
    axs[i].set_xticklabels([str(6 + 120 * i), str(12 + 120 * i), str(18 + 120 * i), str(24 + 120 * i)])

    # Hide spines between subplots
    if i > 0:
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].tick_params(left=False)
        # axs[i].yaxis.tick_left()
        # axs[i].yaxis.tick_right()
    
    if i == 6:
        axs[i].spines['right'].set_visible(True)

    # Set y-axis labels for the first subplot
    if i == 0:
        axs[i].spines['right'].set_visible(False)
        axs[i].set_yticks([1, 2, 3])
        axs[i].set_yticklabels(['1', '2', '3'])
    else:
        axs[i].set_yticks([])
        axs[i].set_yticklabels([])

    # Add diagonal lines to indicate shared y-axis
    if i < 6:
        d = 0.015
        kwargs = dict(transform=axs[i].transAxes, color='k', clip_on=False)
        axs[i].plot((1 - d, 1 + d), (-d, +d), **kwargs)
        axs[i].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        kwargs.update(transform=axs[i + 1].transAxes)
        axs[i + 1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
        axs[i + 1].plot((-d, +d), (-d, +d), **kwargs)


# Set plot title and labels
plt.suptitle('Wi-Fi Scanninger', fontsize=12)
fig.text(0.5, 0.01, 'Sekunder', ha='center', fontsize=10)
axs[0].set_ylabel('Antal på hinanden følgende\n detekteringer før skift')

# Adjust spacing between subplots
# plt.subplots_adjust(wspace=0)

# Display the plot
plt.show()