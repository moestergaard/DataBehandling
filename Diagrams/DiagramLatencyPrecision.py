import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

colorsGeneral = ['#252525', 'blue', '#2171b5', 'red', '#fc4e2a']

points = [
    [
        (6, 1, colorsGeneral[0]), (12, 1, colorsGeneral[1]), (18, 1, colorsGeneral[2]), (24, 1, colorsGeneral[2]),
        (6, 2, colorsGeneral[0]), (12, 2, colorsGeneral[0]), (18, 2, colorsGeneral[1]), (24, 2, colorsGeneral[2]),
        (6, 3, colorsGeneral[0]), (12, 3, colorsGeneral[0]), (18, 3, colorsGeneral[0]), (24, 3, colorsGeneral[0])
    ],
    [
        (126, 1, colorsGeneral[2]), (132, 1, colorsGeneral[2]), (138, 1, colorsGeneral[2]), (144, 1, colorsGeneral[3]),
        (126, 2, colorsGeneral[2]), (132, 2, colorsGeneral[2]), (138, 2, colorsGeneral[2]), (144, 2, colorsGeneral[2]),
        (126, 3, colorsGeneral[1]), (132, 3, colorsGeneral[2]), (138, 3, colorsGeneral[2]), (144, 3, colorsGeneral[2])
    ],
    [
        (246, 1, colorsGeneral[4]), (252, 1, colorsGeneral[1]), (258, 1, colorsGeneral[2]), (264, 1, colorsGeneral[2]),
        (246, 2, colorsGeneral[2]), (252, 2, colorsGeneral[2]), (258, 2, colorsGeneral[2]), (264, 2, colorsGeneral[2]),
        (246, 3, colorsGeneral[2]), (252, 3, colorsGeneral[2]), (258, 3, colorsGeneral[2]), (264, 3, colorsGeneral[2])
    ],
    [
        (366, 1, colorsGeneral[2]), (372, 1, colorsGeneral[2]), (378, 1, colorsGeneral[2]), (384, 1, colorsGeneral[3]),
        (366, 2, colorsGeneral[2]), (372, 2, colorsGeneral[2]), (378, 2, colorsGeneral[2]), (384, 2, colorsGeneral[2]),
        (366, 3, colorsGeneral[2]), (372, 3, colorsGeneral[2]), (378, 3, colorsGeneral[2]), (384, 3, colorsGeneral[2])
    ],
    [
        (486, 1, colorsGeneral[4]), (492, 1, colorsGeneral[1]), (498, 1, colorsGeneral[2]), (504, 1, colorsGeneral[2]),
        (486, 2, colorsGeneral[2]), (492, 2, colorsGeneral[2]), (498, 2, colorsGeneral[2]), (504, 2, colorsGeneral[2]),
        (486, 3, colorsGeneral[2]), (492, 3, colorsGeneral[2]), (498, 3, colorsGeneral[2]), (504, 3, colorsGeneral[2])
    ],
    [
        (606, 1, colorsGeneral[2]), (612, 1, colorsGeneral[2]), (618, 1, colorsGeneral[2]), (624, 1, colorsGeneral[3]),
        (606, 2, colorsGeneral[2]), (612, 2, colorsGeneral[2]), (618, 2, colorsGeneral[2]), (624, 2, colorsGeneral[2]),
        (606, 3, colorsGeneral[2]), (612, 3, colorsGeneral[2]), (618, 3, colorsGeneral[2]), (624, 3, colorsGeneral[2])
    ],
    [
        (726, 1, colorsGeneral[4]), (732, 1, colorsGeneral[1]), (738, 1, colorsGeneral[2]), (744, 1, colorsGeneral[2]),
        (726, 2, colorsGeneral[2]), (732, 2, colorsGeneral[2]), (738, 2, colorsGeneral[2]), (744, 2, colorsGeneral[2]),
        (726, 3, colorsGeneral[2]), (732, 3, colorsGeneral[2]), (738, 3, colorsGeneral[2]), (744, 3, colorsGeneral[2])
    ],
    [
        (846, 1, colorsGeneral[2]), (852, 1, colorsGeneral[2]), (858, 1, colorsGeneral[2]), (864, 1, colorsGeneral[3]),
        (846, 2, colorsGeneral[2]), (852, 2, colorsGeneral[2]), (858, 2, colorsGeneral[2]), (864, 2, colorsGeneral[2]),
        (846, 3, colorsGeneral[2]), (852, 3, colorsGeneral[2]), (858, 3, colorsGeneral[2]), (864, 3, colorsGeneral[2])
    ],
    [
        (966, 1, colorsGeneral[4]), (972, 1, colorsGeneral[1]), (978, 1, colorsGeneral[2]), (984, 1, colorsGeneral[2]),
        (966, 2, colorsGeneral[2]), (972, 2, colorsGeneral[2]), (978, 2, colorsGeneral[2]), (984, 2, colorsGeneral[2]),
        (966, 3, colorsGeneral[2]), (972, 3, colorsGeneral[2]), (978, 3, colorsGeneral[2]), (984, 3, colorsGeneral[2])
    ],
    [
        (1086, 1, colorsGeneral[2]), (1092, 1, colorsGeneral[2]), (1098, 1, colorsGeneral[2]), (1104, 1, colorsGeneral[3]),
        (1086, 2, colorsGeneral[3]), (1092, 2, colorsGeneral[4]), (1098, 2, colorsGeneral[4]), (1104, 2, colorsGeneral[1]),
        (1086, 3, colorsGeneral[2]), (1092, 3, colorsGeneral[2]), (1098, 3, colorsGeneral[2]), (1104, 3, colorsGeneral[2])
    ]
]

fig, axs = plt.subplots(1, 10, sharey=True, facecolor='w', figsize=(16, 5))
fig.subplots_adjust(right=0.85)
fig.subplots_adjust(left=0.05)
fig.subplots_adjust(bottom=0.2)

# Iterate through each subplot and plot the data
for i in range(10):
    points_i = points[i]
    x_values = [point[0] for point in points_i]
    y_values = [point[1] for point in points_i]
    colors = [point[2] for point in points_i]

    axs[i].scatter(x_values, y_values, c=colors)
    axs[i].set_xlim((i * 120) + 4, (i * 120) + 26)
    axs[i].set_xticks([(i * 120) + 6, (i * 120) + 12, (i * 120) + 18, (i * 120) + 24])
    axs[i].set_xticklabels([str(6 + 120 * i), str(12 + 120 * i), str(18 + 120 * i), str(24 + 120 * i)], rotation = 45, fontsize=10)

    # Hide spines between subplots
    if i > 0:
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].tick_params(left=False)
    
    if i == 9:
        axs[i].spines['right'].set_visible(True)

    # Set y-axis labels for the first subplot
    if i == 0:
        axs[i].spines['right'].set_visible(False)
        axs[i].set_yticks([1, 2, 3])
        axs[i].set_yticklabels(['1', '2', '3'])

    # Add diagonal lines to indicate shared y-axis
    if i < 9:
        d = 0.015
        kwargs = dict(transform=axs[i].transAxes, color='k', clip_on=False)
        axs[i].plot((1 - d, 1 + d), (-d, +d), **kwargs)
        axs[i].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        kwargs.update(transform=axs[i + 1].transAxes)
        axs[i + 1].plot((-d, +d), (1 - d, 1 + d), **kwargs)
        axs[i + 1].plot((-d, +d), (-d, +d), **kwargs)


axs[0].set_ylabel('Antal på hinanden følgende\n prædikteringer før skift', fontsize = 12)
axs[4].set_xlabel('Sekunder', fontsize = 12, labelpad=10)
axs[4].set_title('Wi-Fi Scanninger', fontsize=14, pad=15)

labels = ['Intet rum', 'Korrekt rum', 'Stadigvæk korrekt rum', 'Forkert rum', 'Stadigvæk forkert rum']
handles = [Line2D([0], [0], marker = 'o', linestyle='none', color=color) for color in colorsGeneral]
plt.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)

plt.show()