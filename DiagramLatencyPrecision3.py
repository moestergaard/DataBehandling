"""
Broken axis example, where the x-axis will have a portion cut out.
"""
import matplotlib.pylab as plt
import numpy as np

x = np.linspace(0, 10, 100)
x[75:] = np.linspace(40, 42.5, 25)
y = np.sin(x)

# Create two subplots with shared y-axis
f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w', gridspec_kw={'width_ratios': [3, 1]})

# Plot the same data on both axes
ax.plot(x, y)
ax2.plot(x, y)

# Set the x-axis limits for each subplot
ax.set_xlim(0, 7.5)
ax2.set_xlim(144, 240)

# Hide the spines between ax and ax2
ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.yaxis.tick_left()
ax.tick_params(labelright='off')
ax2.yaxis.tick_right()

# Add broken axes
ax.set_ylim(-1.2, 1.2)
ax2.set_ylim(-1.2, 1.2)
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Create the gap in the x-axis between 24 and 126
d1 = .02  # how big to make the diagonal lines in axes coordinates
kwargs1 = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((24-d1, 24+d1), (-d1, +d1), **kwargs1)
ax.plot((24-d1, 24+d1), (1-d1, 1+d1), **kwargs1)
kwargs2 = dict(transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot((24-d1, 24+d1), (-d1, +d1), **kwargs2)
ax2.plot((24-d1, 24+d1), (1-d1, 1+d1), **kwargs2)

# Create the gap in the x-axis between 144 and 240
d2 = .05
kwargs3 = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((144-d2, 144+d2), (-d2, +d2), **kwargs3)
ax.plot((144-d2, 144+d2), (1-d2, 1+d2), **kwargs3)
kwargs4 = dict(transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot((144-d2, 144+d2), (-d2, +d2), **kwargs4)
ax2.plot((144-d2, 144+d2), (1-d2, 1+d2), **kwargs4)

# Adjust the layout to make room for the broken axes labels
plt.subplots_adjust(wspace=0.05)

plt.show()
