# Import libraries
import matplotlib.pyplot as plt
import numpy as np
 
 
# Creating dataset
np.random.seed(10)
 
data_1 = [0.8917,	0.8871,	0.8510,	0.8619,	0.8309]
data_2 = [0.8960,	0.8766,	0.8638,	0.8496,	0.8507]
data_3 = [0.7885,	0.7823,	0.7658,	0.7781,	0.7361]
data_4 = [0.8290,	0.8218,	0.8022,	0.8231,	0.7959]
data_5 = [0.8118,	0.7741,	0.8052,	0.7906,	0.7856]
data_6 = [0.7863,	0.7900,	0.7679,	0.7659,	0.7707]
dataSVM3 = [data_1, data_2, data_3, data_4, data_5, data_6]

""" SVM 4 """
dataSVM4 = [
    [0.8053, 0.7730, 0.7553, 0.7163, 0.6936],
    [0.7950, 0.7882, 0.7419, 0.7217, 0.6793],
    [0.6348, 0.6375, 0.6342, 0.6216, 0.6237],
    [0.6400, 0.6525, 0.6442, 0.6511, 0.6123],
    [0.6265, 0.5296, 0.6289, 0.6325, 0.6139],
    [0.6337, 0.6510, 0.6462, 0.6315, 0.5989]
]

""" NN3-B """
dataNN3B = [
    [0.7572, 0.6048, 0.5374, 0.5091, 0.4611],
    [0.7785, 0.6651, 0.6102, 0.6018, 0.5110],
    [0.6535, 0.5382, 0.5050, 0.4695, 0.4516],
    [0.7193, 0.6105, 0.5808, 0.5708, 0.4764],
    [0.6680, 0.5673, 0.5118, 0.4905, 0.4586],
    [0.6318, 0.5523, 0.5299, 0.5414, 0.4622]
]

""" NN3-UB """
dataNN3UB = [[0.7188, 0.6823, 0.6036, 0.5104, 0.5049],
        [0.7376, 0.6998, 0.5639, 0.5482, 0.5451],
        [0.6475, 0.6213, 0.5698, 0.4746, 0.4770],
        [0.6837, 0.6402, 0.5411, 0.5225, 0.5230],
        [0.6743, 0.6304, 0.5817, 0.5083, 0.4999],
        [0.6079, 0.5860, 0.5084, 0.4845, 0.4869]]

""" NN4-UB """
dataNN4UB = [[0.6148, 0.5326, 0.4243, 0.3866, 0.3058],
        [0.6822, 0.5867, 0.4486, 0.4286, 0.4193],
        [0.4864, 0.4566, 0.3822, 0.3577, 0.2904],
        [0.5060, 0.5026, 0.4077, 0.3999, 0.3662],
        [0.4824, 0.4711, 0.4022, 0.3627, 0.2862],
        [0.5087, 0.4982, 0.4105, 0.3991, 0.3842]]

fig = plt.figure(figsize =(10, 7))
ax = fig.add_subplot(111)
 
# Creating axes instance
# ax = fig.add_axes([0, 0, 1, 1])
 
# Creating plot
bp = ax.boxplot(dataNN3B)

# Adding title
plt.title("NN3-B")

# x-axis labels
ax.set_xticklabels(['Eget dataset morgen', 'Eget dataset aften', 'Anden dag morgen', 'Anden dag aften', 'Andet tidspunkt morgen', 'Andet tidspunkt aften'], rotation=45)
fig.subplots_adjust(bottom=0.25)
 


ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
 
# show plot
plt.show()


# # Import libraries
# import matplotlib.pyplot as plt
# import numpy as np
 
# # Creating dataset
# np.random.seed(10)
# data_1 = np.random.normal(100, 10, 200)
# data_2 = np.random.normal(90, 20, 200)
# data_3 = np.random.normal(80, 30, 200)
# data_4 = np.random.normal(70, 40, 200)
# data = [data_1, data_2, data_3, data_4]
 
# fig = plt.figure(figsize =(10, 7))
# ax = fig.add_subplot(111)
 
# # Creating axes instance
# bp = ax.boxplot(data, patch_artist = True,
#                 notch ='True', vert = 0)
 
# colors = ['#0000FF', '#00FF00',
#           '#FFFF00', '#FF00FF']
 
# for patch, color in zip(bp['boxes'], colors):
#     patch.set_facecolor(color)
 
# # changing color and linewidth of
# # whiskers
# for whisker in bp['whiskers']:
#     whisker.set(color ='#8B008B',
#                 linewidth = 1.5,
#                 linestyle =":")
 
# # changing color and linewidth of
# # caps
# for cap in bp['caps']:
#     cap.set(color ='#8B008B',
#             linewidth = 2)
 
# # changing color and linewidth of
# # medians
# for median in bp['medians']:
#     median.set(color ='red',
#                linewidth = 3)
 
# # changing style of fliers
# for flier in bp['fliers']:
#     flier.set(marker ='D',
#               color ='#e7298a',
#               alpha = 0.5)
     
# # x-axis labels
# ax.set_yticklabels(['data_1', 'data_2',
#                     'data_3', 'data_4'])
 
# # Adding title
# plt.title("Customized box plot")
 
# # Removing top axes and right axes
# # ticks
# ax.get_xaxis().tick_bottom()
# ax.get_yaxis().tick_left()
     
# # show plot
# plt.show()