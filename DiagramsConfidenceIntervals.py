import numpy as np
import matplotlib.pyplot as plt

# Define data
data_1 = [0.8917,	0.8871,	0.8510,	0.8619,	0.8309]
data_2 = [0.8960,	0.8766,	0.8638,	0.8496,	0.8507]
data_3 = [0.7885,	0.7823,	0.7658,	0.7781,	0.7361]
data_4 = [0.8118,	0.7741,	0.8052,	0.7906,	0.7856]
data_5 = [0.8290,	0.8218,	0.8022,	0.8231,	0.7959]
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

# """ NN3-UB """
# dataNN3UB = [[0.7188, 0.6823, 0.6036, 0.5104, 0.5049],
#         [0.7376, 0.6998, 0.5639, 0.5482, 0.5451],
#         [0.6475, 0.6213, 0.5698, 0.4746, 0.4770],
#         [0.6837, 0.6402, 0.5411, 0.5225, 0.5230],
#         [0.6743, 0.6304, 0.5817, 0.5083, 0.4999],
#         [0.6079, 0.5860, 0.5084, 0.4845, 0.4869]]

# """ NN4-UB """
# dataNN4UB = [[0.6148, 0.5326, 0.4243, 0.3866, 0.3058],
#         [0.6822, 0.5867, 0.4486, 0.4286, 0.4193],
#         [0.4864, 0.4566, 0.3822, 0.3577, 0.2904],
#         [0.5060, 0.5026, 0.4077, 0.3999, 0.3662],
#         [0.4824, 0.4711, 0.4022, 0.3627, 0.2862],
#         [0.5087, 0.4982, 0.4105, 0.3991, 0.3842]]

# Define confidence level
confidence_level = 0.95
z_score = 2.776

# # Loop through data and calculate confidence intervals
# fig, ax = plt.subplots()
# x = np.arange(len(dataSVM3))
# width = 0.01

# for i, data in enumerate(dataSVM3):
#     mean = np.mean(data)
#     std = np.sqrt(np.sum((data-mean)**2)/(len(data)-1))
#     margin_of_error = z_score * (std / np.sqrt(len(data)))
#     lower_bound = mean - margin_of_error
#     upper_bound = mean + margin_of_error
#     conf_int = (lower_bound, upper_bound)

#     ax.errorbar(x[i], mean, yerr=margin_of_error, fmt='o', color='red')
#     ax.plot([x[i], x[i]], [lower_bound, upper_bound], '-', linewidth=4, color='blue')
#     ax.plot([x[i] - width/2, x[i] + width/2], [mean, mean], '-', linewidth=2, color='red')
#     ax.text(x[i], upper_bound+0.01, '{:.4f}'.format(upper_bound), ha='center')
#     ax.text(x[i], lower_bound-0.03, '{:.4f}'.format(lower_bound), ha='center')

# # Set plot parameters
# ax.set_xticks(x)
# ax.set_xticklabels(['Eget dataset morgen', 'Eget dataset aften', 'Anden dag morgen', 'Andet tidspunkt morgen', 'Anden dag aften', 'Andet tidspunkt aften'], rotation=45)
# fig.subplots_adjust(bottom=0.3)
# ax.set_ylim([0.7, 1.0])
# ax.set_title('SVM3 - 95% Konfidensinterval')
# ax.set_ylabel('Nøjagtighed')

# plt.show()










# Loop through data and calculate confidence intervals
fig, ax = plt.subplots()
x = np.arange(len(dataSVM3))
width = 0.01

confidence_level = 0.95
z_score = 2.776

for i, data in enumerate(dataSVM3):
    mean = np.mean(data)
    std = np.sqrt(np.sum((data-mean)**2)/(len(data)-1))
    margin_of_error = z_score * (std / np.sqrt(len(data)))
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    
    # Plot data points
    ax.plot(x[i], mean, marker='o', color='blue')
    
    # Plot error bars
    ax.errorbar(x[i], mean, yerr=margin_of_error, fmt='none', ecolor='blue', capsize=3)
    # ax.text(x[i], upper_bound+0.001, '{:.4f}'.format(upper_bound), ha='center')
    # ax.text(x[i], lower_bound-0.003, '{:.4f}'.format(lower_bound), ha='center')

# Set axis labels and title
ax.set_xticks(x)
ax.set_xticklabels(['data_1', 'data_2', 'data_3', 'data_4', 'data_5', 'data_6'])
ax.set_xlabel('Data set')
ax.set_ylabel('Accuracy')
ax.set_title('SVM3 Confidence Intervals')

# Show plot
plt.show()








import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_confidence_intervals(data1, data2):
    n = len(data1)
    x = np.arange(n)  # positions of the bars on the x-axis
    width = 0.35  # width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, [np.mean(row) for row in data1], width, label='Dataset 1')
    rects2 = ax.bar(x + width/2, [np.mean(row) for row in data2], width, label='Dataset 2')
    for i in range(n):
        # calculate the confidence interval for each row of the datasets
        ci1 = stats.t.interval(0.95, len(data1[i])-1, loc=np.mean(data1[i]), scale=stats.sem(data1[i]))
        ci2 = stats.t.interval(0.95, len(data2[i])-1, loc=np.mean(data2[i]), scale=stats.sem(data2[i]))
        # plot the confidence intervals as error bars on the bars
        ax.errorbar(x[i] - width/2, np.mean(data1[i]), yerr=(ci1[1]-ci1[0])/2, fmt='none', capsize=10, capthick=2, ecolor='black')
        ax.errorbar(x[i] + width/2, np.mean(data2[i]), yerr=(ci2[1]-ci2[0])/2, fmt='none', capsize=10, capthick=2, ecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(['Row {}'.format(i+1) for i in range(n)])
    ax.legend()
    plt.show()
    
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

plot_confidence_intervals(dataSVM4, dataNN3B)