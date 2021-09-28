import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(Methods,method_name_list):
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}

    plt.rc('font', **font)

    fig, ax = plt.subplots(ncols=6, nrows=1, figsize=(80, 10))
    # Defining custom 'xlim' and 'ylim' values.
    # custom_xlim = (0, 100)
    custom_ylim = (0, 1.1)
    # Setting the values for all axes.
    # plt.setp(ax, ylim=custom_ylim)
    # N = 1000

    for index,column in enumerate(Methods[0].keys()):
        if column.find('Unnamed')>=0: continue #this column is just indices obtained from the file
        for i,Method in enumerate(Methods):
            print(method_name_list[i],' ',column)
            linestyle="-"
            color = 'g'
            if method_name_list[i].find("FCN")>=0:
                linestyle = "--"
            if method_name_list[i].find("Deep")>=0:
                linestyle = "--"
            if method_name_list[i].find("SegNet")>=0:
                linestyle = "-"
                color='b'
            ax[index-1].plot(Method[column].to_numpy(),color=color, linewidth=5 , linestyle=linestyle , label=method_name_list[i]+' '+column)

            if column.find("iou")>=0:
                ax[index-1].set_ylim((0, 0.4))
            elif column.find("loss")>=0:
                ax[index-1].set_ylim((0, 0.8))
            elif column.find("pix")>=0:
                ax[index-1].set_ylim((0.7, 1))

    for i in ax:
        # for j in i:
        i.legend()
        i.grid(True)


    # plt.savefig("./models/results/Experiment2.png")
    plt.savefig("./models/results/Experiment3-scaled.png")
    plt.clf()



# SegNet_results = pd.read_csv("./models/results/epoch_based_result_SegNet Experiment3.csv")
SegNet_results = pd.read_csv("./models/results/epoch_based_result_SegNet Experiment1.csv")
Ultimate_results = pd.read_csv("./models/results/epoch_based_result_Ultimateshare Experiment1.csv")

# method_name_list = ["Symmetric_columns","last2stages NoSharing","last2stages",
#                         "FCN","DeeplabV3", "SegNet", "Ultimateshare"]
# experiment_name_list = ["Symmetric_columns Experiment2","last2stages_NoSharing Experiment2","last2stages Experiment2",
#                         "FCN Experiment2","DeeplabV3 Experiment2", "SegNet Experiment2", "Ultimateshare Experiment2"]
method_name_list = ["SegNet", "Proposed"]
experiment_name_list = ["SegNet Experiment3", "Ultimateshare Experiment3"]
pd_result = []
base_path = "./models/results/epoch_based_result_"
for index in range(len(experiment_name_list)):
    experiment_name_list[index]= base_path + experiment_name_list[index] + ".csv" # = ./models/results/epoch_based_result_SegNet Experiment3.csv
    pd_result.append(pd.read_csv(experiment_name_list[index]))

#tr_loss_arr	val_loss_arr	meanioutrain	pixelacctrain	meanioutest	pixelacctest
plot(pd_result,method_name_list)
# tr_loss_arr = panda_results["tr_loss_arr"].to_numpy()
# val_loss_arr = panda_results["val_loss_arr"].to_numpy()
# meanioutrain = panda_results["meanioutrain"].to_numpy()
# pixelacctrain = panda_results["pixelacctrain"].to_numpy()
# meanioutest = panda_results["meanioutest"].to_numpy()
# pixelacctest = panda_results["pixelacctest"].to_numpy()
