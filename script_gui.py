import os
import sys
import csv

if "../functions/" not in sys.path:
    sys.path.append("../functions/")

import matplotlib.pyplot as plt
import pandas as pd

import torch
import numpy as np
from torch import nn
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# lib in '../functions/'
import functions.functions_diagnostic as ft
import functions.functions_network_pytorch as fnp
from sklearn.metrics import precision_recall_fscore_support

import tkinter as tk
from tkinter import filedialog as fd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from matplotlib.figure import Figure


file_name_train = "Th12F_meanFill.csv"  # Train

# ideally we would want to separate GUI elements from AE elements, TODO for later
# TODO There is no need to train the network every time we change the test file
# TODO make the input boxes dependant on the database (i.e. make the GUI work for LUNG too for example)
# TODO Less code repeating during the init phase
class testLatentSpace:
    def __init__(self):

        self.main = tk.Tk()
        # NB: this option will only work on Windows, maximizes the window
        # self.main.state("zoomed")
        self.main.option_add("*Font", "12")  # change font size
        self.main.wm_title("Latent Space Tests")

        self.__init_hyperparam_input__()
        self.__init_plot__()

        tk.Grid.rowconfigure(self.main, 0, weight=1)
        tk.Grid.columnconfigure(self.main, 0, weight=1)

        self.upper_text = tk.Label(
            master=self.main,
            text="Prognosis with confidence score using the latent space of a supervised autoencoder",
            fg="blue",
        )

        self.button_get_csv = tk.Button(
            master=self.main, text="Choose test csv", command=self.get_test_file
        )
        self.button_run = tk.Button(master=self.main, text="Run", command=self.run_net)

        self.top_genes_title = tk.Label(master=self.main, text="Top Features")
        self.top_genes_frame = tk.Frame(master=self.main)

        ############ Put widgets on the window grid ################

        self.upper_text.grid(
            column=0, row=0, sticky="NSEW", columnspan=len(self.hyparam_fields)
        )

        for i, frame in enumerate(self.hyparam_frames_list):
            frame.grid(column=i, row=1, sticky="NSEW")

        self.plot_frame.grid(
            column=0, row=2, columnspan=len(self.hyparam_frames_list), sticky="NSEW"
        )

        self.button_get_csv.grid(
            column=0, row=3, columnspan=len(self.hyparam_frames_list), sticky="NSEW"
        )
        self.button_run.grid(
            column=0, row=4, columnspan=len(self.hyparam_frames_list), sticky="NSEW"
        )

        self.top_genes_title.grid(
            column=len(self.hyparam_frames_list), row=0, sticky="NSEW"
        )
        self.top_genes_frame.grid(
            column=len(self.hyparam_frames_list), row=1, rowspan=4, sticky="NSEW"
        )

    def __init_hyperparam_input__(self):

        #  All controllable hyperparameters
        self.SEED = tk.IntVar(value=6)
        self.ETA = tk.IntVar(value=50)
        self.N_EPOCHS = tk.IntVar(value=20)
        self.doScale = tk.BooleanVar(value=False)
        self.doLog = tk.BooleanVar(value=False)
        self.n_hidden = tk.IntVar(value=64)

        self.hyparam_frames_list = []
        self.hyparam_fields = []

        frame, wid = self.__init_widget_helper__("Seed", "Entry", self.SEED)
        self.hyparam_frames_list.append(frame)
        self.hyparam_fields.append(wid)

        frame, wid = self.__init_widget_helper__("eta", "Entry", self.ETA)
        self.hyparam_frames_list.append(frame)
        self.hyparam_fields.append(wid)

        frame, wid = self.__init_widget_helper__("Nb Epochs", "Entry", self.N_EPOCHS)
        self.hyparam_frames_list.append(frame)
        self.hyparam_fields.append(wid)

        frame, wid = self.__init_widget_helper__(
            "Do scaling", "CheckButton", self.doScale
        )
        self.hyparam_frames_list.append(frame)
        self.hyparam_fields.append(wid)

        frame, wid = self.__init_widget_helper__(
            "Do log transform", "CheckButton", self.doLog
        )
        self.hyparam_frames_list.append(frame)
        self.hyparam_fields.append(wid)

        frame, wid = self.__init_widget_helper__("N hidden", "Entry", self.n_hidden)
        self.hyparam_frames_list.append(frame)
        self.hyparam_fields.append(wid)

    def get_test_file(self):
        full_path = fd.askopenfilename(
            filetypes=[("CSV Files", "*.csv")], initialdir="."
        )
        self.file_name_test = full_path.split("/")[-1]
        self.button_run["text"] = f"Run on {self.file_name_test}"

    def __init_widget_helper__(self, frame_title, widget_type, var=None, values=None):
        frame = tk.LabelFrame(text=frame_title)
        if widget_type == "CheckButton":
            wid = tk.Checkbutton(master=frame, variable=var)
            wid.select()
        elif widget_type == "Entry":
            wid = tk.Entry(master=frame, textvariable=var)
        elif widget_type == "Spinbox":
            if values is not None:
                wid = tk.Spinbox(master=frame, values=values, textvariable=var)
            else:
                wid = tk.Spinbox(master=frame, textvariable=var)
        else:
            print(f"Invalid widget type : {widget_type}. Defaulted to Entry")
            wid = tk.Entry(master=frame, textvariable=var)

        wid.pack()
        return frame, wid

    def __init_plot__(self):
        self.plot_frame = tk.LabelFrame(master=self.main, text="Plot")
        self.fig = Figure()
        self.t = np.arange(0, 3, 0.01)  # peu importe
        self.ax = self.fig.add_subplot()

        self.canvas = FigureCanvasTkAgg(
            self.fig, master=self.plot_frame
        )  # A tk.DrawingArea
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        # pack_toolbar=False will make it easier to use a layout manager later on, but is new
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

    def update_topGenes_display(self, outputPath):
        # put the newest topgenes in the grid

        self.top_genes_frame.destroy()  # clean up
        self.top_genes_frame = tk.Frame(master=self.main)
        self.top_genes_frame.grid(
            column=len(self.hyparam_frames_list), row=1, rowspan=3, sticky="NSEW"
        )
        with open(f"{outputPath}topGenes_for_display.csv") as file:
            reader = csv.reader(file, delimiter=";")
            r = 0  # row
            for col in reader:
                c = 0
                for row in col:
                    if c == 1 and r > 0:  # for weights (only keep 5 decimal places)
                        label = tk.Label(
                            master=self.top_genes_frame,
                            text=f"{float(row):.5f}",
                            relief=tk.RIDGE,
                        )
                    else:
                        label = tk.Label(
                            master=self.top_genes_frame, text=row, relief=tk.RIDGE
                        )
                    label.grid(row=r, column=c, sticky="NSEW")
                    c += 1
                r += 1

    def ShowPcaTsne(
        self,
        X,
        Y,
        data_encoder,
        center_distance,
        class_len,
        tit,
        pcafit=None,
        test_legends=None,
    ):
        """ Visualization with PCA and Tsne
        Args:
            X: numpy - original imput matrix
            Y: numpy - label matrix  
            data_encoder: tensor  - latent sapce output, encoded data  
            center_distance: numpy - center_distance matrix
            class_len: int - number of class 
        Return:
            Non, just show results in 2d space  
        """

        # Define the color list for plot
        color = [
            "#1F77B4",
            "#FF7F0E",
            "#2CA02C",
            "#D62728",
            "#9467BD",
            "#8C564B",
            "#E377C2",
            "#BCBD22",
            "#17BECF",
            "#40004B",
            "#762A83",
            "#9970AB",
            "#C2A5CF",
            "#E7D4E8",
            "#F7F7F7",
            "#D9F0D3",
            "#A6DBA0",
            "#5AAE61",
            "#1B7837",
            "#00441B",
            "#8DD3C7",
            "#FFFFB3",
            "#BEBADA",
            "#FB8072",
            "#80B1D3",
            "#FDB462",
            "#B3DE69",
            "#FCCDE5",
            "#D9D9D9",
            "#BC80BD",
            "#CCEBC5",
            "#FFED6F",
        ]

        # Do pca for original data
        pca = PCA(n_components=2)
        pca_centre = PCA(n_components=2)

        # Do pca for encoder data if cluster>2
        if data_encoder.shape[1] != 3:  # layer code_size >2  (3= 2+1 data+labels)
            data_encoder_pca = data_encoder[:, :-1]

            if tit == "Latent Space Test":
                X_encoder_pca = pcafit.transform(data_encoder_pca)
            else:
                X_encoder_pca = pca.fit(data_encoder_pca).transform(data_encoder_pca)
            Y_encoder_pca = data_encoder[:, -1].astype(int)
        else:
            X_encoder_pca = data_encoder[:, :-1]
            Y_encoder_pca = data_encoder[:, -1].astype(int)
        if tit == "Latent Space Test":
            color_encoder = [color[i + class_len] for i in Y_encoder_pca]
        else:
            color_encoder = [color[i] for i in Y_encoder_pca]

        # Plot
        title2 = "Latent Space"

        self.ax.set_title(title2)
        if tit == "Latent Space Test":
            for i, patient in enumerate(X_encoder_pca):
                point = self.ax.scatter(
                    patient[0], patient[1], c=color_encoder[i], marker="s"
                )
                if test_legends is not None:
                    point.set_label(test_legends[i])
        else:
            self.ax.scatter(X_encoder_pca[:, 0], X_encoder_pca[:, 1], c=color_encoder)
        return pca

    def run_net(self):
        # ------------ Parameters ---------

        ####### Set of parameters : #######

        # Set seed
        seed = [self.SEED.get()]
        ETA = self.ETA.get()  # Control feature selection

        # Set device (Gpu or cpu)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        nfold = 1
        N_EPOCHS = self.N_EPOCHS.get()
        N_EPOCHS_MASKGRAD = (
            self.N_EPOCHS.get()
        )  # number of epochs for trainning masked graident
        # learning rate
        LR = 0.0005
        BATCH_SIZE = 8
        LOSS_LAMBDA = 0.0005  # Total loss =λ * loss_autoencoder +  loss_classification

        #    DoTopGenes = True
        doTopGenes = True

        # Scaling
        doScale = self.doScale.get()
        #    doScale = False

        # dolog transform
        doLog = self.doLog.get()

        # Loss functions for reconstruction
        criterion_reconstruction = nn.SmoothL1Loss(reduction="sum")  # SmoothL1Loss

        # Loss functions for classification
        criterion_classification = nn.CrossEntropyLoss(reduction="sum")

        TIRO_FORMAT = True

        # Choose Net
        #    net_name = 'LeNet'
        net_name = "netBio"
        n_hidden = self.n_hidden.get()  # number of neurons on the netBio hidden layer

        # Save Results or not
        SAVE_FILE = True
        # Output Path
        outputPath = "results_diag/" + file_name_train.split(".")[0] + "/"
        if not os.path.exists(outputPath):  # make the directory if it does not exist
            os.makedirs(outputPath)

        # Do pca
        doPCA = True
        run_model = "No_proj"
        # Do projection at the middle layer or not
        DO_PROJ_middle = False

        # Do projection (True)  or not (False)
        #    GRADIENT_MASK = False
        GRADIENT_MASK = True
        if GRADIENT_MASK:

            run_model = "ProjectionLastEpoch"
        # Choose projection function
        if not GRADIENT_MASK:
            TYPE_PROJ = "No_proj"
            TYPE_PROJ_NAME = "No_proj"
        else:
            #        TYPE_PROJ = ft.proj_l1ball         # projection l1
            TYPE_PROJ = ft.proj_l11ball  # original projection l11
            #        TYPE_PROJ = ft.proj_l21ball        # projection l21

            TYPE_PROJ_NAME = TYPE_PROJ.__name__

        # ------------ Main loop ---------
        # Load data

        (
            X,
            Y,
            feature_name,
            label_name,
            patient_name,
            X_test,
            Y_test,
            patient_name_test,
        ) = ft.ReadData(
            file_name_train,
            self.file_name_test,
            TIRO_FORMAT=TIRO_FORMAT,
            doScale=doScale,
            doLog=doLog,
        )  # Load files data

        feature_len = len(feature_name)
        class_len = len(label_name)
        print(
            "Number of features: {}, Number of classes: {}".format(
                feature_len, class_len
            )
        )

        train_dl, val_dl, train_len, val_len, _ = ft.CrossVal(
            X, Y, patient_name, BATCH_SIZE, seed=seed[0]
        )

        dtrain = ft.LoadDataset(X, Y, patient_name)
        train_dl = torch.utils.data.DataLoader(
            dtrain, batch_size=BATCH_SIZE, shuffle=True
        )

        dtest = ft.LoadDataset(X_test, Y_test, patient_name_test)
        test_dl = torch.utils.data.DataLoader(dtest, batch_size=1)

        train_len = len(dtrain)
        test_len = len(dtest)

        accuracy_train = np.zeros((nfold * len(seed), class_len + 1))
        accuracy_test = np.zeros((nfold * len(seed), class_len + 1))
        data_train = np.zeros((nfold * len(seed), 7))
        data_test = np.zeros((nfold * len(seed), 7))
        correct_prediction = []
        s = 0
        for SEED in seed:

            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            for i in range(nfold):

                print(
                    "Len of train set: {}, Len of test set:: {}".format(
                        train_len, test_len
                    )
                )
                print("----------- Début iteration ", i, "----------------")
                # Define the SEED to fix the initial parameters
                np.random.seed(SEED)
                torch.manual_seed(SEED)
                torch.cuda.manual_seed(SEED)

                # run AutoEncoder
                if net_name == "LeNet":
                    net = ft.LeNet_300_100(
                        n_inputs=feature_len, n_outputs=class_len
                    ).to(
                        device
                    )  # LeNet
                if net_name == "netBio":
                    net = ft.netBio(feature_len, class_len, n_hidden).to(
                        device
                    )  # netBio

                weights_entry, spasity_w_entry = fnp.weights_and_sparsity(net)

                if GRADIENT_MASK:
                    run_model = "ProjectionLastEpoch"

                optimizer = torch.optim.Adam(net.parameters(), lr=LR)
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, 150, gamma=0.1
                )
                (
                    data_encoder,
                    data_decoded,
                    epoch_loss,
                    best_test,
                    net,
                ) = ft.RunAutoEncoder(
                    net,
                    criterion_reconstruction,
                    optimizer,
                    lr_scheduler,
                    train_dl,
                    train_len,
                    val_dl,
                    val_len,
                    N_EPOCHS,
                    outputPath,
                    SAVE_FILE,
                    DO_PROJ_middle,
                    run_model,
                    criterion_classification,
                    LOSS_LAMBDA,
                    feature_name,
                    TYPE_PROJ,
                    ETA,
                )
                labelpredict = data_encoder[:, :-1].max(1)[1].cpu().numpy()
                labelpredict = data_encoder[:, :-1].max(1)[1].cpu().numpy()
                # Do masked gradient

                if GRADIENT_MASK:
                    print("\n--------Running with masked gradient-----")
                    print("-----------------------")
                    zero_list = []
                    tol = 1.0e-3
                    for index, param in enumerate(list(net.parameters())):
                        if (
                            index < len(list(net.parameters())) / 2 - 2
                            and index % 2 == 0
                        ):
                            ind_zero = torch.where(torch.abs(param) < tol)
                            zero_list.append(ind_zero)

                    # Get initial network and set zeros
                    # Recall the SEED to get the initial parameters
                    np.random.seed(SEED)
                    torch.manual_seed(SEED)
                    torch.cuda.manual_seed(SEED)

                    # run AutoEncoder
                    if net_name == "LeNet":
                        net = ft.LeNet_300_100(
                            n_inputs=feature_len, n_outputs=class_len
                        ).to(
                            device
                        )  # LeNet
                    if net_name == "netBio":
                        net = ft.netBio(feature_len, class_len, n_hidden).to(
                            device
                        )  # FairNet
                    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
                    lr_scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer, 150, gamma=0.1
                    )

                    for index, param in enumerate(list(net.parameters())):
                        if (
                            index < len(list(net.parameters())) / 2 - 2
                            and index % 2 == 0
                        ):
                            param.data[zero_list[int(index / 2)]] = 0

                    run_model = "MaskGrad"
                    (
                        data_encoder,
                        data_decoded,
                        epoch_loss,
                        best_test,
                        net,
                    ) = ft.RunAutoEncoder(
                        net,
                        criterion_reconstruction,
                        optimizer,
                        lr_scheduler,
                        train_dl,
                        train_len,
                        val_dl,
                        val_len,
                        N_EPOCHS_MASKGRAD,
                        outputPath,
                        SAVE_FILE,
                        zero_list,
                        run_model,
                        criterion_classification,
                        LOSS_LAMBDA,
                        feature_name,
                        TYPE_PROJ,
                        ETA,
                    )

                data_encoder = data_encoder.cpu().detach().numpy()
                data_decoded = data_decoded.cpu().detach().numpy()

                (
                    data_encoder_test,
                    data_decoded_test,
                    class_train,
                    class_test,
                    _,
                    correct_pred,
                    softmax,
                    Ytrue,
                    Ypred,
                    data_encoder_train,
                    data_decoded_train,
                ) = ft.runBestNet(
                    train_dl,
                    test_dl,
                    best_test,
                    outputPath,
                    i,
                    class_len,
                    net,
                    feature_name,
                    test_len,
                )

                data_encoder_train = data_encoder_train.cpu().detach().numpy()
                data_decoded_train = data_decoded_train.cpu().detach().numpy()

                if SEED == seed[-1]:
                    if i == 0:

                        Ypredf = Ypred
                        LP_test = data_encoder_test.detach().cpu().numpy()
                    else:

                        Ypredf = np.concatenate((Ypredf, Ypred))
                        LP_test = np.concatenate(
                            (LP_test, data_encoder_test.detach().cpu().numpy())
                        )

                accuracy_train[s * 4 + i] = class_train
                accuracy_test[s * 4 + i] = class_test

                # silhouette score
                X_encoder = data_encoder[:, :-1]
                labels_encoder = data_encoder[:, -1]
                data_encoder_test = data_encoder_test.cpu().detach()

                data_train[s * 4 + i, 0] = metrics.silhouette_score(
                    X_encoder, labels_encoder, metric="euclidean"
                )

                X_encodertest = data_encoder_test[:, :-1]
                labels_encodertest = data_encoder_test[:, -1]
                # data_test[s * 4 + i, 0] = metrics.silhouette_score(
                #     X_encodertest, labels_encodertest, metric="euclidean"
                # )
                # ARI score

                data_train[s * 4 + i, 1] = metrics.adjusted_rand_score(
                    labels_encoder, labelpredict
                )
                data_test[s * 4 + i, 1] = metrics.adjusted_rand_score(
                    Y_test, data_encoder_test[:, :-1].max(1)[1].detach().cpu().numpy()
                )

                # AMI Score
                data_train[s * 4 + i, 2] = metrics.adjusted_mutual_info_score(
                    labels_encoder, labelpredict
                )
                data_test[s * 4 + i, 2] = metrics.adjusted_mutual_info_score(
                    Y_test, data_encoder_test[:, :-1].max(1)[1].detach().cpu().numpy()
                )

                # UAC Score
                if class_len == 2:
                    data_train[s * 4 + i, 3] = metrics.roc_auc_score(
                        labels_encoder, labelpredict
                    )
                    try:
                        data_test[s * 4 + i, 3] = metrics.roc_auc_score(
                            Y_test,
                            data_encoder_test[:, :-1].max(1)[1].detach().cpu().numpy(),
                        )
                    except ValueError:
                        print(
                            "Only one class present in y_true. ROC AUC score is not defined in that case. Defaulted to 0.0"
                        )
                        data_test[s * 4 + i, 3] = 0.0

                # F1 precision recal
                data_train[s * 4 + i, 4:] = precision_recall_fscore_support(
                    labels_encoder, labelpredict, average="macro", zero_division=0
                )[:-1]
                data_test[s * 4 + i, 4:] = precision_recall_fscore_support(
                    Y_test,
                    data_encoder_test[:, :-1].max(1)[1].numpy(),
                    average="macro",
                    zero_division=0,
                )[:-1]

                # Recupération des labels corects
                correct_prediction += correct_pred

                # Get Top Genes of each class

                #         method = 'Shap'       # (SHapley Additive exPlanation) A nb_samples should be define
                nb_samples = 300  # Randomly choose nb_samples to calculate their Shap Value, time vs nb_samples seems exponential
                #        method = 'Captum_ig'   # Integrated Gradients
                method = "Captum_dl"  # Deeplift
                #        method = 'Captum_gs'  # GradientShap

                if doTopGenes:
                    if i == 0:
                        df_topGenes = ft.topGenes(
                            X,
                            Y,
                            feature_name,
                            class_len,
                            feature_len,
                            method,
                            nb_samples,
                            device,
                            net,
                        )
                        df_topGenes.index = df_topGenes.iloc[:, 0]
                    else:
                        df_topGenes = ft.topGenes(
                            X,
                            Y,
                            feature_name,
                            class_len,
                            feature_len,
                            method,
                            nb_samples,
                            device,
                            net,
                        )
                        df = pd.read_csv(
                            "{}{}_topGenes_{}_{}.csv".format(
                                outputPath, str(TYPE_PROJ_NAME), method, str(nb_samples)
                            ),
                            sep=";",
                            header=0,
                            index_col=0,
                        )
                        df_topGenes.index = df_topGenes.iloc[:, 0]
                        df_topGenes = df.join(df_topGenes.iloc[:, 1], lsuffix="_",)

                    df_topGenes.to_csv(
                        "{}{}_topGenes_{}_{}.csv".format(
                            outputPath, str(TYPE_PROJ_NAME), method, str(nb_samples)
                        ),
                        sep=";",
                    )

            if SEED == seed[0]:
                df_softmax = softmax
                df_softmax.index = df_softmax["Name"]
                # softmax.to_csv('{}softmax.csv'.format(outputPath),sep=';',index=0)
            else:
                softmax.index = softmax["Name"]
                df_softmax = df_softmax.join(softmax, rsuffix="_")
            s += 1

        # little df manip for better display on the GUI
        # store correct column names that are on line 1
        new_header = df_topGenes.iloc[0]
        df_topGenes.drop(
            "topGenes", inplace=True
        )  # remove correct column names from data rows
        df_topGenes.columns = (
            new_header  # change previous column names to the right ones
        )
        df_topGenes.to_csv(
            f"{outputPath}topGenes_for_display.csv", sep=";", index=False
        )
        self.update_topGenes_display(outputPath)
        try:
            df = pd.read_csv(
                "{}Labelspred_softmax.csv".format(outputPath), sep=";", header=0
            )
            data_pd = pd.read_csv(
                "data/" + str(file_name_train[:-12]) + ".csv",
                delimiter=";",
                decimal=",",
                header=0,
                encoding="ISO-8859-1",
            )
        except:
            data_pd = pd.read_csv(
                "data/" + str(self.file_name_test),
                delimiter=";",
                decimal=",",
                header=0,
                encoding="ISO-8859-1",
            )

        proba = df.values[:, 2:].astype(float)

        df.index = df.iloc[:, 0]
        df = df.join(data_pd.T, rsuffix="_", how="right")
        # df.iloc[: , 1:4].to_csv('{}Labelspred_softmax.csv'.format(outputPath),sep=';')

        print("Confidence score of our diagnosis : ")
        print(df_softmax)

        # Reconstruction by using the centers in laten space and datas after interpellation
        center_mean, center_distance = ft.Reconstruction(
            0.2, data_encoder, net, class_len
        )
        self.ax.clear()
        # Do pca,tSNE for encoder data
        if doPCA:
            plt.figure()
            tit = "Latent Space"
            pcafit = self.ShowPcaTsne(
                X, Y, data_encoder_train, center_distance, class_len, tit
            )

            tit = "Latent Space Test"
            test_legends = []
            for _, row in df_softmax.iterrows():
                true_label = int(row["Labels"])
                pred_label = 0 if (row["Proba class 0"] > row["Proba class 1"]) else 1
                test_legends.append(
                    f"""{self.file_name_test[:-3]} - Label pred: {pred_label}, with score {row[f"Proba class {pred_label}"]:.3f}"""
                )
            self.ShowPcaTsne(
                X,
                Y,
                LP_test,
                center_distance,
                class_len,
                tit,
                pcafit,
                test_legends=test_legends,
            )
            self.ax.legend()

        self.canvas.draw()  # update the plot

    def show(self):
        tk.mainloop()


if __name__ == "__main__":
    tls = testLatentSpace()
    tls.show()
