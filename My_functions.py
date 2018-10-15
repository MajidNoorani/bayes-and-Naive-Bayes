import numpy as np
import collections
import math


def new_data(data):
    ########### extracting the feature values in train_set and dedicating numerical values to each one #################
    data_copy = []
    values_complete = []
    values = []
    transposed_train_set = list(map(list, zip(*data)))
    for i in range(len(data[0])):
        values_complete.append(collections.Counter(transposed_train_set[i]))
        values.append(sorted(values_complete[i].keys()))

    new_values = {}
    for i in range(len(values)):
        new_values[i] = []
        for j in range(len(values[i])):
            new_values[i].append(j)
    ##############################################################################################

    for q in range(len(data)):
        a = []
        for i in range(len(values)):

            if data[q][i] in values[i]:
                a.append(new_values[i][values[i].index(data[q][i])])
            else:
                a.append(-1)
        data_copy.append(a)


    return data_copy

def split(new_data):
    new_data_N = []
    new_data_Y = []
    for i in range(len(new_data)):
        if new_data[i][0] == 0:
            new_data_N.append(new_data[i])
        else:
            new_data_Y.append(new_data[i])

    return new_data_N, new_data_Y

def mean_func(new_data):
    # calculation of mean and variance of 2 Gaussian distributions
    mean = []
    new_data_t = list(map(list, zip(*new_data)))

    for i in range(1, len(new_data_t)):
        mean.append(np.mean(new_data_t[i]))

    return mean

def mean_and_cov(new_data):
    mean = mean_func(new_data)
    cov = []

    for r in range(1, len(new_data[1])):
        a = []
        for c in range(1 , len(new_data[1])):
            sum = 0
            for i in range(len(new_data)):
                sum = sum + (new_data[i][r] - mean[r-1])*(new_data[i][c]-mean[c-1])
            a.append(sum/(len(new_data)))
        cov.append(a)

    return mean, cov

def label_ML(mean0, cov0, mean1, cov1, data):

    new_input = new_data(data)

    for i in range(len(mean0)):
        if cov0[i][i] == 0:
            cov0[i][i] = 0.1
        if cov1[i][i] == 0:
            cov1[i][i] = 0.1

    new_label = []
    for i in range(len(new_input)):
        GD0 = (1 / ((2 * math.pi) ** 11)) * (1 / (np.linalg.det(cov0) ** 0.5)) * np.exp(
            (-1 / 2) * (np.matrix(new_input[i][1:]) - np.matrix(mean0)) * np.linalg.inv(cov0) * np.matrix.transpose(np.matrix(new_input[i][1:]) - np.matrix(mean0)))
        GD1 = (1 / ((2 * math.pi) ** 11)) * (1 / (np.linalg.det(cov1) ** 0.5)) * np.exp(
            (-1 / 2) * (np.matrix(new_input[i][1:]) - np.matrix(mean1)) * np.linalg.inv(cov1) * np.matrix.transpose(np.matrix(new_input[i][1:]) - np.matrix(mean1)))

        if GD0 > GD1:
            new_label.append(0)
        else:
            new_label.append(1)

    new_input_t = list(map(list, zip(*new_input)))
    true_label = new_input_t[0]
    T = 0
    for i in range(len(new_label)):
        if (true_label[i] == 1 and new_label[i] == 1) or (true_label[i] == 0 and new_label[i] == 0):
            T = T+1

    accuracy = T/len(new_label)

    return accuracy

def label_MAP(mean0, cov0, mean1, cov1, data, prior_set):

    new_prior = new_data(prior_set)
    new_input = new_data(data)

    for i in range(len(mean0)):
        if cov0[i][i] == 0:
            cov0[i][i] = 0.1
        if cov1[i][i] == 0:
            cov1[i][i] = 0.1


    new_prior_t = list(map(list, zip(*new_prior)))
    prior_know = collections.Counter(new_prior_t[0])

    new_label = []
    for i in range(len(new_input)):
        GD0 = ((prior_know[0])/(prior_know[0]+prior_know[1]))*(1 / ((2 * math.pi) ** 11)) * (1 / (np.linalg.det(cov0) ** 0.5)) * np.exp(
            (-1 / 2) * (np.matrix(new_input[i][1:]) - np.matrix(mean0)) * np.linalg.inv(cov0) * np.matrix.transpose(np.matrix(new_input[i][1:]) - np.matrix(mean0)))
        GD1 = (prior_know[1]/(prior_know[0]+prior_know[1]))*(1 / ((2 * math.pi) ** 11)) * (1 / (np.linalg.det(cov1) ** 0.5)) * np.exp(
            (-1 / 2) * (np.matrix(new_input[i][1:]) - np.matrix(mean1)) * np.linalg.inv(cov1) * np.matrix.transpose(np.matrix(new_input[i][1:]) - np.matrix(mean1)))
        if GD0 > GD1:
            new_label.append(0)
        else:
            new_label.append(1)

    new_input_t = list(map(list, zip(*new_input)))
    true_label = new_input_t[0]
    T = 0
    for i in range(len(new_label)):
        if (true_label[i] == 1 and new_label[i] == 1) or (true_label[i] == 0 and new_label[i] == 0):
            T = T+1

    accuracy = T/len(new_label)

    return accuracy

def naive_bayes(data):

    new_data_N = []
    new_data_Y = []
    for i in range(len(data)):
        if data[i][0] == '0':
            new_data_N.append(data[i])
        else:
            new_data_Y.append(data[i])

    values_complete_total = []
    values_total = []
    transposed_train_set = list(map(list, zip(*data)))
    for i in range(len(data[0])):
        values_complete_total.append(collections.Counter(transposed_train_set[i]))
        values_total.append(sorted(values_complete_total[i].keys()))


    values_complete_N = []
    new_data_N_t = list(map(list, zip(*new_data_N)))
    for i in range(len(new_data_N[0])):
        values_complete_N.append(collections.Counter(new_data_N_t[i]))

    values_complete_Y = []
    new_data_Y_t = list(map(list, zip(*new_data_Y)))
    for i in range(len(new_data_Y[0])):
        values_complete_Y.append(collections.Counter(new_data_Y_t[i]))

    probability_Y = []
    probability_N = []
    for i in range(len(values_total)):
        a = {}
        for j in values_total[i]:
            if j in values_complete_Y[i]:

                a[j] = (values_complete_Y[i][j]/values_complete_Y[0]['1'])
            else:
                a[j] = (0)
        probability_Y.append(a)
    probability_Y[0] = values_complete_total[0]['1'] / len(data)


    for i in range(len(values_total)):
        a = {}
        for j in values_total[i]:
            if j in values_complete_N[i]:

                a[j] = (values_complete_N[i][j]/values_complete_N[0]['0'])
            else:
                a[j] = (0)
        probability_N.append(a)
    probability_N[0] = values_complete_total[0]['0'] / len(data)

    return probability_N ,probability_Y

def label_naive(N_lookupTable, Y_lookupTable, data):

    new_label = []
    for i in range(len(data)):
        Y_probability = Y_lookupTable[0]
        for j in range(1, len(data[0])):
            Y_probability = Y_probability * Y_lookupTable[j][data[i][j]]

        N_probability = N_lookupTable[0]
        for j in range(1, len(data[0])):
            N_probability = N_probability * N_lookupTable[j][data[i][j]]


        if Y_probability > N_probability:
            new_label.append(1)
        else:
            new_label.append(0)



    data_t = list(map(list, zip(*data)))
    true_label = data_t[0]
    T = 0
    for i in range(len(new_label)):
        if (true_label[i] == '1' and new_label[i] == 1) or (true_label[i] == '0' and new_label[i] == 0):
            T = T + 1

    accuracy = T / len(new_label)


    return accuracy

def smoothed_naive(data,m):


    new_data_N = []
    new_data_Y = []
    for i in range(len(data)):
        if data[i][0] == '0':
            new_data_N.append(data[i])
        else:
            new_data_Y.append(data[i])

    values_complete_total = []
    values_total = []
    transposed_train_set = list(map(list, zip(*data)))
    for i in range(len(data[0])):
        values_complete_total.append(collections.Counter(transposed_train_set[i]))
        values_total.append(sorted(values_complete_total[i].keys()))


    values_complete_N = []
    new_data_N_t = list(map(list, zip(*new_data_N)))
    for i in range(len(new_data_N[0])):
        values_complete_N.append(collections.Counter(new_data_N_t[i]))

    values_complete_Y = []
    new_data_Y_t = list(map(list, zip(*new_data_Y)))
    for i in range(len(new_data_Y[0])):
        values_complete_Y.append(collections.Counter(new_data_Y_t[i]))

    probability_Y = []
    probability_N = []
    for i in range(len(values_total)):
        a = {}
        for j in values_total[i]:
            if j in values_complete_Y[i]:

                a[j] = (values_complete_Y[i][j]+(m/len(values_total[i])))/(values_complete_Y[0]['1']+ m)
            else:
                a[j] = 1/len(values_total[i])
        probability_Y.append(a)
    probability_Y[0] = values_complete_total[0]['1'] / len(data)


    for i in range(len(values_total)):
        a = {}
        for j in values_total[i]:
            if j in values_complete_N[i]:

                a[j] = (values_complete_N[i][j]+(m/len(values_total[i])))/(values_complete_N[0]['0'] + m)
            else:
                a[j] = 1/len(values_total[i])
        probability_N.append(a)
    probability_N[0] = values_complete_total[0]['0'] / len(data)

    return probability_N ,probability_Y
