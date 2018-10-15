import My_functions


Train_data = open("noisy_train.ssv").read()
Train_data = [item.split() for item in Train_data.split('\n')[3:-1]]
Test_data = open("noisy_test.ssv").read()
Test_data = [item.split() for item in Test_data.split('\n')[3:-1]]
Validation_data = open("noisy_valid.ssv").read()
Validation_data = [item.split() for item in Validation_data.split('\n')[3:-1]]

######################################## Part A ###############################

new_data = My_functions.new_data(Train_data)

new_data_N, new_data_Y = My_functions.split(new_data)

mean_N, cov_N = My_functions.mean_and_cov(new_data_N)

mean_Y, cov_Y = My_functions.mean_and_cov(new_data_Y)

accuracy_ML_train = My_functions.label_ML(mean_N, cov_N, mean_Y, cov_Y, Train_data)
accuracy_ML_test = My_functions.label_ML(mean_N, cov_N, mean_Y, cov_Y, Test_data)

print(' -------Part A-------')
print(' Accuracy of ML on Train Set: ',accuracy_ML_train)
print(' Accuracy of ML on Test Set : ',accuracy_ML_test)
print('')

######################################## Part B ###############################

prior_set1 = Train_data.copy()
prior_set2 = Train_data + Validation_data

accuracy_MAP1_train = My_functions.label_MAP(mean_N, cov_N, mean_Y, cov_Y, Train_data, prior_set1)
accuracy_MAP1_test = My_functions.label_MAP(mean_N, cov_N, mean_Y, cov_Y, Test_data, prior_set1)

accuracy_MAP2_train = My_functions.label_MAP(mean_N, cov_N, mean_Y, cov_Y, Train_data, prior_set2)
accuracy_MAP2_test = My_functions.label_MAP(mean_N, cov_N, mean_Y, cov_Y, Test_data, prior_set2)
print(' -------Part B-------')
print(' Prior : Train Set')
print(' Accuracy of MAP on Train Set: ',accuracy_MAP1_train)
print(' Accuracy of MAP on Test Set : ',accuracy_MAP1_test)
print(' Prior : Train Set + Validation Set')
print(' Accuracy of MAP on Train Set: ',accuracy_MAP2_train)
print(' Accuracy of MAP on Test Set : ',accuracy_MAP2_test)
print('')

######################################## Part C ###########################


probability_N, probability_Y = My_functions.naive_bayes(Train_data)

accuracy_Naive_train = My_functions.label_naive(probability_N, probability_Y, Train_data)
accuracy_Naive_test = My_functions.label_naive(probability_N, probability_Y, Test_data)

print(' -------Part C-------')
print(' Accuracy of Naive on Train Set: ', accuracy_Naive_train )
print(' Accuracy of Naive on Test Set : ', accuracy_Naive_test )
print('')
######################################## Part D ###########################
m_optimum = 0
optimum = 0
print(' -------Part D-------')
print(' Accuracy for different values of M:')
print('                 Train Set           Test Set')
for m in range(0,1000,100):
    probability_N, probability_Y = My_functions.smoothed_naive(Train_data, m)
    p = My_functions.label_naive(probability_N, probability_Y, Validation_data)
    if p > optimum:
        optimum = p
        m_optimum = m

    probability_N, probability_Y = My_functions.smoothed_naive(Train_data, m)
    accuracy_smoothed_train = My_functions.label_naive(probability_N, probability_Y, Train_data)
    accuracy_smoothed_test = My_functions.label_naive(probability_N, probability_Y, Test_data)

    print(' M = ', m ,':',accuracy_smoothed_train ,'   ', accuracy_smoothed_test)


print(' Optimum M = ', m_optimum)
print('')


