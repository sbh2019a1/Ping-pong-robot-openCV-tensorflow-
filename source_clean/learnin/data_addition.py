import numpy as np
import os

for data_num in range(1000):
    print("iteration:", data_num)
    num = 0
    file_name = '../pingpong/data/data_'+str(num)+'.txt'

    while os.path.isfile(file_name):
        num = num + 1
        file_name = '../pingpong/data/data_' + str(num) + '.txt'

    Address_train = '../pingpong/data_3/data_' + str(data_num) + '.txt'
    load_data = np.loadtxt(Address_train, delimiter='\t')
    np.savetxt(file_name, load_data, fmt='%f', delimiter='\t')