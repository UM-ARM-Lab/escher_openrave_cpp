import os
import numpy as np
import matplotlib.pyplot as plt

for x in range(16):
    if os.path.exists(str(x) + '.txt'):
        data_list = []
        with open(str(x) + '.txt', 'r') as file:
            for line in file:
                line_list = line.strip('\n').strip(' ').split(' ')
	        data_list.append(line_list)
    	data = np.clip(np.array(data_list, dtype=float), 0, 500)
        # print(data.shape)
    	plt.figure(x)
    	plt.imshow(data, cmap='gray')
        # plt.show()
  	plt.savefig(str(x) + '.png')

