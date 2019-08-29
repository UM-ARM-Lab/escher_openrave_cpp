import os
import numpy as np
import matplotlib.pyplot as plt

for x in range(30):
    for y in range(12):
        for theta in range(24):
            location_id = str(x) + '_' + str(y) + '_' + str(theta)
            """
            if os.path.exists(location_id + '_ground.txt'):
    		data_list = []
    		with open(location_id + '_ground.txt', 'r') as file:
        	    for line in file:
            		line_list = line.strip('\n').strip(' ').split(' ')
            	        data_list.append(line_list)
    		data = np.array(data_list, dtype=float)
    		# print(data.shape)
    		plt.figure((x * 10000 + y * 100 + theta) * 2)
    		plt.imshow(data, cmap='gray')
    		plt.savefig(location_id + '_ground.png')
            """
            if os.path.exists(location_id + '_wall.txt'):
                data_list = []
                with open(location_id + '_wall.txt', 'r') as file:
                    for line in file:
                        line_list = line.strip('\n').strip(' ').split(' ')
                        data_list.append(line_list)
                data = np.array(data_list, dtype=float)
                # print(data.shape)
                plt.figure((x * 10000 + y * 100 + theta) * 2 + 1)
                plt.imshow(data, cmap='gray')
                plt.savefig(location_id + '_wall.png')

