import pickle, IPython, os, math, shutil, getopt, sys
import numpy as np
import matplotlib.pyplot as plt

NUM_ENVIRONMENT_PER_TYPE = 50


def main():
    environment_type = None
    try:
        inputs, _ = getopt.getopt(sys.argv[1:], "e:", ['environment_type'])

        for opt, arg in inputs:
            if opt == '-e':
                environment_type = int(arg)

    except getopt.GetoptError:
        print('usage: -e: [environment_type]')
        exit(1)

    colors = {0:'black',
              1:'brown',
              2:'red',
              3:'orange',
              4:'gold',
              5:'green',
              6:'cyan',
              7:'blue',
              8:'purple',
              9:'pink'}

    for environment_index in range(1):
        if os.path.exists('../data/medium_dataset_normal_wall/dynamic_cost_plus_type_' + str(environment_type) + '_' + str(environment_index)):
            print('process data in file dynamic_cost_plus_type_{}_{}'.format(environment_type, environment_index))
            with open('../data/medium_dataset_normal_wall/dynamic_cost_plus_type_' + str(environment_type) + '_' + str(environment_index), 'r') as file:
                data = pickle.load(file)
                p1_list = sorted(data.keys(), key=lambda element: (element[0], element[1], element[2]))
                for p1i, p1 in enumerate(p1_list):
                    p2_list = sorted(data[p1].keys(), key=lambda element: (element[0], element[1], element[2]))
                    for p2i, p2 in enumerate(p2_list):
                        plt.figure(environment_index * 10000 + p1i * 100 + p2i)
                        if len(data[p1][p2]) >= 3:
                            print(str(environment_type) + '_' + str(environment_index) + '_' + str(p1[0]) + str(p1[1]) + str(p1[2]) + '_' + str(p2[0]) + str(p2[1]) + str(p2[2]))
                        for transition_type in data[p1][p2].keys():
                            clipped = np.clip(data[p1][p2][transition_type][:,6], 0, 6000)
                            hist, _ = np.histogram(clipped, bins=np.arange(0, 6100, 100))
                            plt.plot(range(60), hist, '-o', label=str(transition_type), color=colors[transition_type])
                        plt.title('environment type: {} environment index: {} p1: {} p2: {}'.format(environment_type, environment_index, str(p1[0]) + str(p1[1]) + str(p1[2]), str(p2[0]) + str(p2[1]) + str(p2[2])))
                        plt.xlabel('dynamic cost')
                        plt.ylabel('number of transitions')
                        plt.legend()
                        plt.savefig('../data/test_plus_type/{}.png'.format(str(environment_type) + '_' + str(environment_index) + '_' + str(p1[0]) + str(p1[1]) + str(p1[2]) + '_' + str(p2[0]) + str(p2[1]) + str(p2[2])))
                        
                        

if __name__ == '__main__':
    main()

