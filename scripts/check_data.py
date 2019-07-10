import pickle, IPython
import numpy as np

def main():
    for i in range(12):
        file = open('../data/transitions_type_' + str(i), 'r')
        sampled_X_contact_part = pickle.load(file)
        sampled_X_contact_part = np.array(sampled_X_contact_part)

        file = open('../data/dynopt_result/dataset/dynopt_total_data_' + str(i), 'r')
        original_X = pickle.load(file)

        # file = open('../data/CoM/com_combinations_' + str(i), 'r')
        # sampled_com = pickle.load(file)
        print('\ntransition type: {}'.format(i))
        print('original:\tsampled:')
        for j in range(sampled_X_contact_part.shape[1]):
            print('{:4.2f}\t\t{:4.2f}'.format(np.mean(original_X[:, j + 1]), np.mean(sampled_X_contact_part[:, j])))


if __name__ == "__main__":
    main()