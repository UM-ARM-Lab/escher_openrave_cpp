import pickle

def main():
    file = open('../data/ground_truth_0.05/transitions_6', 'r')
    all_transitions = pickle.load(file)
    types = {}
    for i in range(10):
        # types[i] = []
        types[i] = 0

    for transition in all_transitions:
        # types[transition['contact_transition_type']].append(transition['feature_vector_contact_part'])
        types[transition['contact_transition_type']] += 1
        # if transition['contact_transition_type'] != 0 and transition['contact_transition_type'] != 3:
        #     print(transition['contact_transition_type'])
        #     print(transition['p2'])
        #     print(transition['normalized_init_l_leg'])
        #     print(transition['normalized_init_r_leg'])
        #     print(transition['normalized_init_l_arm'])
        #     print(transition['normalized_init_r_arm']) 
        #     raw_input()

    for i in range(10):
        # file_handle = open('../data/transitions_type_' + str(i), 'w')
        # print('There are {} transitions of type {}'.format(len(types[i]), i))
        # pickle.dump(types[i], file_handle)
        print('There are {} transitions of type {}'.format(types[i], i))
    

if __name__ == "__main__":
    main()

