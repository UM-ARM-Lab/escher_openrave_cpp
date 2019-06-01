import pickle

def main():
    file = open('../data/ground_truth/transitions_2', 'r')
    all_transitions = pickle.load(file)
    types = {}
    for i in range(10):
        # types[i] = []
        types[i] = 0

    for transition in all_transitions:
        # types[transition['contact_transition_type']].append(transition['feature_vector_contact_part'])
        types[transition['contact_transition_type']] += 1

    for i in range(10):
        # file_handle = open('../data/transitions_type_' + str(i), 'w')
        # print('There are {} transitions of type {}'.format(len(types[i]), i))
        # pickle.dump(types[i], file_handle)
        print('There are {} transitions of type {}'.format(types[i], i))
    

if __name__ == "__main__":
    main()

