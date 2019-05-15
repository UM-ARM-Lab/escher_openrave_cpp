import pickle

def main():
    file_handle = open('../data/transitions/all_transitions', 'r')
    all_transitions = pickle.load(file_handle)
    types = {}
    for i in range(10):
        types[i] = []

    for transition in all_transitions:
        types[transition['contact_transition_type']].append(transition)

    for i in range(10):
        file_handle = open('../data/transitions/transitions_type_' + str(i), 'w')
        print('There are {} transitions of type {}'.format(len(types[i]), i))
        pickle.dump(types[i], file_handle)
    

if __name__ == "__main__":
    main()

