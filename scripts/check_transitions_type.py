import pickle, IPython


def main():
    file = open('../data/test_dataset/transitions_10', 'r')
    all_transitions = pickle.load(file)
    prev_environment_index = 0
    types = {}
    for i in range(10):
        types[i] = 0
    for transition in all_transitions:
        if transition['environment_index'] != prev_environment_index:
            print('environment index: {}'.format(prev_environment_index))
            for i in range(10):
                print('There are {} transitions of type {}'.format(types[i], i))
            total = types[0] + types[1] + types[2] + types[3] + types[4] + types[5] + types[6] + types[7] + types[8] + types[9]
            print('\nTotal: {}'.format(total))
            IPython.embed()
            types = {}
            for i in range(10):
                types[i] = 0
            prev_environment_index = transition['environment_index']
        types[transition['contact_transition_type']] += 1

    print('environment index: {}'.format(prev_environment_index))
    for i in range(10):
        print('There are {} transitions of type {}'.format(types[i], i))
    total = types[0] + types[1] + types[2] + types[3] + types[4] + types[5] + types[6] + types[7] + types[8] + types[9]
    print('\nTotal: {}'.format(total))


if __name__ == "__main__":
    main()

