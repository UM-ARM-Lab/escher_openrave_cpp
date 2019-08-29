import IPython

transition_model = {}
theta_index= -9

with open('torso_pose_transition_model_22.5.txt', 'r') as file:
    for line in file:
	if line[0] == 'O':
	    theta_index += 1
	    transition_model[theta_index] = set()
	else:
            dx, dy, dz = line.split(' ')
	    transition_model[theta_index].add((int(dx), int(dy), int(dz)))

compare_n90 = set()
for t in transition_model[-8]:
    compare_n90.add((-t[1], t[0], t[2]))
assert(compare_n90 == transition_model[-4])

compare_0 = set()
for t in transition_model[-8]:
    compare_0.add((-t[0], -t[1], t[2]))
assert(compare_0 == transition_model[0])

compare_90 = set()
for t in transition_model[-8]:
    compare_90.add((t[1], -t[0], t[2]))
assert(compare_90 == transition_model[4])

compare_n675 = set()
for t in transition_model[-7]:
    compare_n675.add((-t[1], t[0], t[2]))
assert(compare_n675 == transition_model[-3])

compare_225 = set()
for t in transition_model[-7]:
    compare_225.add((-t[0], -t[1], t[2]))
assert(compare_225 == transition_model[1])

compare_1125 = set()
for t in transition_model[-7]:
    compare_1125.add((t[1], -t[0], t[2]))
assert(compare_1125 == transition_model[5])


compare_n45 = set()
for t in transition_model[-6]:
    compare_n45.add((-t[1], t[0], t[2]))
assert(compare_n45 == transition_model[-2])

compare_45 = set()
for t in transition_model[-6]:
    compare_45.add((-t[0], -t[1], t[2]))
assert(compare_45 == transition_model[2])

compare_135 = set()
for t in transition_model[-6]:
    compare_135.add((t[1], -t[0], t[2]))
assert(compare_135 == transition_model[6])

compare_n225 = set()
for t in transition_model[-5]:
    compare_n225.add((-t[1], t[0], t[2]))
assert(compare_n225 == transition_model[-1])

compare_675 = set()
for t in transition_model[-5]:
    compare_675.add((-t[0], -t[1], t[2]))
assert(compare_675 == transition_model[3])

compare_1575 = set()
for t in transition_model[-5]:
    compare_1575.add((t[1], -t[0], t[2]))
assert(compare_1575 == transition_model[7])

with open('test.txt', 'w') as file:
    for t in transition_model[3]:
    	file.write('(' + str(t[0]) + ',' + str(t[1]) + ',' + str(t[2]) + ')' + ',')




# IPython.embed()
