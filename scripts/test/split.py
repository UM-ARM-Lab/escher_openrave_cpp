
line_list = []
with open('env_13_dijkstra_with_dynamics.txt', 'r') as file:
    for line in file:
        line_list.append(line)

for i in range(16):
    with open(str(i) + '.txt', 'w') as file:
        for j in range(82):
            file.write(line_list[i * 82 + j])

