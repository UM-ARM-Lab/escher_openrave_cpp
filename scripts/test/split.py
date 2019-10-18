
line_list = []
with open('env_11_dijkstra_once.txt', 'r') as file:
    for line in file:
        line_list.append(line)

for i in range(24):
    with open(str(i) + '_11_flip.txt', 'w') as file:
        for j in range(82):
            file.write(line_list[i * 82 + j])

