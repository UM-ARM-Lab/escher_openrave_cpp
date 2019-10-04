lines = []
with open('env_11_pred_cost.txt', 'r') as file:
    for line in file:
        lines.append(line)
lines.sort()


with open('env_11_pred_cost_sorted.txt', 'w') as file:
    for line in lines:
        file.write(line)
