total_count = 0
abnormal_count = 0
with open('env_5_log.txt', 'r') as file:
    for line in file:
        if line[0] == '(':
            total_count += 1
            position1 = line.find('(')
            position2 = line.find(')')
            p1x, p1y, p1theta = line[position1+1: position2].split(',')
            position3 = line.rfind('(')
            position4 = line.rfind(')')
            p2x, p2y, p2theta = line[position3+1: position4].split(',')
            dynamic_cost = float(line.split(' ')[-1])
            if dynamic_cost > 1000:
		print(line)
    #print(abnormal_count * 1.0 / total_count)
