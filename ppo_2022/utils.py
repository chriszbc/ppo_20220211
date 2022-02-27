

def reformat_action_set() :
    my_file = open('RRM.txt', 'r+')
    actionQ = []
    reline = []

    for i in range(9598):
        content = my_file.readline()
        line = content.replace('function 0 -> node', '')
        line = line.replace('function 1 -> node', '')
        line = line.replace('function 2 -> node', '')
        line = line.replace('function 3 -> node', '')
        line = line.replace('function 4 -> node', '')
        line = line.replace('->', ' ')
        line = line.replace('  ', '')
        line = line.strip()
        if '#' in line:
            continue
        if '------' in line:
            continue
        if '======' in line:
            continue
        reline.append(line)

    file_name = 'action_set.txt'
    with open(file_name, 'a') as file_obj:
        for i in reline:
            file_obj.write(i)
            file_obj.write('\n')



def process_action_set():
    my_file = open("action_set.txt", 'r+')
    action_set = []  # [[[0, 6, 7, 10, 11, 1], [0, 2, 3, 4, 5, 6, 7, 1], [0, 2, 6, 7, 1]], [[...
    function_set = []  # [[[10, 10, 11], [2, 7, 6], [2, 2, 2]], [[...
    action_flow = []
    action_function = []
    for i in range(6000):
        content = my_file.readline()
        # print(content)
        line = content.strip()
        node = line.split(' ')
        flow = []
        if i%4 == 0:
            action_function.append([])
            for n in node:
                if n == '':
                    continue
                flow.append(int(n))
            action_flow.append(flow)
        else:
            for n in node:
                if n == '':
                    continue
                action_function[-1].append(int(n))

    index = 0
    for i in range(500):
        action = []
        function = []
        for j in range(3):
            action.append(action_flow[index])
            function.append(action_function[index])
            index += 1
        action_set.append(action)
        function_set.append(function)

    return action_set, function_set

