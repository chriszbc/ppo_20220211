import numpy as np

def ReadingActionSet():
    my_file = open('results_123/RRM_result_t_2.txt', 'r+')
    actionQ = []
    reline = []


    for i in range(4806):
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

    file_name = 'results_123/action_set_2.txt'
    with open(file_name, 'a') as file_obj:
        for i in reline:
            file_obj.write(i)
            file_obj.write('\n')

def process2():
    my_file = open('results_123/action_set_123.txt', 'r+')
    state_total = []
    actionset = []
    service_num = 1

    for r in range(7):
        contents = []
        uni = []
        if r == 0:
            my_file = open('results_123/action_set_1.txt', 'r+')
            service_num = 1
        if r == 1:
            my_file = open('results_123/action_set_2.txt', 'r+')
            service_num = 1
        if r == 2:
            my_file = open('results_123/action_set_3.txt', 'r+')
            service_num = 1
        if r == 3:
            my_file = open('results_123/action_set_12.txt', 'r+')
            service_num = 2
        if r == 4:
            my_file = open('results_123/action_set_13.txt', 'r+')
            service_num = 2
        if r == 5:
            my_file = open('results_123/action_set_23.txt', 'r+')
            service_num = 2
        if r == 6:
            my_file = open('results_123/action_set_123.txt', 'r+')
            service_num = 3

        for i in range(20000):
            content = my_file.readline()
            # print(content)
            line = content.strip()
            node = line.split(' ')

            if node[0] == "Mapping":
                break

            j = int(i/service_num)
            if i % service_num == 0:
                contents.append([])
                # contents[j].append(node)
            for n in node:
                contents[j].append(int(n))

            for c in contents:
                if c not in uni:
                    uni.append(c)

        state_total.append(uni)

    for a in state_total:
        print(len(a))
        actionset.extend(a)

    # print(actionset)
    print(len(actionset))
    print(actionset)


def process():
    my_file = open('results_123/map123.txt', 'r+')
    service_line = []
    reflect = []
    reflect_ = []
    for i in range(10000):
        j = int(i / 6)
        content = my_file.readline()
        # print(content)
        line = content.strip()
        if line == '':
            break
        node = line.split(' ')

        # print(node)
        # actionQ.append([int(x) for x in node])
        if i % 6 == 0:
            service_line.append([])
            reflect.append([])
            for a in node:
                service_line[j].append(int(a))
        else:
            for a in node:
                reflect[j].append(int(a))
    for i in range(len(reflect)):
        if i % 3 == 0:
            reflect_.append([])
        reflect_[int(i/3)].extend(reflect[i])

    # print(service_line)
    # print(reflect)
    # b = np.array(service_line).reshape(500, 10)
    # b1 = b.tolist()
    # r = [[] for i in range(500)]
    # for i in range(len(reflect)):
    #     r[int(i/10)].append(reflect[i])
    # print(b1)
    print(reflect_)
    print(len(reflect))


if __name__ == '__main__':
    # ReadingActionSet()
    process()
