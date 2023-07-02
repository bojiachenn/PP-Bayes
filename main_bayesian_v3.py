# coding=UTF-8
import random
import math
import pandas as pd
import numpy as np
from scipy.stats.contingency import margins
import time
import hashlib
from collections import defaultdict

global attribute_num
global HashToAttr_tab  # 轉換用
HashToAttr_tab = {}


# region Bayes Network
def DParentSet(S, fa, k):
    Pi = []

    for i in range(k, 0, -1):
        Pi_1 = set_or_operator(dp[i - 1], fa)
        Pi_2 = dp[i]
        if Conditional_entropy(Pi_1, S) >= Conditional_entropy(Pi_2, S):
            dp[i] = Pi_1.copy()
        else:
            dp[i] = Pi_2.copy()
        Pi.extend(dp[i])
        Pi = list(set(Pi))
    return Pi


def set_or_operator(set, node):
    a = set.copy()
    if len(a) == 0:
        a.append(node)
    else:
        try:
            str.isdigit()
        except:
            a.append(node)
    return a


def Conditional_entropy(chosen_parent, new_node):
    CE = 0
    d_i_in_new_node = np.unique(data[new_node]).tolist()  # different index in new_node
    for ii in chosen_parent:
        d_i_in_chosen_parent = np.unique(data[ii]).tolist()  # different index in chosen_parent

        # 直:different chosen_parent index, 橫: different new_node index
        each_called_data = np.zeros((len(d_i_in_chosen_parent), len(d_i_in_new_node)),
                                    dtype=np.int)  # different attributes in new_node
        if ii != new_node:
            for num in range(num_of_data):
                each_called_data[d_i_in_chosen_parent.index(data[ii][num])][
                    d_i_in_new_node.index(data[new_node][num])] += 1
            # print(each_called_data)
            for j in range(len(d_i_in_new_node)):
                sum = 0
                for k in range(len(d_i_in_chosen_parent)):
                    sum += each_called_data[k][j]

                a = 0
                for k in range((len(d_i_in_chosen_parent))):
                    if each_called_data[k][j] != 0:
                        a += -1 * (each_called_data[k][j] / sum * math.log2(each_called_data[k][j] / sum))
                CE += (sum / np.sum(each_called_data) * a)
    return CE


def exponential(S, Pi, O, attribute_num, pb):
    scores = []
    for i in Pi:
        scores.append(Conditional_entropy([i], S))
    scores = scores / np.linalg.norm(scores, ord=1)
    return [S, [np.random.choice(Pi, 1, p=scores)[0]]]


def ABN(data, attribute, pb):  # pb:privacy_budget
    ran = random.randint(0, attribute_num - 1)
    ran = attribute_num - 1 # 欲成為Target之attribute index
    print("Target(ran):", attribute[ran])
    N = []
    V = []
    N.append(attribute[ran])
    V.append(attribute[ran])
    fa = attribute[ran]

    for i in range(1, attribute_num):
        Pi = []
        Omega = []
        Pi.extend(DParentSet(attribute[i], fa, i))
        Omega.extend([attribute[i], Pi])
        N.append(exponential(attribute[i], Pi, Omega, attribute_num, pb))
        # print("--------------------")
        fa = attribute[i]
    return N


# endregion


class Graph:
    def __init__(self, arg_nodes):
        self.nodes = arg_nodes
        # adjlit is implemented as a dictionary. The default dictionary would create an empty
        # list as a default (value) for the nonexistent keys.
        self.adjlist = defaultdict(list)  # Stores the graph in an adjacency list
        self.incoming_edge_count = []  # For a node, it stores the number of incoming edges.
        self.topo_order = []  # Stores the nodes in lexical topologically sorted order.
        self.zero_incoming_edge_node = []  # Stores the nodes that have zero incoming edges.
        self.path_list = defaultdict(list)

    # Create an adjacency list for storing the edges
    def AddEdge(self, src, dst):
        self.adjlist[src].append(dst)
        self.incoming_edge_count[dst] += 1

    def TopologicalSort(self):

        for node in range(self.nodes):
            if self.incoming_edge_count[node] == 0:
                self.zero_incoming_edge_node.append(node)

        while self.zero_incoming_edge_node:
            self.zero_incoming_edge_node.sort()
            src = self.zero_incoming_edge_node.pop(0)

            # Iterate through the adjacency list
            if src in self.adjlist:
                for node in self.adjlist[src]:
                    self.incoming_edge_count[node] -= 1
                    if self.incoming_edge_count[node] == 0:
                        self.zero_incoming_edge_node.append(node)

            self.topo_order.append(src)
        return dict(self.adjlist)


def Create_RouteTable(form):
    cypher_Attr = list(set([i.Row_Attr for i in form] + [i.Col_Attr for i in form]))
    # print("Encrypted Attr:", [HashToAttr_tab[i] for i in cypher_Attr])
    # print("Hash-Attr對照表", HashToAttr_tab)

    g = Graph(len(cypher_Attr))
    g.incoming_edge_count = [0] * len(cypher_Attr)
    for i in form:
        # print(HashToAttr_tab[i.Row_Attr], HashToAttr_tab[i.Col_Attr])
        g.AddEdge(cypher_Attr.index(i.Col_Attr), cypher_Attr.index(i.Row_Attr))
    route_tab = g.TopologicalSort()
    for key in list(route_tab):
        value = [cypher_Attr[j] for j in route_tab[key]]
        route_tab[cypher_Attr[key]] = value
        route_tab.pop(key)
    return route_tab


class Form:
    def __init__(self, Form_Row_Attr, Form_Col_Attr, Form_Row, Form_Col):
        self.Row_Attr = Form_Row_Attr
        self.Col_Attr = Form_Col_Attr
        self.Row_index = Form_Row
        self.Col_index = Form_Col
        self.Prob_form = [['' for col in range(len(self.Col_index))] for row in range(len(self.Row_index))]

    def PrintForm(self):
        temp = []
        for i in range(len(self.Row_index)):
            for j in range(len(self.Col_index)):
                temp.append([self.Col_index[j], self.Row_index[i], self.Prob_form[i][j]])
        df = pd.DataFrame(np.array(temp), columns=[self.Col_Attr, self.Row_Attr, 'Probability'])
        print(pd.crosstab(df[self.Col_Attr], df[self.Row_Attr], values=df.Probability, aggfunc=np.sum), "\n")


def PRG(seed):
    s = hashlib.sha256()
    s.update(str(seed).encode("utf-8"))
    a_string = int(s.hexdigest(), 16)
    return int(str(a_string)[:10])


def Chr_to_int(char):
    temp_int_chr = ''
    for i in char:
        temp_int_chr = temp_int_chr + str(ord(i))
    return int(temp_int_chr)


def Create_prob_form(data, N, flag=0):
    Form_list = []
    total_data = data.shape[0]
    for i in range(1, attribute_num):
        S_i = N[i][0]  # different index in new_node
        Pi_i = N[i][1]  # different index in chosen_parent
        Pi = "".join(Pi_i)

        d_i_in_new_node = np.unique(data[S_i]).tolist()  # different index in new_node
        d_i_in_chosen_parent = np.unique(data[Pi]).tolist()  # different index in chosen_parent

        temp_form = Form(S_i, Pi, d_i_in_new_node, d_i_in_chosen_parent)
        temp_form.Prob_form = np.zeros((len(d_i_in_new_node), len(d_i_in_chosen_parent)),
                                       dtype=int)  # different attributes in new_node

        for num in range(total_data):
            temp_form.Prob_form[d_i_in_new_node.index(data[S_i][num])][d_i_in_chosen_parent.index(data[Pi][num])] += 1
        temp_form.Prob_form = temp_form.Prob_form / num_of_data

        x, y = margins(temp_form.Prob_form)
        x = x.tolist()

        for j in range(len(temp_form.Row_index)):
            for k in range(len(temp_form.Col_index)):
                if flag == 1:
                    temp_form.Prob_form[j][k] = None
                # print(temp_form.Prob_form[j][k], x[j][0], temp_form.Prob_form[j][k] / x[j][0])
                temp_form.Prob_form[j][k] = format(temp_form.Prob_form[j][k] / x[j][0], '.10f')
        Form_list.append(temp_form)
    return Form_list


def Split_Form(form, S):
    Form_1 = []
    s0 = PRG(S)
    s1 = PRG(s0)
    for i in form:
        temp_form = Form(i.Row_Attr, i.Col_Attr, i.Row_index, i.Col_index)
        for j in range(len(i.Row_index)):
            for k in range(len(i.Col_index)):
                temp_str = PRG(int(str(s1) + str(Chr_to_int(i.Col_Attr)) + str(Chr_to_int(i.Row_index[j])) + str(
                    Chr_to_int(i.Col_index[k]))))
                temp_form.Prob_form[j][k] = str(temp_str)
        # temp_form.PrintForm()
        Form_1.append(temp_form)
    return Form_1


def Minus_Form(BN_form, Form_1):
    Form_2 = []
    s0 = PRG(S)
    for i in range(len(Form_1)):
        s = hashlib.sha256()
        s.update(Form_1[i].Row_Attr.encode("utf-8"))
        temp_Row_Attr = s.hexdigest()

        s = hashlib.sha256()
        s.update(Form_1[i].Col_Attr.encode("utf-8"))
        temp_Col_Attr = s.hexdigest()
        temp_Row_index = []
        temp_Col_index = []

        HashToAttr_tab.update(
            {temp_Row_Attr: Form_1[i].Row_Attr, temp_Col_Attr: Form_1[i].Col_Attr})  # Attribute變成Hash對照表

        for j in Form_1[i].Row_index:
            temp_Row_index.append(
                str(Chr_to_int(j) ^ PRG(int(str(s0) + str(Chr_to_int(j)) + str(Chr_to_int(Form_1[i].Row_Attr))))))
        for j in Form_1[i].Col_index:
            temp_Col_index.append(
                str(Chr_to_int(j) ^ PRG(int(str(s0) + str(Chr_to_int(j)) + str(Chr_to_int(Form_1[i].Col_Attr))))))

        temp_form = Form(temp_Row_Attr, temp_Col_Attr, list(temp_Row_index), list(temp_Col_index))

        for j in range(len(Form_1[i].Row_index)):
            for k in range(len(Form_1[i].Col_index)):
                temp = str(BN_form[i].Prob_form[j][k]).split(".")[1]
                while len(temp) < 10:
                    temp = temp + '0'
                # print(temp, "-", Form_1[i].Prob_form[j][k], "=", int(temp) - int(Form_1[i].Prob_form[j][k]))
                temp_form.Prob_form[j][k] = str(int(temp) - int(Form_1[i].Prob_form[j][k]))
        # temp_form.PrintForm()
        Form_2.append(temp_form)
    return Form_2


class CSU(Form):
    def __init__(self, file):
        self.seed  # MO assigned
        self.Form  # MO assigned
        self.share  # MO assigned
        self.prob_Form_1 = Split_Form(self.Form, self.seed)
        self.route_dict = Create_RouteTable(self.prob_Form_1)
        self.target = list(self.route_dict.keys())[0]
        self.user_file = file
        self.attr = []
        self.index = []
        self.prob_a = []
        self.prob_b = []

    def formed(self):
        s0 = PRG(self.seed)
        user_Attr = []
        user_Index = []
        self.path=[]

        file = open(self.user_file, 'r')
        lines = file.readlines()
        for line in lines:
            info = line.replace('\n','').split(',')
            self.attr.append(info[0])
            self.index.append(info[1])

            s = hashlib.sha256()
            s.update(info[0].encode("utf-8"))
            user_Attr.append(s.hexdigest())

            user_Index.append(str(
                Chr_to_int(info[1]) ^ PRG(int(str(s0) + str(Chr_to_int(info[1])) + str(Chr_to_int(info[0]))))))
        file.close()
        for i in self.attr:
            self.path.extend(Find_path(i, self.target, self.route_dict))

        return user_Attr, user_Index


class CSP(Form):
    def __init__(self, query):
        self.prob_Form_2  # MO assigned
        self.share  # MO assigned
        self.route_dict = Create_RouteTable(self.prob_Form_2)
        self.user_Attr = query[0]
        self.user_Index = query[1]
        self.target = list(self.route_dict.keys())[0]
        self.path = Find_path(self.user_Attr, self.target, self.route_dict)
        self.prob_a = []
        self.prob_b = []


def Find_path(end, start, graph, path=[]):
    path = [start] + path
    if start == end:
        return [path]
    paths = []
    print(graph)
    # print(start, list(Graph.keys()))
    if start not in list(graph.keys()):
        return [path]
    for node in graph[start]:
        if node not in path:
            newpaths = Find_path(start, node, graph, path)
            for newpath in newpaths:
                if end in newpath:
                    paths.append(newpath)
    # print(paths)
    return paths


def get_Prob(Form, next_Attr, Attr, Index=None):
    prob = []
    for i in Form:
        if i.Row_Attr == Attr and i.Col_Attr == next_Attr:
            if Index != None:
                for j in range(len(i.Col_index)):
                    prob.append(i.Prob_form[i.Row_index.index(Index)][j])
                    # print("1",prob)
            else:
                prob.extend(i.Prob_form)
                # print("2",prob)

    return prob



# region Secure Multiplication with secret sharing
digits_20 = 10000000000000000000
digits_10 = 1000000000

def generate_shares():  # only MO can apply
    a = random.randint(digits_10, 10 * digits_10 - 1)
    a_1 = random.randint(digits_10, 10 * digits_10 - 1)
    a_2 = a - a_1

    b = random.randint(digits_10, 10 * digits_10 - 1)
    b_1 = random.randint(digits_10, 10 * digits_10 - 1)
    b_2 = b - b_1

    ab = a * b
    ab_1 = random.randint(digits_20, 10 * digits_20 - 1)
    ab_2 = ab - ab_1
    share_1 = [a_1, b_1, ab_1]
    share_2 = [a_2, b_2, ab_2]
    return share_1, share_2


def get_ep(x, y, share):
    e = x - share[0]
    p = y - share[1]
    return e, p


def MUL(share, e, p, flag=0):  # flag值 表示 CSU為1、CSP為2
    a = share[0]
    b = share[1]
    ab = share[2]
    if flag == 1:
        return p * a + e * b + ab
    elif flag == 2:
        return p * a + e * b + ab + e * p
    return False


# endregion


def Round_Trick(xy1, xy2):
    # Provider
    r_head = random.randint(digits_10, 10 * digits_10 - 1)
    r_prime = random.randint(digits_10/10, digits_10 - 1)
    r = int(str(r_head) + "0" + str(r_prime))

    r_2 = random.randint(digits_20, 10 * digits_20 - 1)
    r_1 = r - r_2       # Send to User

    r_head_1 = random.randint(digits_10, 10 * digits_10 - 1)  # Send to User
    r_head_2 = r_head - r_head_1
    # print("head",r_head_1,r_head_2,r_head)
    z_1 = xy1 + r_1     # User
    z_2 = xy2 + r_2     # Provider, send to User

    z = z_1 + z_2       # REC

    # User
    first = int(str(z)[:10])
    first_1 = random.randint(digits_10, 10 * digits_10 - 1)
    first_2 = first - first_1   # Send to Provider
    # P.S. first > r_head
    xy_b1 = first_1 - r_head_1  # User
    xy_b2 = first_2 - r_head_2  # Provider
    # print("first",first_1 , first_2, first)
    return xy_b1, xy_b2

def secure_comparison(a, b):
    x_1 = int(a[0])
    x_2 = int(b[0])
    y_1 = int(a[1])
    y_2 = int(b[1])

    l = random.randint(digits_10, 10 * digits_10 - 1)
    l_1 = random.randint(digits_10, 10 * digits_10 - 1)
    l_2 = l - l_1

    s_1 = x_1 + l_1
    s_2 = x_2 + l_2

    h_1 = y_1 + l_1
    h_2 = y_2 + l_2

    s = s_1 + s_2
    h = h_1 + h_2
    # print("s:",s,"h",h)
    if s > h:
        # print("0")
        return 0
    else:
        # print("1")
        return 1


if __name__ == "__main__":

    start = time.time()

    # region Initialize

    # data(小貝式)
    attribute = ["Sickness", "isSmoke", "Age", "Gender", "ParentSmoked"]
    data = pd.read_csv("data.txt", names=attribute, engine='python', sep=r'\s*,\s*', na_values="?", dtype=str)

    # adult.data(1000筆)
    attribute = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
              "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
              "Hours per week", "Country", "Target"]
    data = pd.read_csv("adult.data(1000).txt", names=attribute, engine='python', sep=r'\s*,\s*', na_values="?", dtype=str)

    attribute_num = len(attribute)
    num_of_data = data.shape[0]
    dp = []
    for i in range(attribute_num + 1):
        dp.append([])
    # endregion

    # N = ABN(data, attribute, 0.5)  # Create Bayes Network
    N= ['Target', ['Workclass', ['Target']], ['fnlwgt', ['Workclass']], ['Education', ['fnlwgt']],
              ['Education-Num', ['fnlwgt']], ['Martial Status', ['fnlwgt']], ['Occupation', ['fnlwgt']],
              ['Relationship', ['Education-Num']], ['Race', ['fnlwgt']], ['Sex', ['Education-Num']],
              ['Capital Gain', ['Education-Num']], ['Capital Loss', ['Occupation']], ['Hours per week', ['fnlwgt']],
              ['Country', ['fnlwgt']], ['Target', ['Hours per week']]]

    print("Bayes Network:", N)
    end = time.time()
    print("Bayes Time:", end - start)
#region
    start = time.time()
    BN_Form = Create_prob_form(data, N)  # Create Bayes Network Probability Form
    # for i in BN_Form:
    #     i.PrintForm()
    empty_Form = Create_prob_form(data, N, 1)
    # for i in empty_Form:
    #     i.PrintForm()

    S = random.randint(1000000000, 9999999999)  # Create seed S
    # print("Seed(S):", S)
    Form_1 = Split_Form(BN_Form, S)  # Create [Prob]1
    # for i in Form_1:
    #     i.PrintForm()
    Form_2 = Minus_Form(BN_Form, Form_1)  # Create [Prob]2 = Prob - [Prob]1
    # for i in Form_2:
    #     i.PrintForm()
    end = time.time()
    # print("拆分表格 Time:", end-start)

    CSU.seed = S  # Assigned Seed to CSU
    CSU.Form = empty_Form  # Assigned Empty_Form to CSU
    CSP.prob_Form_2 = Form_2  # Assigned [Prob]2 to CSP

    shares = generate_shares()  # Generate (a, b, ab)
    CSU.share = shares[0]  # Assigned [a]1, [b]1, [ab]1 to CSU
    CSP.share = shares[1]  # Assigned [a]2, [b]2, [ab]2 to CSP

    user_data = "user_data.txt"
    user = CSU(user_data)
    user_query = user.formed()
    provider = CSP(user_query)  # Get user's data
    # print("User data:\n     -Attr:", user_query[0], "\n     -Index:", user_query[1])

    print("path:", user.path)
#endregion
    start = time.time()
    for p in range(len(user.path)):
        index = user.path[p].index(user.attr)
        path_u = user.path[p][index:]
        # print(path_u)
        path_p = provider.path[p][index:]
        user.prob_a = get_Prob(user.prob_Form_1, path_u[1], path_u[0], user.index)
        provider.prob_a = get_Prob(provider.prob_Form_2, path_p[1], path_p[0], provider.user_Index)
        test_prob_A = get_Prob(BN_Form, path_u[1], path_u[0], user.index)

        for i in range(2, len(path_u)):  # 回推路徑，計算機率
            user_cur = path_u[i - 1]
            user.prob_b = get_Prob(user.prob_Form_1, path_u[i], user_cur)
            provider_cur = path_p[i - 1]
            provider.prob_b = get_Prob(provider.prob_Form_2, path_p[i], provider_cur)
            p1 = []
            p2 = []
            np1 = []
            np2 = []
            test = []
            test_prob_B = get_Prob(BN_Form, path_u[i], user_cur)
            # print(test_prob_B)

            for j in range(len(user.prob_a)):  # Secure Multiplication
                for k in range(len(user.prob_b[j])):
                    user_ep = get_ep(int(user.prob_a[j]), int(user.prob_b[j][k]), user.share)
                    provider_ep = get_ep(int(provider.prob_a[j]), int(provider.prob_b[j][k]), provider.share)
                    e = user_ep[0] + provider_ep[0]
                    p = user_ep[1] + provider_ep[1]

                    p1.append(MUL(user.share, e, p, 1))
                    p2.append(MUL(provider.share, e, p, 2))
                    test.append(test_prob_A[j]*test_prob_B[j][k])

            interval = 0
            for j in user.prob_Form_1:  # 計算next_node有幾個index
                if j.Col_Attr == path_u[i]:
                    interval = len(j.Col_index)
                    # print(path_u[i], j.Col_index)
            t = []
            for j in range(interval):   # 兩兩相乘好的機率相加
                temp1 = 0
                temp2 = 0
                test_a = 0
                for k in range(j, len(p1), interval):
                    temp1 += p1[k]
                    temp2 += p2[k]
                    test_a += test[k]
                np1.append(temp1)
                np2.append(temp2)
                t.append(test_a)

            print("After MUL:")
            print("xy1:",np1)
            print("xy2:",np2)
            test_prob_A = t
            print(test_prob_A)
            xy1 = []
            xy2 = []
            for b in range(len(np1)):
                temp = Round_Trick(np1[b], np2[b])          # Round_Trick 取得前10bits
                xy1.append(temp[0])
                xy2.append(temp[1])

            user.prob_a = xy1
            provider.prob_a = xy2
    end = time.time()
    # print("算條件機率 Time:", end-start)

    print("After Rounding trick", user.target)
    print("(bar)xy1:", user.prob_a)
    print("(bar)xy2:", provider.prob_a)
    print(int(user.prob_a[0])+int(provider.prob_a[0])+int(user.prob_a[1])+int(provider.prob_a[1]))
    target_index = secure_comparison(user.prob_a, provider.prob_a)
    print(test_prob_A)
    for i in user.prob_Form_1:
        if i.Row_Attr == user.path[0][-2]:
            print(i.Col_index[target_index])
