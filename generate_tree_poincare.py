import os
import pickle
import argparse
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def get_map(code2name_path,name2prov_path):
    # code -> name
    with open(code2name_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
    count = 0
    code2name = {}
    for line in lines:
        line = line.strip('\n')
        code2name[str(count)] = line
        count += 1
    
    # name -> prov; prov -> idx
    df_name2prov = pd.read_csv(name2prov_path,encoding='utf-8')
    name2prov = {}
    prov2idx = {}
    count = 0
    for i in range(df_name2prov.shape[0]):
        name = str(df_name2prov['name'][i])
        prov = str(df_name2prov['prov'][i])
        name2prov[name] = prov
        if prov not in prov2idx:
            prov2idx[prov] = count
            count += 1
    
    candidate = ['上海市','南京市','合肥市','杭州市','郑州市','济南市','北京市','天津市','石家庄市','沈阳市','长春市','哈尔滨市','呼和浩特市','太原市','福州市','南昌市','广州市','长沙市','武汉市','南宁市','贵阳市','昆明市','海口市','重庆市','成都市','拉萨市','乌鲁木齐市','西安市','银川市','西宁市','兰州市','香港','澳门','台北市']
    candidate_map = {'杭州市':"Hangzhou",'合肥市':"Hefei",'南京市':"Nanjing",'上海市':"Shanghai",'郑州市':"Zhengzhou",'济南市':"Jinan",'北京市':"Beijing",'天津市':"Tianjin",'太原市':"Taiyuan",'沈阳市':"Shenyang",'长春市':"Changchun",'哈尔滨市':"Harbin",'呼和浩特市':"Hohhot",'西安市':"Xi'an",'银川市':"Yinchuan",'西宁市':"Xining",'兰州市':"Lanzhou",'福州市':"Fuzhou",'重庆市':"Chongqing",'成都市':"Chengdu",'贵阳市':"Guiyang",'南昌市':"Nanchang",'广州市':"Guangzhou",'长沙市':"Changsha",'武汉市':"Wuhan",'石家庄市':"Shijiazhuang",'南宁市':"Nanning",'拉萨市':"Lhasa",'乌鲁木齐市':"Urumqi",'昆明市':"Kunming",'海口市':"Haikou",'香港':"Hong Kong",'澳门':"Macao",'台北市':"Taipei"}

    return code2name, name2prov, prov2idx, candidate, candidate_map

def EuclideanDistance(u,v):
    return float(np.linalg.norm(u-v))

def PoincareDistance(u,v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    euclidean_dist = np.linalg.norm(u-v)
    return float(np.arccosh(1+2*(euclidean_dist**2)/((1-norm_u**2)*(1-norm_v**2))))

def not_optimize_group_boundaries(groups,curve_path,forbidden_pairs,log_path,crosslayer):

    initial_sizes = [len(group) for group in groups]

    plt.figure()
    plt.plot(range(len(initial_sizes)), initial_sizes, label='Before Optimization', marker='o')
    plt.xlabel('Group Index')
    plt.ylabel('Number of Elements in Group')
    plt.legend()
    plt.title('Group Sizes')
    plt.savefig(curve_path)
    initial_ratios = [initial_sizes[i]/initial_sizes[i-1] for i in range(2, len(initial_sizes))]
    initial_ratios = [round(ratio, 3) for ratio in initial_ratios]
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"K Stats:\n")
        f.write(f"Initial ratios: {initial_ratios}\n")
        f.write(f"Total distance: {calculate_total_edge_distance(groups,forbidden_pairs,crosslayer)}\n")
    return groups

def calculate_total_edge_distance(groups,forbidden_pairs,crosslayer):
    total_distance = 0.0
    for i in range(1, len(groups)):
        for city_name, city_data in groups[i]:
            min_dist = float('inf')
            flag = 0
            if i == 1 or i == 2:
                candidate_groups = [groups[i-1]]
            else:
                candidate_groups_index = [index for index in range(max(1,i-crosslayer),i)]
                candidate_groups = [groups[index] for index in candidate_groups_index]
            for prev_group in candidate_groups:
                for prev_city_name, prev_city_data in prev_group:
                    if (city_name,prev_city_name) in forbidden_pairs or (prev_city_name,city_name) in forbidden_pairs:
                        continue
                    dist = PoincareDistance(np.array([city_data[2], city_data[3]]), np.array([prev_city_data[2], prev_city_data[3]]))
                    if dist < min_dist:
                        min_dist = dist
                        flag = 1
            if flag == 1:
                total_distance += min_dist
    return total_distance

def optimize_group_boundaries(groups,curve_path,forbidden_pairs,log_path,crosslayer):

    initial_sizes = [len(group) for group in groups]
    optimized = False
    while not optimized:
        optimized = True
        for i in range(3, len(groups)):
            if len(groups[i]) > 1 and (len(groups[i])-1) >= (len(groups[i-1])+1):
                new_groups = [list(group) for group in groups]
                new_groups[i-1].append(new_groups[i].pop(0))

                new_total_distance = calculate_total_edge_distance(new_groups,forbidden_pairs,crosslayer)

                if new_total_distance < calculate_total_edge_distance(groups,forbidden_pairs,crosslayer):
                    groups = new_groups
                    optimized = False

            if len(groups[i-1]) > 1 and (len(groups[i-1]) - 1)>= len(groups[i-2]):
                new_groups = [list(group) for group in groups]
                new_groups[i].insert(0, new_groups[i-1].pop(-1))

                new_total_distance = calculate_total_edge_distance(new_groups,forbidden_pairs,crosslayer)

                if new_total_distance < calculate_total_edge_distance(groups,forbidden_pairs,crosslayer):
                    groups = new_groups
                    optimized = False

    optimized_sizes = [len(group) for group in groups]
    plt.figure()
    plt.plot(range(len(initial_sizes)), initial_sizes, label='Before Optimization', marker='o')
    plt.plot(range(len(optimized_sizes)), optimized_sizes, label='After Optimization', marker='o')
    plt.xlabel('Group Index')
    plt.ylabel('Number of Elements in Group')
    plt.legend()
    plt.title('Group Sizes Before and After Optimization')
    plt.savefig(curve_path)
    initial_ratios = [initial_sizes[i]/initial_sizes[i-1] for i in range(2, len(initial_sizes))]
    optimized_ratios = [optimized_sizes[i]/optimized_sizes[i-1] for i in range(2, len(optimized_sizes))]
    initial_ratios = [round(ratio, 3) for ratio in initial_ratios]
    optimized_ratios = [round(ratio, 3) for ratio in optimized_ratios]
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"K Stats:\n")
        f.write(f"Initial ratios: {initial_ratios}\n")
        f.write(f"Optimized ratios: {optimized_ratios}\n")
        f.write(f"Total distance: {calculate_total_edge_distance(groups,forbidden_pairs,crosslayer)}\n")
    return groups


def find_k(n, b, a):
    k = sp.symbols('k')
    equation = sp.Eq(b*k**(a-1),n) # 
    k_solution = sp.solve(equation, k)
    positive_k = [sol.evalf() for sol in k_solution if sol.is_real and sol > 0]
    return positive_k

def adjust_k(sorted_city_list, b, k):
    total_length = len(sorted_city_list)
    groups = []
    index = 1 # group0 = (0,0)
    groups.append([sorted_city_list[0]])
    if index < total_length:  # group1 = b 
        groups.append(sorted_city_list[index:index + b])  
        index += b  
    if k > 2:   # group2 = b*(k-1)
        n = b * (k - 1) 
    else:  
        n = b  
    while index < total_length:  # groupn = b*(k-1)*k 
        size = round(n)  
        if index + size > total_length:  
            break 
        groups.append(sorted_city_list[index:index + size])  
        index += size  
        n *= k
    remaining = total_length - index
    last_group_size = n / k

    final_k = k

    if remaining < last_group_size:
        new_k_small = find_k(total_length,b,len(groups)+1)
        new_k_large = find_k(total_length,b,len(groups))
        diff_small = abs(k - new_k_small[0])
        diff_large = abs(k - new_k_large[0])   
 
        if diff_small < diff_large:
            final_k = new_k_small[0]
        else:
            final_k = new_k_large[0]

    return final_k 

def main(data_path,save_path,margin,code2name_path,name2prov_path,k_level,b,curve_path,edge_path,city_path,log_path,forbidden_pairs,crosslayer):
    
    code2name, name2prov, prov2idx, candidate, candidate_map = get_map(code2name_path, name2prov_path)

    with open(data_path, 'rb') as f:
        emb_dict = pickle.load(f)

    colors = []
    labels = []
    x_vals = []
    y_vals = []
    city_list = {}

    x_vals.append(0.0)
    y_vals.append(0.0)
    colors.append(-1)
    labels.append('Origin')
    city_list['Origin'] = [0.0,0.0,0.0,0.0,0.0]

    for k, v in emb_dict.items():
        euclidean_len = EuclideanDistance(v, np.array([0.0, 0.0]))
        if euclidean_len >= 1.0:
            continue
        poincare_len = PoincareDistance(v, np.array([0.0, 0.0]))
        x_vals.append(v[0]*poincare_len/euclidean_len)
        y_vals.append(v[1]*poincare_len/euclidean_len)
        colors.append(prov2idx[name2prov[code2name[str(k)]]])
        labels.append(code2name[str(k)])
        city_list[code2name[str(k)]] = [v[0]*poincare_len/euclidean_len, v[1]*poincare_len/euclidean_len, v[0], v[1], poincare_len]

    sorted_city_list = sorted(city_list.items(),key=lambda item: item[1][4])

    final_k = adjust_k(sorted_city_list,b,k_level)

    groups = []
    # group0 = (0,0)
    index = 1
    groups.append([sorted_city_list[0]])
    # group1 = b 
    if index < len(sorted_city_list):  
        groups.append(sorted_city_list[index:index + b])  
        index += b  
    # group2 = b*(k-1)
    if final_k > 2:  
        n = b * (final_k - 1)  
    else:  
        n = b  
    # groupn = b*(k-1)*k 
    while index < len(sorted_city_list):  
        size = round(n)  
        if index + size > len(sorted_city_list):  
            size = len(sorted_city_list) - index  
        groups.append(sorted_city_list[index:index + size])  
        index += size  
        n *= final_k

    with open(log_path, 'a', encoding='utf-8') as f:
        for i, group in enumerate(groups):  #for check
            group_size = len(group)  
            f.write(f'初始化Group {i} size: {group_size}\n')

    groups = optimize_group_boundaries(groups,curve_path,forbidden_pairs,log_path,crosslayer)
    
    with open(log_path, 'a', encoding='utf-8') as f:
        for i, group in enumerate(groups):  #for check 
            group_size = len(group)  
            f.write(f'优化后Group {i} size: {group_size}\n')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"final_k: {final_k}\n")

    edges_plot = []
    edges_save = []
    city_to_group = {}

    for i in range(1, len(groups)):
        for city_name, city_data in groups[i]:
            min_dist = float('inf')
            closest_city_name = None
            closest_city = None
            flag = 0
            if i == 1 or i == 2:
                candidate_groups = [groups[i-1]]
            else:
                candidate_groups_index = [index for index in range(max(1,i-crosslayer),i)]
                candidate_groups = [groups[index] for index in candidate_groups_index]
            for prev_group in candidate_groups:
                for prev_city_name, prev_city_data in prev_group:
                    if (city_name,prev_city_name) in forbidden_pairs or (prev_city_name,city_name) in forbidden_pairs:
                        continue
                    dist = PoincareDistance(np.array([city_data[2], city_data[3]]), np.array([prev_city_data[2], prev_city_data[3]]))
                    if dist < min_dist:
                        min_dist = dist
                        closest_city_name = prev_city_name
                        closest_city = prev_city_data
                        flag = 1
            if flag == 1:
                edges_plot.append([(city_data[0], city_data[1]), (closest_city[0], closest_city[1])])
                edges_save.append((city_name, closest_city_name))
                city_to_group[city_name] = i   

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 5
    plt.figure(figsize=(15, 15))
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylim(ymin=-1 * margin, ymax=1 * margin)
    ax.set_xlim(xmin=-1 * margin, xmax=1 * margin)

    step = [i for i in range(margin+1)]
    for i in step:
        if i == margin:
            circle = plt.Circle((0, 0), i, color='black', clip_on=False, fill=False, linewidth=0.5, label='')
        else:
            circle = plt.Circle((0, 0), i, color='gray', clip_on=False, fill=False, linewidth=0.2, label='')
        ax.add_artist(circle)

    for i, group in enumerate(groups):
        max_radius = max([city[1][4] for city in group])
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"max_radius: {max_radius}\n")
        if i == len(groups) - 1:
            linewidth = 2.0
        else:
            linewidth = 0.5
        ring = plt.Circle((0, 0), max_radius, color='red', fill=False, linewidth=linewidth)
        ax.add_artist(ring)

    plt.scatter(x_vals, y_vals, c=colors, cmap='Spectral', marker='o', alpha=0.6, s=80)

    for edge in edges_plot:
        point1, point2 = edge
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'k--')
    annotations = []
    for i in range(len(labels)):
        if labels[i] in candidate:
            annotations.append(plt.annotate(candidate_map[labels[i]], (x_vals[i], y_vals[i]), weight='roman', xytext=(-0.2, 0.1), textcoords='offset points', size=10))

    plt.savefig(save_path)

    with open(edge_path, 'w', encoding='utf-8') as f:
        for edge in edges_save:
            f.write(f'{edge[0]} - {edge[1]}\n')

    with open(city_path, 'w', encoding='utf-8') as f:
        for city, group in city_to_group.items():
            f.write(f'{city}: Group {group}\n')

    city_edge_count = defaultdict(int)
    for edge in edges_save:
        city1, city2 = edge
        city_edge_count[city1] += 1
        city_edge_count[city2] += 1

    group_edge_count = defaultdict(list)
    for city, group in city_to_group.items():
        group_edge_count[group].append(city_edge_count[city]-1)
        
    group_stats = {}
    for group, edge_counts in group_edge_count.items():
        if len(edge_counts) > 1:
            mean = np.mean(edge_counts)
            std = np.std(edge_counts)
            group_stats[group] = {'mean': mean, 'std': std}
        else:
            mean = np.mean(edge_counts)
            group_stats[group] = {'mean': mean, 'std': 0}
    
    with open(log_path, 'a', encoding='utf-8') as f:
        for group, stats in group_stats.items():
            f.write(f'Group {group} 的连边数量均值为: {stats["mean"]}, 标准差为: {stats["std"]}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--country', type=str)
    parser.add_argument('--v', type=int)
    parser.add_argument('--b', type=int)
    parser.add_argument('--crosslayer', type=int)
    parser.add_argument('--k', type=float)
    parser.add_argument('--margin', type=int, default=9)
    args = parser.parse_args()
    country = args.country
    v = args.v
    b = args.b
    crosslayer = args.crosslayer
    k = args.k
    threshold = args.threshold
    scale_factor = args.scale_factor
    margin = args.margin

    code2name_path = './exp_emb/utils/CN_code2name.csv'
    name2prov_path = './exp_emb/utils/CN_name2prov.csv'

    data_path = f'./exp_emb/{country}_v{v}_embedding.pkl'
    
    base_save_path = f'./result/{country}_v_{v}/crosslayer_{crosslayer}/'
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)
    save_path =  f"{base_save_path}{country}_poincare_b_{b}_k_{k}_crosslayer_{crosslayer}.png"
    curve_path = f"{base_save_path}{country}_poincare_b_{b}_k_{k}_crosslayer_{crosslayer}_curve.png"
    edge_path = f"{base_save_path}{country}_poincare_b_{b}_k_{k}_crosslayer_{crosslayer}_edge.txt"  
    city_path = f"{base_save_path}{country}_poincare_b_{b}_k_{k}_crosslayer_{crosslayer}_city.txt"
    log_path = f"{base_save_path}{country}_poincare_b_{b}_k_{k}_crosslayer_{crosslayer}_log.txt"

    forbidden_pairs = [
        ('重庆市','沈阳市'),('沈阳市','重庆市'),('成都市','沈阳市'),
        ('沈阳市','成都市'),('贵阳市','哈尔滨市'),('成都市','哈尔滨市'),
        ('成都市','长春市'),('成都市','大连市'),('贵阳市','长春市'),('日喀则市','哈尔滨市') ]

    main(data_path,save_path,margin,code2name_path,name2prov_path,k,b,curve_path,edge_path,city_path,log_path,forbidden_pairs,crosslayer)

# python generate_tree_poincare.py --country CN --v 1 --b 3 --crosslayer 3 --k 2.315
