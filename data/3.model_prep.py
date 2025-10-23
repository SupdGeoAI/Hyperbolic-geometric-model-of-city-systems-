import pickle

def load_data_as_graph(path):
    edges = []

    with open(path,'r',encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split("\t")
        o_id = str(parts[0])
        d_id = str(parts[1])
        num = float(parts[2])
        edges.append((o_id,d_id,{'weight':num}))
    
    return edges

if __name__ == '__main__':
    data_paths = [
        './align_data/CN_long-term_2018-19.txt',
        './align_data/CN_long-term_2019-20.txt',
        './align_data/CN_long-term_2020-21.txt',
        './align_data/CN_long-term_2021-22.txt',
        './align_data/CN_long-term_2022-23.txt',
        './align_data/CN_short-term_20191018-31.txt',
        './align_data/CN_short-term_20201018-31.txt',
        './align_data/CN_short-term_20211018-31.txt',
        './align_data/US_long-term_2018-19.txt',
        './align_data/US_long-term_2019-20.txt',
        './align_data/US_long-term_2020-21.txt',
        './align_data/US_short-term_201903-0415.txt',
        './align_data/US_short-term_202003-0415.txt',
        './align_data/US_short-term_202103-0415.txt'
    ]

    for data_path in data_paths:
        edges = load_data_as_graph(data_path)
        with open('./edge_data/'+data_path.split("/")[-1].split(".")[0]+".pkl",'wb') as f:
            pickle.dump(edges,f)