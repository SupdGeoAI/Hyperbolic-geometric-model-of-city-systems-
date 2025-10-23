import pandas as pd
from tqdm import tqdm

def align_CN():
    CN_paths = [
        './processed_data/CN_long-term_2018-19.txt',
        './processed_data/CN_long-term_2019-20.txt',
        './processed_data/CN_long-term_2020-21.txt',
        './processed_data/CN_long-term_2021-22.txt',
        './processed_data/CN_long-term_2022-23.txt',
        './processed_data/CN_short-term_20191018-31.txt',
        './processed_data/CN_short-term_20201018-31.txt',
        './processed_data/CN_short-term_20211018-31.txt'
    ]

    all_sets = []
    for path in tqdm(CN_paths):
        with open(path,'r',encoding='utf-8') as f:
            lines = f.readlines()
        city_set = set([])
        for line in lines:
            parts = line.strip().split("\t")
            city_set.add(parts[0])
            city_set.add(parts[1])
        all_sets.append(city_set)

    reserve_cities = all_sets[0]
    for all_set in all_sets:
        reserve_cities = reserve_cities & all_set
    
    df_reserve_cities = pd.DataFrame(reserve_cities,columns=['name'])
    df_reserve_cities.to_csv('./utils/CN_code2name.csv',index=False)

def align_US():
    US_paths = [
        './processed_data/US_long-term_2018-19.txt',
        './processed_data/US_long-term_2019-20.txt',
        './processed_data/US_long-term_2020-21.txt',
        './processed_data/US_short-term_201903-0415.txt',
        './processed_data/US_short-term_202003-0415.txt',
        './processed_data/US_short-term_202103-0415.txt'
    ]

    all_sets = []
    for path in tqdm(US_paths):
        with open(path,'r',encoding='utf-8') as f:
            lines = f.readlines()
        city_set = set([])
        for line in lines:
            parts = line.strip().split("\t")
            city_set.add(parts[0])
            city_set.add(parts[1])
        all_sets.append(city_set)

    reserve_cities = all_sets[0]
    for all_set in all_sets:
        reserve_cities = reserve_cities & all_set
    
    msa_geoid_path = './utils/US_msa2geoid.csv'
    df_msa_geoid = pd.read_csv(msa_geoid_path)
    geoid_set = set([])
    for _, row in df_msa_geoid.iterrows():
        LSAD = str(row['LSAD'])
        if LSAD == 'M1':
            geoid_set.add(str(row['GEOID']))
    reserve_cities = reserve_cities & geoid_set

    df_reserve_cities = pd.DataFrame(reserve_cities,columns=['name'])
    df_reserve_cities.to_csv('./utils/US_code2geoid.csv',index=False)

def convert_CN():
    code_path = './utils/CN_code2name.csv'
    name2code = {}
    with open(code_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        name2code[lines[i].strip()] = str(i)

    CN_paths = [
        './processed_data/CN_long-term_2018-19.txt',
        './processed_data/CN_long-term_2019-20.txt',
        './processed_data/CN_long-term_2020-21.txt',
        './processed_data/CN_long-term_2021-22.txt',
        './processed_data/CN_long-term_2022-23.txt',
        './processed_data/CN_short-term_20191018-31.txt',
        './processed_data/CN_short-term_20201018-31.txt',
        './processed_data/CN_short-term_20211018-31.txt'
    ]

    for path in tqdm(CN_paths):
        with open(path,'r',encoding='utf-8') as f:
            lines = f.readlines()
        all_lines = []
        for line in lines:
            parts = line.strip().split("\t")
            o_name = parts[0]
            d_name = parts[1]
            num = float(parts[2])
            if o_name in name2code and d_name in name2code:
                all_lines.append(
                    name2code[o_name] + "\t" + name2code[d_name] + "\t" + str(num) + "\n"
                )
        with open("./align_data/"+path.split("/")[-1],'w',encoding='utf-8') as f:
            f.writelines(all_lines)

def convert_US():
    code_path = './utils/US_code2geoid.csv'
    name2code = {}
    with open(code_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        name2code[lines[i].strip()] = str(i)

    US_paths = [
        './processed_data/US_long-term_2018-19.txt',
        './processed_data/US_long-term_2019-20.txt',
        './processed_data/US_long-term_2020-21.txt',
        './processed_data/US_short-term_201903-0415.txt',
        './processed_data/US_short-term_202003-0415.txt',
        './processed_data/US_short-term_202103-0415.txt'
    ]

    for path in tqdm(US_paths):
        with open(path,'r',encoding='utf-8') as f:
            lines = f.readlines()
        all_lines = []
        for line in lines:
            parts = line.strip().split("\t")
            o_name = parts[0]
            d_name = parts[1]
            num = float(parts[2])
            if o_name in name2code and d_name in name2code:
                all_lines.append(
                    name2code[o_name] + "\t" + name2code[d_name] + "\t" + str(num) + "\n"
                )
        with open("./align_data/"+path.split("/")[-1],'w',encoding='utf-8') as f:
            f.writelines(all_lines)

if __name__ == '__main__':
    align_CN()
    align_US()
    convert_CN()
    convert_US()