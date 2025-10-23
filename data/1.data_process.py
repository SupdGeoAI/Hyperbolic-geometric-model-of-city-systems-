import os
import pandas as pd
from tqdm import tqdm

# 1. CN - short-term data
def CN_short(year):
    data_dir = f'./origin_data/{year}08-10/'
    dates = [
        '1018','1019','1020','1021','1022','1023','1024',
        '1025','1026','1027','1028','1029','1030','1031'
    ]
    data_files = [f'{year}{d}.txt' for d in dates]
    processed_path = f'./processed_data/CN_short-term_{year}1018-31.txt'

    all_lines = []
    for data_file in data_files:
        data_path = os.path.join(data_dir,data_file)
        with open(data_path,'r',encoding='utf-8') as f:
            lines = f.readlines()
        lines = lines[1:]
        all_lines.extend(lines)
    
    summary = {}
    for line in tqdm(all_lines):
        parts = line.strip('\n').split('\t')
        o_name = str(parts[0])
        d_name = str(parts[1])
        num = float(parts[3])
        
        key = (o_name,d_name)
        
        if key not in summary:
            summary[key] = num
        else:
            summary[key] += num

    with open(processed_path,'w') as file:
        for key, value in summary.items():
            file.write(f"{key[0]}\t{key[1]}\t{value/14.0}\n")

# 2. CN - long-term data
def CN_long():
    data_path = f'./origin_data/job_{year}.csv'
    processed_path = f'./processed_data/CN_long-term_{year}.txt'

    with open(data_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
    
    for i in tqdm(range(len(lines))):
        parts = lines[i].strip('\n').split(',')
        o_name = parts[0]
        d_name = parts[1]
        num = parts[2]
        lines[i] = f"{o_name}\t{d_name}\t{num}\n"

    with open(processed_path,'w') as file:
        file.writelines(lines)

# 3. US - short-term data
def US_short(year):
    data_path = f'./origin_data/msa2msa_{year}_03_15_04_15_mean.csv'
    processed_path = f'./processed_data/US_short-term_{year}03-0415.txt'
    df_data = pd.read_csv(data_path)
    all_lines = []
    for _, row in tqdm(df_data.iterrows()):
        geoid_o = str(int(row['GEOID_O']))
        geoid_d = str(int(row['GEOID_D']))
        num = float(row['visitor_flows'])
        all_lines.append(f"{geoid_o}\t{geoid_d}\t{num}\n")

    with open(processed_path,'w') as file:
        file.writelines(all_lines)

# 4. US - long-term data
def US_long(year):
    fips2geoid_path = './utils/US_fips2geoid.csv'
    df_fips2geoid = pd.read_csv(fips2geoid_path)

    fips2geoid = {}
    for _, row in df_fips2geoid.iterrows():
        statefp = str(int(row['STATEFP']))
        countyfp = str(int(row['COUNTYFP']))
        geoid = str(int(row['GEOID']))
        key = (statefp, countyfp)
        fips2geoid[key] = geoid

    input_paths = [f'./origin_data/countyinflow{year}.csv',f'./origin_data/countyoutflow{year}.csv']
    save_path = f'./processed_data/US_long-term_{year}.txt'

    pairs = {}
    for path in input_paths:
        with open(path,'r',encoding='utf-8') as f:
            lines = f.readlines()
        lines = lines[1:]
        for line in tqdm(lines):
            parts = line.strip("\n").split(",")
            y2_state_id = str(parts[0])
            y2_county_id = str(parts[1])
            y1_state_id = str(parts[2])
            y1_county_id = str(parts[3])
            num = float(parts[6])
            y2_key = (y2_state_id,y2_county_id)
            y1_key = (y1_state_id,y1_county_id)
            if y2_key in fips2geoid and y1_key in fips2geoid:
                tmp_key = (fips2geoid[y2_key],fips2geoid[y1_key])
                if tmp_key in pairs:
                    pairs[tmp_key] += num
                else:
                    pairs[tmp_key] = num
    all_lines = []
    for k,v in pairs.items():
        all_lines.append(
            k[0] + "\t" + k[1] + "\t" + str(v) + "\n"
        )

    with open(save_path,'w',encoding='utf-8') as f:
        f.writelines(all_lines)
    
if __name__ == '__main__':
    for year in ["2019","2020","2021"]:
        CN_short(year)
        US_short(year)
    for year in ["2018-19","2019-20","2020-21","2021-22","2022-23"]:
        CN_long(year)
    for year in ["2018-19","2019-20","2020-21"]:
        US_long(year)