'''This code get statistics from previously calculated topologies,
empty cell in xlsx means 0 as all values are nans. We used nans for not affecting the mean calculation'''


import sys
import os
sys.path.append(os.path.join('/'.join(sys.path[0].split("/")[:-2])))
from utils.util import getStatistics
import argparse
import time
import pandas as pd
import numpy as np
import statistics as st
import pickle

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('--dataPath',   type=str, required=True)
    parser.add_argument('--resPath',    type=str)
    parser.add_argument('--labels',     type=str, required=True, nargs='+')
    args = parser.parse_args()


    with open(args.dataPath, 'rb') as f:
        data = pickle.load(f)

    data = np.nan_to_num(data)
    #Save statiscal summary on excel with exp name
    exp_name = args.dataPath.split('/')[-1].split('.')[0]
    
    for b in range(3):
        res_dataframe = []
        for i, index_name in enumerate(args.labels):
            tmp = getStatistics(data[i,:,b])
            tmp.append(st.mode(data[i,:,b]))
            res_dataframe.append(tmp)

        
        indexs  = ["{}_{}_{}".format(exp_name, index, b) for index in args.labels] 
        columns = ['mean', 'std', 'min', 'max', 'median', 'lowQuart', 'upQuart', 'lowWhisker', 'upWhisker', 'mode']
        df = pd.DataFrame(res_dataframe, index=indexs, columns=columns)
        
        if not os.path.exists(args.resPath):
            df.to_excel(args.resPath, sheet_name='sheet1')
        else:
            with pd.ExcelWriter(args.resPath, engine="openpyxl", mode='a',if_sheet_exists="overlay") as writer:
                startrow = writer.sheets['sheet1'].max_row
                df.to_excel(writer, sheet_name='sheet1', startrow=startrow, header=False)

if __name__ == '__main__':
    start = time.time()
    main()
    print("Total duration processing: {} s ".format(time.time()-start))