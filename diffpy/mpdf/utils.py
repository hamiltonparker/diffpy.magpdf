import numpy as np
import pandas as pd

def  read_roth(f_name):
    df = pd.read_csv(f_name,delim_whitespace=True,skiprows=[0,1],header=None,usecols=[1,3,5])
    df.rename(columns={1:"x", 3:"y", 5:"spin"}, inplace=True)
    pos = df[['x','y']].values
    spin = df['spin'].values
    size = len(df)
    pad = np.zeros(size)
    spin = np.column_stack((pad,pad,spin))
    pos = np.column_stack((pos,pad))

    a = 0
    b = 0
    ang = 0

    with open(f_name,'r') as f:
        line = f.readline()
        if "cell lengths" in line:
            items = line.split()
            a = float(items[2])
            b = float(items[4])
            ang = np.radians(float(items[6]))

    a_vec = [a,0,0]
    b_vec = [b*np.cos(ang), b*np.sin(ang),0]
    pos = np.outer(pos[:,0],a_vec) + np.outer(pos[:,1],b_vec)
    return pos,spin


