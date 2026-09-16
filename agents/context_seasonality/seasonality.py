import pandas as pd

def _index_table(df, group, label):
    g=df.groupby(group, observed=True)['quantity'].agg(['mean','median','count']).reset_index()
    base=df['quantity'].mean()
    g['seasonal_index']=g['mean']/base if base else 1.0
    g['dimension']=label
    return g

def seasonality_summary(df):
    frames=[]
    for group,label in [(['day_of_week_num','day_of_week'],'weekday'),(['month'],'month'),(['quarter'],'quarter'),(['week_of_year'],'week_of_year')]:
        frames.append(_index_table(df,group,label))
    return frames
