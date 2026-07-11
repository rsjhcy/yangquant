#!/usr/bin/env python3
"""Cross-validation: Group 3 (tech stocks)"""
import sys,io,numpy as np,pandas as pd
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

df=pd.read_parquet('data/backtest_tech.parquet')
df=df.sort_values(['symbol','date']).reset_index(drop=True)
symbols=sorted(df['symbol'].unique())
print(f'Group 3 (Tech): {len(df)} rows, {len(symbols)} stocks, {df["date"].nunique()} days')

MIN_HISTORY=60;TOP_N=3

def compute_indicators(group):
    g=group.sort_values('date').copy()
    close=g['close'].values.astype(float);high=g['high'].values.astype(float)
    low=g['low'].values.astype(float);volume=g['volume'].values.astype(float);n=len(close)
    g['ret_1d']=np.append(np.diff(close)/close[:-1],np.nan)
    g['ret_5d']=np.append(np.diff(close,5)/close[:-5],[np.nan]*5)
    g['ret_20d']=np.append(np.diff(close,20)/close[:-20],[np.nan]*20)
    for p in[5,10,20,60]:g[f'ma_{p}']=pd.Series(close).rolling(p,min_periods=p//2).mean().values
    g['ma20_dev']=(close-g['ma_20'].values)/g['ma_20'].values
    ma5v=g['ma_5'].values;slope=np.diff(ma5v)/np.maximum(np.abs(ma5v[:-1]),1e-9)
    g['ma5_slope']=np.append([np.nan],slope)
    delta=np.diff(close,prepend=close[0]);gain=np.maximum(delta,0);loss=np.maximum(-delta,0)
    avg_gain=pd.Series(gain).rolling(14).mean().values;avg_loss=pd.Series(loss).rolling(14).mean().values
    g['rsi']=100-100/(1+avg_gain/np.maximum(avg_loss,1e-9))
    ma20=g['ma_20'].values;std20=pd.Series(close).rolling(20).std().values
    g['bb_position']=(close-ma20)/np.maximum(2*std20,1e-9)
    tr=np.maximum(high-low,np.maximum(abs(high-np.append([close[0]],close[:-1])),abs(low-np.append([close[0]],close[:-1]))))
    g['atr']=pd.Series(tr).rolling(14).mean().values;g['atr_pct']=g['atr'].values/close
    g['vol_ma20']=pd.Series(volume).rolling(20).mean().values
    g['vol_ratio']=volume/np.maximum(g['vol_ma20'].values,1)
    g['vol_surge_days']=0;streak=0
    for i in range(n):
        if g['vol_ratio'].iloc[i]>1.3:streak+=1
        else:streak=0
        g.loc[g.index[i],'vol_surge_days']=streak
    return g

print('Indicators...')
frames=[compute_indicators(df[df['symbol']==sym].copy()) for sym in symbols if len(df[df['symbol']==sym])>=MIN_HISTORY]
df=pd.concat(frames,ignore_index=True)
symbols=sorted(df['symbol'].unique());dates=sorted(df['date'].unique())
print(f'{len(df)} rows, {len(symbols)} stocks, {len(dates)} days')

by_sym={sym:df[df['symbol']==sym].set_index('date') for sym in symbols}
br={'1d':[],'5d':[]};ar={'1d':[],'5d':[]};scored=0

for di,dt in enumerate(dates):
    if(di+1)%200==0:print(f'  [{di+1}/{len(dates)}] {dt}')
    day_data=df[df['date']==dt]
    if len(day_data)<5:continue
    up_r=np.mean(day_data['ma20_dev'].dropna().values>0) if 'ma20_dev' in day_data.columns else 0.5
    mkt_20=np.mean(day_data['ret_20d'].dropna().values) if 'ret_20d' in day_data.columns else 0
    if mkt_20>0.05 and up_r>0.6:regime='trending_up'
    elif mkt_20<-0.05 or up_r<0.3:regime='trending_down'
    else:regime='ranging'

    if regime=='trending_up':w={'m5':0.18,'m20':0.12,'tma':0.18,'ms':0.10,'vb':0.12,'rsi':0.08,'bb':0.02,'lv':0.05,'vs':0.10,'ra':0.05}
    elif regime=='trending_down':w={'m5':0.05,'m20':0.05,'tma':0.05,'ms':0.05,'vb':0.08,'rsi':0.15,'bb':0.22,'lv':0.20,'vs':0.05,'ra':0.10}
    else:w={'m5':0.12,'m20':0.08,'tma':0.10,'ms':0.08,'vb':0.18,'rsi':0.10,'bb':0.14,'lv':0.08,'vs':0.08,'ra':0.04}

    scores=[]
    for sym in symbols:
        if sym not in by_sym:continue
        sdf=by_sym[sym]
        if dt not in sdf.index:continue
        row=sdf.loc[dt]
        r5=float(row['ret_5d']) if pd.notna(row['ret_5d']) else 0
        r20=float(row['ret_20d']) if pd.notna(row['ret_20d']) else 0
        ap=float(row['atr_pct']) if pd.notna(row['atr_pct']) else 0.03
        md=float(row['ma20_dev']) if pd.notna(row['ma20_dev']) else 0
        ms_v=float(row['ma5_slope']) if pd.notna(row['ma5_slope']) else 0
        vr=float(row['vol_ratio']) if pd.notna(row['vol_ratio']) else 1.0
        rs=float(row['rsi']) if pd.notna(row['rsi']) else 50
        bp=float(row['bb_position']) if pd.notna(row['bb_position']) else 0
        vs=int(row['vol_surge_days']) if pd.notna(row['vol_surge_days']) else 0
        r1=float(row['ret_1d']) if pd.notna(row['ret_1d']) else 0

        fm5=min(100,max(0,50+(r5/max(ap,0.005))*80))
        fm20=min(100,max(0,50+r20*200))
        ftma=min(100,max(0,55+md*300-abs(md)*200))
        fms=min(100,max(0,50+ms_v*2000))
        fvb=min(100,50+vr*25) if vr>1.3 and r1>0 else min(100,40+vr*10)
        frsi=min(100,50+(rs-45)*2) if 45<=rs<=70 else (70 if rs<30 else (10 if rs>80 else 35))
        fbb=85 if -1<=bp<=-0.5 else (70 if -0.5<bp<=0 else (55 if 0<bp<=0.5 else 25))
        flv=min(100,max(5,90-ap*250))
        fvs=min(100,20+vs*20)
        fra=min(100,max(0,50+(r20/max(ap,0.005))*20))

        bt=fm5*w['m5']+fm20*w['m20']+ftma*w['tma']+fms*w['ms']+fvb*w['vb']+frsi*w['rsi']+fbb*w['bb']+flv*w['lv']+fvs*w['vs']+fra*w['ra']
        at=fm5*w['m5']*1.5+fm20*w['m20']*1.5+ftma*w['tma']+fms*w['ms']+fvb*w['vb']*1.5+frsi*w['rsi']+fbb*w['bb']+flv*w['lv']+fvs*w['vs']*1.5+fra*w['ra']
        facs=[fm5,fm20,ftma,fms,fvb,frsi,fbb,flv,fvs,fra]
        hc=sum(1 for f in facs if f>60)
        if hc>=7:bt*=1.15;at*=1.15
        elif hc>=5:bt*=1.08;at*=1.08
        if r1>0.095:bt*=0.3;at*=0.3

        f1=float(row['ret_1d']) if pd.notna(row['ret_1d']) else np.nan
        f5=float(row['ret_5d']) if pd.notna(row['ret_5d']) else np.nan
        scores.append({'b':bt,'a':at,'f1':f1,'f5':f5})

    if len(scores)<TOP_N:continue
    scored+=1
    scores.sort(key=lambda x:x['b'],reverse=True)
    for rk,target in[('f1',br['1d']),('f5',br['5d'])]:
        v=[s[rk] for s in scores[:TOP_N] if pd.notna(s[rk])]
        if v:target.append(np.mean(v))
    scores.sort(key=lambda x:x['a'],reverse=True)
    for rk,target in[('f1',ar['1d']),('f5',ar['5d'])]:
        v=[s[rk] for s in scores[:TOP_N] if pd.notna(s[rk])]
        if v:target.append(np.mean(v))

print(f'\nScored {scored} days')
print()
print('='*80)
print('  FINAL: 3 Groups x 90 Stocks x 3 Years')
print('='*80)
print(f'  {"Group":22s} {"N":>5s} {"Daily":>8s} {"Sharpe":>7s} {"MaxDD":>7s} {"WinRate":>7s} {"Cum":>8s}')
print('  '+'-'*68)

def stats(rets,name,n):
    if not rets or not rets['1d']: return
    r=np.mean(rets['1d'])
    s=np.mean(rets['1d'])/np.std(rets['1d'])*np.sqrt(252) if np.std(rets['1d'])>0 else 0
    v=np.cumprod([1+x for x in rets['1d']])
    mdd_=np.min((v-np.maximum.accumulate(v))/np.maximum.accumulate(v)) if len(v)>0 else 0
    wr=sum(1 for x in rets['1d'] if x>0)/len(rets['1d'])*100 if rets['1d'] else 0
    cum=np.prod([1+x for x in rets['1d']])-1 if rets['1d'] else 0
    print(f'  {name:22s} {n:>5d} {r:>+7.3%} {s:>7.2f} {mdd_:>+7.2%} {wr:>6.1f}% {cum:>+7.1%}')

stats(br,'Group3 Tech Balanced',29)
stats(ar,'Group3 Tech Aggressive',29)

# Reference from previous runs
print(f'  {"Group1 FinCons Balanced":22s} {30:>5d} {0.00137:>+7.3%} {1.60:>7.2f} {-0.1915:>+7.2%} {52.4:>6.1f}% {1.3936:>+7.1%}')
print(f'  {"Group2 MfgEng Balanced":22s} {30:>5d} {0.00183:>+7.3%} {2.00:>7.2f} {-0.1954:>+7.2%} {54.8:>6.1f}% {2.6045:>+7.1%}')

print('='*80)

# Average Sharpe
sh_vals=[1.60,2.00,s,0]
sh_vals=[1.60,2.00,s] if s else [1.60,2.00]
avg=np.mean(sh_vals);std=np.std(sh_vals)
print(f'\nAvg Sharpe: {avg:.2f} +/- {std:.2f}')
if std<0.6:print('VERDICT: Strategy is ROBUST across diverse stock universes')
elif std<1.0:print('VERDICT: Strategy is REASONABLY STABLE with some sector variance')
else:print('VERDICT: High sector variance, diversify picks')
