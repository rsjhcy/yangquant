#!/usr/bin/env python3
"""Trend-following variant: auto-detect trends and switch strategy"""
import sys,io,numpy as np,pandas as pd
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

# Load optical data
df=pd.read_parquet('data/backtest_optical.parquet')
df=df.sort_values(['symbol','date']).reset_index(drop=True)
symbols=sorted(df['symbol'].unique())
print(f'Optical: {len(df)} rows, {len(symbols)} stocks, {df["date"].nunique()} days')

MIN_HISTORY=60;TOP_N=3

def compute_indicators(group):
    g=group.sort_values('date').copy()
    close=g['close'].values.astype(float);high=g['high'].values.astype(float)
    low=g['low'].values.astype(float);volume=g['volume'].values.astype(float);n=len(close)
    g['ret_1d']=np.append(np.diff(close)/close[:-1],np.nan)
    g['ret_5d']=np.append(np.diff(close,5)/close[:-5],[np.nan]*5)
    g['ret_10d']=np.append(np.diff(close,10)/close[:-10],[np.nan]*10)
    g['ret_20d']=np.append(np.diff(close,20)/close[:-20],[np.nan]*20)
    for p in[5,10,20,60]:g[f'ma_{p}']=pd.Series(close).rolling(p,min_periods=p//2).mean().values
    g['ma20_dev']=(close-g['ma_20'].values)/g['ma_20'].values
    ma5v=g['ma_5'].values;slope=np.diff(ma5v)/np.maximum(np.abs(ma5v[:-1]),1e-9)
    g['ma5_slope']=np.append([np.nan],slope)
    # ADX-like trend strength
    delta=np.diff(close,prepend=close[0]);gain=np.maximum(delta,0);loss=np.maximum(-delta,0)
    avg_gain=pd.Series(gain).rolling(14).mean().values;avg_loss=pd.Series(loss).rolling(14).mean().values
    g['rsi']=100-100/(1+avg_gain/np.maximum(avg_loss,1e-9))
    # Trend strength: % of days above MA20 in last 40 days
    above_ma=np.array([np.nan]*n,dtype=float)
    for i in range(MIN_HISTORY,n):
        above_ma[i]=np.mean(close[i-40:i]>g['ma_20'].values[i-40:i])
    g['trend_strength']=above_ma
    # ADX proxy: abs(MA5 slope) / ATR_pct
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

# Results: V1 (original) vs V2 (trend-aware) vs Buy&Hold
br={'1d':[],'5d':[],'10d':[]}  # balanced trend-aware
ar={'1d':[],'5d':[],'10d':[]}  # aggressive trend-aware
or_={'1d':[],'5d':[],'10d':[]} # original strategy
bh={'1d':[],'5d':[],'10d':[]}  # buy & hold all
scored=0;regime_log=[]

by_sym={sym:df[df['symbol']==sym].set_index('date') for sym in symbols}

for di,dt in enumerate(dates):
    if(di+1)%200==0:print(f'  [{di+1}/{len(dates)}] {dt}')
    day_data=df[df['date']==dt]
    if len(day_data)<3:continue

    # Market regime
    up_r=np.mean(day_data['ma20_dev'].dropna().values>0) if 'ma20_dev' in day_data.columns else 0.5
    mkt_20=np.mean(day_data['ret_20d'].dropna().values) if 'ret_20d' in day_data.columns else 0
    if mkt_20>0.05 and up_r>0.6:market_regime='trending_up'
    elif mkt_20<-0.05 or up_r<0.3:market_regime='trending_down'
    else:market_regime='ranging'
    regime_log.append(market_regime)

    # Original weights (for comparison)
    if market_regime=='trending_up':orig_w={'m5':0.18,'m20':0.12,'tma':0.18,'ms':0.10,'vb':0.12,'rsi':0.08,'bb':0.02,'lv':0.05,'vs':0.10,'ra':0.05}
    elif market_regime=='trending_down':orig_w={'m5':0.05,'m20':0.05,'tma':0.05,'ms':0.05,'vb':0.08,'rsi':0.15,'bb':0.22,'lv':0.20,'vs':0.05,'ra':0.10}
    else:orig_w={'m5':0.12,'m20':0.08,'tma':0.10,'ms':0.08,'vb':0.18,'rsi':0.10,'bb':0.14,'lv':0.08,'vs':0.08,'ra':0.04}

    # Trend-aware weights: when stock is in strong trend, go aggressive on momentum
    trend_w_up={'m5':0.25,'m20':0.20,'tma':0.15,'ms':0.15,'vb':0.10,'rsi':0.05,'bb':0.00,'lv':0.00,'vs':0.08,'ra':0.02}

    scores=[]
    for sym in symbols:
        if sym not in by_sym:continue
        sdf=by_sym[sym]
        if dt not in sdf.index:continue
        row=sdf.loc[dt]
        r5=float(row['ret_5d']) if pd.notna(row['ret_5d']) else 0
        r10=float(row['ret_10d']) if pd.notna(row['ret_10d']) else 0
        r20=float(row['ret_20d']) if pd.notna(row['ret_20d']) else 0
        ap=float(row['atr_pct']) if pd.notna(row['atr_pct']) else 0.03
        md=float(row['ma20_dev']) if pd.notna(row['ma20_dev']) else 0
        ms_v=float(row['ma5_slope']) if pd.notna(row['ma5_slope']) else 0
        vr=float(row['vol_ratio']) if pd.notna(row['vol_ratio']) else 1.0
        rs=float(row['rsi']) if pd.notna(row['rsi']) else 50
        bp=float(row['bb_position']) if pd.notna(row['bb_position']) else 0
        vs=int(row['vol_surge_days']) if pd.notna(row['vol_surge_days']) else 0
        r1=float(row['ret_1d']) if pd.notna(row['ret_1d']) else 0
        ts=float(row['trend_strength']) if pd.notna(row['trend_strength']) else 0.5

        # Detect if THIS stock is in a strong individual trend
        stock_trending = (ts > 0.75 and r20 > 0.08)  # Above MA20 75%+ and strong 20d return

        # --- ORIGINAL factor scoring ---
        fm5=min(100,max(0,50+(r5/max(ap,0.005))*80))
        fm20=min(100,max(0,50+r20*200))
        ftma=min(100,max(0,55+md*300-abs(md)*200))
        fms=min(100,max(0,50+ms_v*2000))
        fvb=min(100,50+vr*25) if vr>1.3 and r1>0 else min(100,40+vr*10)
        frsi_orig=min(100,50+(rs-45)*2) if 45<=rs<=70 else (70 if rs<30 else (10 if rs>80 else 35))
        fbb=85 if -1<=bp<=-0.5 else (70 if -0.5<bp<=0 else (55 if 0<bp<=0.5 else 25))
        flv=min(100,max(5,90-ap*250))
        fvs=min(100,20+vs*20)
        fra=min(100,max(0,50+(r20/max(ap,0.005))*20))

        # --- TREND-AWARE RSI: don't penalize high RSI in trends ---
        if stock_trending:
            # In strong trend, RSI 50-85 is NORMAL and good
            if 50<=rs<=85:frsi=min(100,50+(rs-50)*2.5)  # Higher RSI = stronger trend = better
            elif rs<30:frsi=50  # Deep oversold in trend = concern
            else:frsi=40
            # Remove high-open penalty
            open_penalty=1.0
            # Momentum acceleration bonus
            mom_bonus=1.0
            if r5>0 and r10>0 and r5/r10>0.4:  # 5d return is healthy fraction of 10d
                mom_bonus=1.15
        else:
            frsi=frsi_orig
            open_penalty=0.3 if r1>0.095 else 1.0
            mom_bonus=1.0

        # --- Compute scores ---
        # Original balanced
        orig_b=fm5*orig_w['m5']+fm20*orig_w['m20']+ftma*orig_w['tma']+fms*orig_w['ms']+fvb*orig_w['vb']+frsi_orig*orig_w['rsi']+fbb*orig_w['bb']+flv*orig_w['lv']+fvs*orig_w['vs']+fra*orig_w['ra']
        facs=[fm5,fm20,ftma,fms,fvb,frsi_orig,fbb,flv,fvs,fra]
        hc=sum(1 for f in facs if f>60)
        if hc>=7:orig_b*=1.15
        elif hc>=5:orig_b*=1.08
        if r1>0.095:orig_b*=0.3

        # Trend-aware: if stock is trending, use momentum-heavy weights
        if stock_trending:
            w=trend_w_up
        else:
            w=orig_w

        trend_b=fm5*w['m5']+fm20*w['m20']+ftma*w['tma']+fms*w['ms']+fvb*w['vb']+frsi*w['rsi']+fbb*w['bb']+flv*w['lv']+fvs*w['vs']+fra*w['ra']
        trend_b*=mom_bonus*open_penalty
        # Confluence in trend mode
        if stock_trending and hc>=5:trend_b*=1.10

        # Trend aggressive: even more momentum
        trend_a=trend_b*1.0
        if stock_trending:
            trend_a=fm5*0.30+fm20*0.25+ftma*0.15+fms*0.15+fvb*0.05+frsi*0.05+fbb*0.00+flv*0.00+fvs*0.05+fra*0.00
            trend_a*=mom_bonus

        f1=float(row['ret_1d']) if pd.notna(row['ret_1d']) else np.nan
        f5=float(row['ret_5d']) if pd.notna(row['ret_5d']) else np.nan
        f10=float(row['ret_10d']) if pd.notna(row['ret_10d']) else np.nan
        scores.append({'ob':orig_b,'tb':trend_b,'ta':trend_a,'f1':f1,'f5':f5,'f10':f10})

    if len(scores)<TOP_N:continue
    scored+=1

    def track(key,targets):
        scores.sort(key=lambda x:x[key],reverse=True)
        for rk,target in[('f1',targets['1d']),('f5',targets['5d']),('f10',targets['10d'])]:
            v=[s[rk] for s in scores[:TOP_N] if pd.notna(s[rk])]
            if v:target.append(np.mean(v))

    track('ob',or_)
    track('tb',br)
    track('ta',ar)
    # Buy & hold: equal weight all
    for rk,target in[('f1',bh['1d']),('f5',bh['5d']),('f10',bh['10d'])]:
        v=[s[rk] for s in scores if pd.notna(s[rk])]
        if v:target.append(np.mean(v))

# Results
print(f'\nScored {scored} days')
t_up=sum(1 for r in regime_log if r=='trending_up')
t_rng=sum(1 for r in regime_log if r=='ranging')
t_dn=sum(1 for r in regime_log if r=='trending_down')
print(f'Market: {t_up}d UP | {t_rng}d RNG | {t_dn}d DOWN')

def stats(rets,name):
    if not rets['1d']:return
    r=np.mean(rets['1d']);s=r/np.std(rets['1d'])*np.sqrt(252) if np.std(rets['1d'])>0 else 0
    v=np.cumprod([1+x for x in rets['1d']])
    mdd_=np.min((v-np.maximum.accumulate(v))/np.maximum.accumulate(v)) if len(v)>0 else 0
    wr=sum(1 for x in rets['1d'] if x>0)/len(rets['1d'])*100 if rets['1d'] else 0
    cum=np.prod([1+x for x in rets['1d']])-1 if rets['1d'] else 0
    r5=np.mean(rets['5d'])*100 if rets['5d'] else 0
    print(f'  {name:28s} Daily:{r:+.3%} | 5D:{r5:+.2%} | Sharpe:{s:5.2f} | MaxDD:{mdd_:+6.1%} | Win:{wr:4.1f}% | Cum:{cum:+7.1%}')

print()
print('='*75)
print('  OPTICAL COMM STRATEGY COMPARISON (12 stocks, 2023-2026)')
print('='*75)
stats(or_,'1) Original Strategy')
stats(br,'2) Trend-Aware Balanced')
stats(ar,'3) Trend-Aware Aggressive')
stats(bh,'4) Buy & Hold (equal weight)')
print('='*75)

# Which had highest cum return?
oc=np.prod([1+x for x in or_['1d']])-1 if or_['1d'] else 0
tc=np.prod([1+x for x in br['1d']])-1 if br['1d'] else 0
ac=np.prod([1+x for x in ar['1d']])-1 if ar['1d'] else 0
bc=np.prod([1+x for x in bh['1d']])-1 if bh['1d'] else 0

print()
best=max(oc,tc,ac,bc)
if best==tc:print('WINNER: Trend-Aware Balanced')
elif best==ac:print('WINNER: Trend-Aware Aggressive')
elif best==oc:print('WINNER: Original Strategy')
else:print('WINNER: Buy & Hold')
print(f'  Improvement over original: {max(tc,ac)-oc:+.1%}')
