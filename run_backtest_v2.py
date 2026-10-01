from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import paris_v2_backtest as base
from paris_v2_core import DEFAULT_CONFIG, assign_starters


def n(df, col):
    return pd.to_numeric(df[col], errors='coerce').fillna(0.0) if col in df.columns else pd.Series(0.0, index=df.index)


def aggregate_fixed(raw: pd.DataFrame) -> pd.DataFrame:
    df = base.normalize_columns(raw)
    if 'season_type' in df.columns:
        df = df[df['season_type'].astype(str).str.upper().eq('REG')].copy()

    if 'fumbles_lost' in df.columns:
        lost = n(df, 'fumbles_lost')
    else:
        lost = n(df,'sack_fumbles_lost') + n(df,'rushing_fumbles_lost') + n(df,'receiving_fumbles_lost')

    # Exact league scoring except the rare +1 bonus for 50+ yard TDs; that will be
    # added in the play-by-play pass after the base model is stable.
    df['paris_fp'] = (
        n(df,'passing_yards')/25 + 5*n(df,'passing_tds') - 2*n(df,'interceptions')
        + n(df,'rushing_yards')/10 + 6*n(df,'rushing_tds')
        + n(df,'receiving_yards')/10 + 6*n(df,'receiving_tds') + n(df,'receptions')
        - 2*lost
    )
    df['paris_fumbles_lost'] = lost

    cols = ['passing_yards','passing_tds','interceptions','attempts','rushing_yards','rushing_tds',
            'carries','receiving_yards','receiving_tds','receptions','targets','paris_fumbles_lost']
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0

    aggs = {c:(c,'sum') for c in cols + ['paris_fp']}
    aggs['games'] = ('week','nunique') if 'week' in df.columns else ('paris_fp','size')
    s = df.groupby(['season','player_id','player_name','position'], dropna=False).agg(**aggs).reset_index()
    g = s.games.clip(lower=1)
    s['ppg'] = s.paris_fp/g
    s['opportunity'] = np.select(
        [s.position.eq('QB'), s.position.eq('RB'), s.position.isin(['WR','TE'])],
        [(s.attempts + .75*s.carries)/g, (s.carries + 1.5*s.targets)/g, s.targets/g], default=0.0)
    touches = (s.carries+s.targets).replace(0,np.nan)
    att = s.attempts.replace(0,np.nan)
    s['efficiency'] = np.select(
        [s.position.eq('QB'), s.position.eq('RB'), s.position.isin(['WR','TE'])],
        [((s.passing_yards+s.rushing_yards)/att).fillna(0),
         ((s.rushing_yards+s.receiving_yards)/touches).fillna(0),
         (s.receiving_yards/s.targets.replace(0,np.nan)).fillna(0)], default=0.0)
    s['pos_pct'] = s.groupby(['season','position']).ppg.rank(pct=True, method='average')
    s['top5'] = (s.groupby(['season','position']).ppg.rank(ascending=False,method='min') <= 5).astype(int)
    s['team_fp_pg'] = 0.0
    return s


def add_metadata(df, players):
    out=df.copy()
    if players is None:
        out['age']=np.nan; out['draft_pick']=300.0; return out
    p=players.copy()
    if 'gsis_id' in p.columns and 'player_id' not in p.columns:
        p=p.rename(columns={'gsis_id':'player_id'})
    if 'draft_number' in p.columns and 'draft_pick' not in p.columns:
        p=p.rename(columns={'draft_number':'draft_pick'})
    keep=[c for c in ['player_id','birth_date','draft_pick'] if c in p.columns]
    if 'player_id' not in keep:
        out['age']=np.nan; out['draft_pick']=300.0; return out
    out=out.merge(p[keep].drop_duplicates('player_id'),on='player_id',how='left')
    if 'birth_date' in out.columns:
        birth=pd.to_datetime(out.birth_date,errors='coerce')
        d=pd.to_datetime(out.season.astype(str)+'-09-01',errors='coerce')
        out['age']=(d-birth).dt.days/365.25
    else:
        out['age']=np.nan
    out['draft_pick']=pd.to_numeric(out.get('draft_pick',300),errors='coerce').fillna(300).clip(1,300)
    return out


def model_table(seasons, players):
    df=add_metadata(seasons,players).sort_values(['player_id','season']).copy()
    # For a row representing the end of season Y, lag1 is Y itself, lag2=Y-1,
    # lag3=Y-2. This prevents accidentally ignoring the most recent season.
    for c in ['ppg','pos_pct','top5','opportunity','efficiency','games']:
        df[c+'_lag1']=df[c]
        df[c+'_lag2']=df.groupby('player_id')[c].shift(1)
        df[c+'_lag3']=df.groupby('player_id')[c].shift(2)
    df['team_fp_pg_lag1']=0.0
    df['ppg_trend']=df.ppg_lag1-df.ppg_lag2
    df['opp_trend']=df.opportunity_lag1-df.opportunity_lag2
    df['elite_top5']=.50*df.top5_lag1.fillna(0)+.30*df.top5_lag2.fillna(0)+.20*df.top5_lag3.fillna(0)
    df['elite_cont']=.50*df.pos_pct_lag1.fillna(0).pow(3)+.30*df.pos_pct_lag2.fillna(0).pow(3)+.20*df.pos_pct_lag3.fillna(0).pow(3)
    df['target_year']=df.season+1
    df['target_ppg']=df.groupby('player_id').ppg.shift(-1)
    df['target_games']=df.groupby('player_id').games.shift(-1)
    # Avoid treating tiny-sample spot appearances as full-season outcomes.
    # Four games is the minimum evaluation sample for this preliminary pass.
    df.loc[df['target_games'].fillna(0) < 4, 'target_ppg'] = np.nan
    return df


def overall_metrics(preds):
    rows=[]; ranks=[]
    for (year,var,alpha),g in preds.groupby(['target_year','variant','alpha']):
        b=g.drop_duplicates('player_id').copy()
        if not set(['QB','RB','WR','TE']).issubset(set(b.position)):
            continue
        _,pr=assign_starters(b[['player_id','position','predicted_ppg']],DEFAULT_CONFIG,'predicted_ppg')
        a=b[['player_id','position','target_ppg']].rename(columns={'target_ppg':'actual_ppg'})
        _,ar=assign_starters(a,DEFAULT_CONFIG,'actual_ppg')
        b['pred_vor']=b.predicted_ppg-b.position.map(pr)
        b['actual_vor']=b.target_ppg-b.position.map(ar)
        # A monotonic power transform cannot change rank correlation, so gamma is
        # held at 1.0 for ranking tests. Stud effects belong in predictive features.
        for gamma in [1.0]:
            x=b.copy(); x['gamma']=gamma
            x['pred_signal']=np.sign(x.pred_vor)*np.abs(x.pred_vor).pow(gamma)
            x['overall_rank']=x.pred_signal.rank(ascending=False,method='min')
            x['actual_rank']=x.actual_vor.rank(ascending=False,method='min')
            corr=x.actual_vor.corr(x.pred_signal,method='spearman')
            def ov(k):
                k=min(k,len(x))
                return len(set(x.nlargest(k,'pred_signal').player_id)&set(x.nlargest(k,'actual_vor').player_id))/max(k,1)
            rows.append(dict(year=year,variant=var,alpha=alpha,gamma=gamma,n=len(x),overall_spearman=corr,
                             top12_overlap=ov(12),top24_overlap=ov(24),top50_overlap=ov(50)))
            ranks.append(x)
    return pd.DataFrame(rows), pd.concat(ranks,ignore_index=True)


def candidate_summary(pos,overall):
    pp=pos[pos.year<2023].groupby(['variant','alpha'],as_index=False).agg(pre_pos=('spearman','mean'),pre_mae=('mae','mean'))
    vp=pos[pos.year==2023].groupby(['variant','alpha'],as_index=False).agg(val_pos=('spearman','mean'),val_mae=('mae','mean'))
    po=overall[overall.year<2023].groupby(['variant','alpha','gamma'],as_index=False).agg(pre_overall=('overall_spearman','mean'),pre_top24=('top24_overlap','mean'))
    vo=overall[overall.year==2023].groupby(['variant','alpha','gamma'],as_index=False).agg(val_overall=('overall_spearman','mean'),val_top24=('top24_overlap','mean'))
    z=po.merge(vo,on=['variant','alpha','gamma']).merge(pp,on=['variant','alpha']).merge(vp,on=['variant','alpha'])
    z['selection_index']=.45*z.val_overall+.20*z.pre_overall+.15*z.val_pos+.10*z.pre_pos+.05*z.val_top24+.05*z.pre_top24
    return z.sort_values('selection_index',ascending=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',required=True)
    ap.add_argument('--players-csv')
    ap.add_argument('--out-dir',default='outputs')
    args=ap.parse_args()
    Path(args.out_dir).mkdir(parents=True,exist_ok=True)
    raw=base.load_weekly_csvs(args.data_dir)
    seasons=aggregate_fixed(raw)
    players=pd.read_csv(args.players_csv,low_memory=False) if args.players_csv else None
    table=model_table(seasons,players)
    metrics,preds=base.grid_backtest(table,[2019,2020,2021,2022,2023])
    overall,ranks=overall_metrics(preds)
    summary=candidate_summary(metrics,overall)
    metrics.to_csv(Path(args.out_dir)/'position_metrics.csv',index=False)
    preds.to_csv(Path(args.out_dir)/'rolling_predictions.csv',index=False)
    overall.to_csv(Path(args.out_dir)/'overall_metrics.csv',index=False)
    ranks.to_csv(Path(args.out_dir)/'candidate_rankings.csv',index=False)
    summary.to_csv(Path(args.out_dir)/'candidate_summary.csv',index=False)
    print('=== TOP 20 CANDIDATES ===')
    print(summary.head(20).to_string(index=False))
    if len(summary):
        best=summary.iloc[0]
        y=ranks[(ranks.target_year==2023)&(ranks.variant==best.variant)&(ranks.alpha==best.alpha)&(ranks.gamma==best.gamma)].sort_values('overall_rank').head(30)
        print('\n=== 2023 TOP 30, BEST CANDIDATE (RETURNING VETERANS ONLY) ===')
        print(y[['overall_rank','player_name','position','predicted_ppg','target_ppg','pred_vor','actual_vor']].to_string(index=False))

if __name__=='__main__':
    main()
