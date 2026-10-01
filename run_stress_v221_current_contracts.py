from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

import run_stress_v221 as s
import run_backtest_v22 as v22
import run_backtest_v211 as v211
import run_backtest_v21 as v21


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',required=True)
    ap.add_argument('--players-csv')
    ap.add_argument('--contracts-parquet')
    ap.add_argument('--out-dir',default='outputs_v221p')
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    raw=v21.old.load_weekly_csvs(args.data_dir)
    seasons=v21.aggregate_v21(raw)
    players=pd.read_csv(args.players_csv,low_memory=False) if args.players_csv else None
    contracts=pd.read_parquet(args.contracts_parquet) if args.contracts_parquet else None

    table=v211.augment_table(seasons,players)
    table=v22.augment_targets(table,seasons)
    table=s.add_veteran_features(table)
    table=s.attach_contract_features(table,contracts)

    # Contract coverage diagnostics so a failed join cannot masquerade as a
    # legitimate 'contracts do not matter' result.
    diag = table.groupby('target_year',as_index=False).agg(
        rows=('player_id','size'),
        contract_match_rate=('contract_match','mean'),
        years_remaining_mean=('contract_years_remaining','mean'),
        apy_cap_nonmissing=('contract_apy_cap_pct',lambda x: x.notna().mean()),
        guaranteed_pct_nonmissing=('contract_guaranteed_pct',lambda x: x.notna().mean()),
    )
    diag.to_csv(out/'v221_contract_coverage.csv',index=False)

    preds,role_metrics=s.build_predictions(table)
    metrics,ranks=s.evaluate_value(preds)
    summary=s.summarize(metrics,role_metrics)
    best=summary.iloc[0]
    fp=s.false_positive_report(preds,best)

    role_metrics.to_csv(out/'v221_role_subgroup_metrics.csv',index=False)
    metrics.to_csv(out/'v221_value_stress_metrics.csv',index=False)
    summary.to_csv(out/'v221_stress_summary.csv',index=False)
    fp.to_csv(out/'v221_veteran_false_positives.csv',index=False)
    best_ranks=ranks[(ranks.role_quantile.eq(best.role_quantile))&(ranks.C.eq(best.C))&(ranks.architecture.eq(best.architecture))&(ranks.risk_weight.eq(best.risk_weight))].copy()
    best_ranks.to_csv(out/'v221_selected_rankings.csv',index=False)

    print('=== CONTRACT COVERAGE ===')
    print(diag.to_string(index=False))
    print('\n=== PARIS V2.2.1 VETERAN STRESS TEST (CURRENT PARQUET CONTRACTS) ===')
    print(summary.head(20).to_string(index=False))
    print('\n=== SELECTED ===')
    print(best.to_string())
    print('\n=== VETERAN ROLE METRICS BY ARCHITECTURE ===')
    r=role_metrics[role_metrics.subgroup.eq('veteran')].groupby(['architecture','year'],as_index=False).agg(auc=('auc','mean'),brier=('brier','mean'))
    print(r.to_string(index=False))
    print('\n=== HIGHEST-CONFIDENCE VETERAN ROLE FAILURES ===')
    print(fp.sort_values('role_prob',ascending=False).head(30).to_string(index=False))

if __name__=='__main__':
    main()
