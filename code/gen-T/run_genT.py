from discovery_utils import get_lake
from discover_candidates import CandidateTables
from prune_candidates import OriginatingTables
from targeted_integration import TableIntegration

# Parameters
integration_timeout = 3600
runStarmie = 0

def discover_originating_tables(source_path, datalake_path, benchmark, dl_subset):
    lake_dfs, all_lake_table_cols = get_lake(datalake_path, dl_subset)    
    print(f"In Gen-T, we have a datalake of {len(lake_dfs)} tables ({len(all_lake_table_cols)})")
    source_candidates = []

    # Call CandidateTables to find candidates
    candidate_table_finder = CandidateTables(benchmark,lake_dfs, all_lake_table_cols, source_candidates)
    candidateTablesFound, _ = candidate_table_finder.find_candidates(source_path)
    # Call OriginatingTables to prune candidates to a set of originating tables
    originating_tables_finder = OriginatingTables(benchmark, candidateTablesFound)
    originating_tables, _ = originating_tables_finder.find_originating_tables(source_path, datalake_path)
    origin_tables_matched_cols = {}
    if originating_tables: origin_tables_matched_cols = {t: candidateTablesFound[t] for t in originating_tables}
    return candidateTablesFound, origin_tables_matched_cols

def reclaim_source(source_path, datalake_path, benchmark, originating_tbls):
    table_integrator = TableIntegration(benchmark, originating_tbls, integration_timeout)
    table_integrator.integrate_tables(source_path, datalake_path)
    integration_result = table_integrator.reproducedSourceTable
    return integration_result

def get_reclaimed_table(source_path, datalake_path, benchmark_name, dl_subset):
    candidate_tbls, originating_tbls = discover_originating_tables(source_path, datalake_path, benchmark_name, dl_subset)
    integration_result = reclaim_source(source_path, datalake_path, benchmark_name, originating_tbls)
    return candidate_tbls, originating_tbls, integration_result
