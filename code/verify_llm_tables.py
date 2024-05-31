import glob
import pandas as pd
import argparse
import json
from IPython.display import display
import time
from lake_support import find_lake_support
import sys
sys.path.append("gen-T/")
from run_genT import get_reclaimed_table

def load_json(saveFolder, src_table_path):
    with open(f"../datasets/{saveFolder}/{src_table_path}") as f: 
        dict = json.load(f) # source table: [expected integration set, including err and null tables]
    return dict

def save_supporting_as_json(dict, saveFolder, src_table_path):
    with open(f"../datasets/{saveFolder}/{src_table_path}", "w+") as f: 
        json.dump(dict, f, indent=4)
        
        
def reclaim_table(src_table, src_table_name, saveFolder, DATALAKE_PATH, benchmark):
    # -------------------------- FINDING RECLAIMED TABLE --------------------------
    # # Pass LLM-generated Source Table into Gen-T and reclaim it
    start_time = time.time()
    related_tables = []
    result_is_empty = False
    candidate_tbls, originating_tbls, integration_result = get_reclaimed_table(src_table, DATALAKE_PATH, benchmark, related_tables)
    save_supporting_as_json(candidate_tbls, saveFolder, f"supporting_tables/{src_table_name}")
    print(f"Reclaimed table in {time.time()-start_time:.3f} seconds ({(time.time()-start_time)/60:.3f} minutes)")
    print(f"There are {len(originating_tbls)} originating tables (from {len(candidate_tbls)} candidate tables)")
    if not integration_result.empty:
        integration_result.drop_duplicates(inplace=True)
        integration_result.to_csv(f"../datasets/{saveFolder}/tables_verified/{src_table_name}", index=False)
    else: 
        print(f"The integration result is empty, with {len(originating_tbls)} originating tables")
        result_is_empty = True
    # -------------------------- END RECLAIMED TABLE --------------------------
    print(f"there are {len(originating_tbls)} originating tables.")
    return candidate_tbls, integration_result, result_is_empty
    
def load_reclaimed_table(src_table_name, saveFolder):
    # -------------------------- GET RECLAIMED TABLE and CANDIDATES --------------------------
    integration_result = pd.read_csv(f"../datasets/{saveFolder}/tables_verified/{src_table_name}")
    
    candidate_tbls = load_json(saveFolder, f"supporting_tables/{src_table_name}")
    print(f"Integration result of {src_table_name} has {integration_result.shape[0]} rows and {integration_result.shape[1]} columns")
    print(f"There are {len(candidate_tbls)} candidate tables")
    return candidate_tbls, integration_result
    
def find_support_scores(table, key_col, candidate_tbls, benchmark, saveFolder, src_table_name):
    # -------------------------- FINDING SUPPORT SCORES FOR SOURCE --------------------------
    start_time = time.time()
    num_support_tables, support_scores, related_tables, annotated_support = find_lake_support(table, key_col, candidate_tbls, benchmark=benchmark)
    print(f"Found support scores in {time.time()-start_time:.3f} seconds ({(time.time()-start_time)/60:.3f} minutes)")
    
    # -------------------------- END SUPPORT SCORES FOR SOURCE --------------------------
    return num_support_tables, support_scores, annotated_support
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tpch", choices=['tpch', 'ugen'])
    hp = parser.parse_args()
    
    # ----- TPCH
    if hp.dataset == "tpch":
        benchmark = "tpch"
    
    # ---- UGEN
    if hp.dataset == "ugen":
        benchmark="wikiTables"
    
    DATALAKE_PATH = f"../datasets/{benchmark}/datalake"
    LLM_SOURCE_PATH = f"../datasets/{hp.dataset}/tables_source"

    for srcInd, src_table in enumerate(glob.glob(LLM_SOURCE_PATH+"/*.csv")):
        src_table_name = src_table.split("/")[-1]
        print(f"\t\t *********** Getting support scores for Source Table {src_table_name}, using {benchmark} datalake...")
        # Get Source Table
        s_tab = pd.read_csv(src_table)
        
        print(f"Source Table has {s_tab.shape[0]} rows, {s_tab.shape[1]} columns")
        s_key_name = list(s_tab.columns)[0] # the first column is the key column  
        
        # Find reclaimed table
        candidate_tbls, integration_result, result_is_empty = reclaim_table(src_table, src_table_name, hp.dataset, DATALAKE_PATH, benchmark)
        if result_is_empty: continue
        # # Get reclaimed table
        # try: 
        #     candidate_tbls, integration_result = load_reclaimed_table(src_table_name, hp.dataset)
        # except: continue 
        
        # -------------------------- FINDING SUPPORT SCORES FOR SOURCE --------------------------
        num_support_tabs, s_tab_support_scores, s_annotated_support = find_support_scores(src_table, s_key_name, candidate_tbls, benchmark, hp.dataset, src_table_name)
        display(s_tab_support_scores)
        num_support_tabs.to_csv(f"../datasets/{hp.dataset}/support_scores/source_tables/num_tables_{src_table_name}")
        s_annotated_support.to_csv(f"../datasets/{hp.dataset}/support_scores/source_tables/annotated_{src_table_name}")
        s_tab_support_scores.to_csv(f"../datasets/{hp.dataset}/support_scores/source_tables/{src_table_name}")
        # -------------------------- END SUPPORT SCORES FOR SOURCE --------------------------
        
        # -------------------------- FINDING SUPPORT SCORES FOR RECLAIMED TABLE --------------------------
        reclaimed_tbl = pd.read_csv(f"../datasets/{hp.dataset}/tables_verified/{src_table_name}")
        recl_num_support_tabs, rec_tab_support_scores, r_annotated_support = find_support_scores(f"../datasets/{hp.dataset}/tables_verified/{src_table_name}", s_key_name, candidate_tbls, benchmark, hp.dataset, src_table_name) 
        display(rec_tab_support_scores)
        recl_num_support_tabs.to_csv(f"../datasets/{hp.dataset}/support_scores/verified_tables/num_tables_{src_table_name}")
        r_annotated_support.to_csv(f"../datasets/{hp.dataset}/support_scores/verified_tables/annotated_{src_table_name}")
        rec_tab_support_scores.to_csv(f"../datasets/{hp.dataset}/support_scores/verified_tables/{src_table_name}")
        
        print(f"Reclaimed Table has {reclaimed_tbl.shape[0]} rows, {reclaimed_tbl.shape[1]} columns")
        # -------------------------- END SUPPORT SCORES FOR RECLAIMED TABLE --------------------------
        
        if hp.dataset == "tpch":
            # -------------------------- FINDING SUPPORT SCORES FOR LLM Baseline TABLE --------------------------
            llm_baseline = pd.read_csv(f"../datasets/{hp.dataset}/tables_llm_corrected_baseline/{src_table_name}")
            baseline_num_support_tabs, baseline_tab_support_scores, b_annotated_support = find_support_scores(f"../datasets/{hp.dataset}/tables_llm_corrected_baseline/{src_table_name}", s_key_name, candidate_tbls, benchmark, hp.dataset, src_table_name) 
            display(baseline_tab_support_scores)
            baseline_num_support_tabs.to_csv(f"../datasets/{hp.dataset}/support_scores/llm_corrected_baseline/num_tables_{src_table_name}")
            b_annotated_support.to_csv(f"../datasets/{hp.dataset}/support_scores/llm_corrected_baseline/annotated_{src_table_name}")
            baseline_tab_support_scores.to_csv(f"../datasets/{hp.dataset}/support_scores/llm_corrected_baseline/{src_table_name}")
            
            print(f"LLM Baseline Table has {llm_baseline.shape[0]} rows, {llm_baseline.shape[1]} columns")
            # -------------------------- END SUPPORT SCORES FOR LLM Baseline TABLE --------------------------
