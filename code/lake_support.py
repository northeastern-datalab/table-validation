'''
Given a Source Table and a Data Lake,
for each value in the Source Table, find how much "support" it has from the Data Lake (how often it occurs)
Assuming our Source Table has a key, and schema matching was performed between the Source and all DL tables.
'''
import glob
import pandas as pd
from IPython.display import display
from tqdm import tqdm

# number of columns ALL results should have (aligned columns with source table)
col_threshold = 2
    
def count_equal_row(row1, row2):
    num_equal = 0
    for ind in range(len(row1)):
        if row1[ind] and row1[ind] == row2[ind]: num_equal += 1
        else:
            if pd.isna(row1[ind]) and pd.isna(row2[ind]): num_equal += 1
    return num_equal


def find_largest_overlap(s_row, s_key_name, df):
    '''
    For each row in the source table, find a row in the dl table that it has the largest overlap with
    '''
    overlaps = {}
    # if key column is in dataframe
    if s_key_name in df.columns:
        if s_row[s_key_name] in df[s_key_name].tolist():
            for index, row2 in df.loc[df[s_key_name] == s_row[s_key_name]].iterrows():
                # overlap_count = sum(s_row == row2)
                overlap_count = count_equal_row(s_row, row2)
                overlaps[index] = overlap_count
    else:
        return None
    check_overlap = all(value == 0 for value in overlaps.values())
    if check_overlap: return None
    max_overlap_index = max(overlaps, key=overlaps.get)
    # save and return a dictionary of overlapping values (val of dict) and the columns they're in (key of dict)
    largest_overlap = df.iloc[max_overlap_index].to_dict()    
    print("largest_overlap", largest_overlap)
    remove_cols = []
    for col, val in largest_overlap.items():
        # if both are not null and don't equal
        if not pd.isna(val) and not pd.isna(s_row[col]) and val != s_row[col]: remove_cols.append(col)
        # if val is null
        elif pd.isna(val) and not pd.isna(s_row[col]): remove_cols.append(col)
        # if source val is null
        elif pd.isna(s_row[col]) and not pd.isna(val): remove_cols.append(col)
    for col in remove_cols:
        del largest_overlap[col]
    # return a dictionary of column name: overlapping value at that column with source row
    return largest_overlap

def find_all_overlaps(s_row, s_key_name, df):
    '''
    For each row in the source table, find a row in the dl table that it has the largest overlap with
    '''
    overlaps = {}
    # if key column is in dataframe
    if s_key_name in df.columns:
        if s_row[s_key_name] in df[s_key_name].tolist():
            for index, row2 in df.loc[df[s_key_name] == s_row[s_key_name]].iterrows():
                overlap_count = count_equal_row(s_row, row2)
                overlaps[index] = overlap_count
    else:
        for index, row2 in df.iterrows():
            overlap_count = count_equal_row(s_row, row2)
            overlaps[index] = overlap_count
    check_overlap = all(value == 0 for value in overlaps.values())
    if check_overlap: return None
    max_overlap_index = max(overlaps, key=overlaps.get)
    # save and return a dictionary of overlapping values (val of dict) and the columns they're in (key of dict)
    all_overlaps = []
    for index in overlaps:
        # if overlaps[index] < 2: continue
        overlap_dict = df.iloc[index].to_dict()    
        remove_cols = []
        for col, val in overlap_dict.items():
            # if both are not null and don't equal
            if not pd.isna(val) and not pd.isna(s_row[col]) and val != s_row[col]: remove_cols.append(col)
            # if val is null
            elif pd.isna(val) and not pd.isna(s_row[col]): remove_cols.append(col)
            # if source val is null
            elif pd.isna(s_row[col]) and not pd.isna(val): remove_cols.append(col)
        for col in remove_cols:
            del overlap_dict[col]
        all_overlaps.append(overlap_dict)
    # return a dictionary of column name: overlapping value at that column with source row
    return all_overlaps

def align_values(projected_s_tab, s_key, projected_dl_tab):
    '''
    Given a source table and a data lake table, both projected on commmon columns,
    return a dictionary of key value: overlapping values at that row
    '''
    overlaps_dict = {}
    for s_row_indx, s_row in projected_s_tab.iterrows():
        # largest_row_overlap = find_largest_overlap(s_row, s_key.name, projected_dl_tab)
        # if largest_row_overlap:
        #     overlaps_dict[s_key.at[s_row_indx]] = largest_row_overlap
        all_row_overlaps = find_all_overlaps(s_row, s_key.name, projected_dl_tab)
        if all_row_overlaps:
            overlaps_dict[s_key.at[s_row_indx]] = all_row_overlaps
    return overlaps_dict


def assign_relationship_scores(aligned_vals, key_col_name, ind_conf_scores, annotated_support, curr_tbl):  
    '''
    Once rows are aligned, assign confidence scores (# supporting tables) for each key-value pair
    '''      
    for key_val, aligned_list in aligned_vals.items():
        for aligned_col_vals in aligned_list:
            if key_col_name not in aligned_col_vals:
                if key_val != aligned_col_vals[key_col_name]: continue #key value not found in aligned dl values
            for col_name, col_val in aligned_col_vals.items():
                ind_conf_scores.at[key_val, col_name] += 1
                annotated_support.at[key_val, col_name] += curr_tbl + "|"
    return ind_conf_scores, annotated_support


def preprocess_datalake(all_tbls, candidate_tbl_dict):
    tableDfs = {}
    for table in all_tbls:
        table_name = table.split("/")[-1]
        if table_name in candidate_tbl_dict:
            df = pd.read_csv(table)
            df = df[df.columns.intersection(list(candidate_tbl_dict[table_name].keys()))]
            df = df.rename(columns=candidate_tbl_dict[table_name])
            tableDfs[table_name] = df
    return tableDfs
    

def find_lake_support(s_tab_path, key_col_name, dl_tables, benchmark="tpch"):
    '''
    Arguments:
        s_tab_name: path to the Source Table
        key_col_name: name of the key column in the Source Table
        dl_tables: set of relevant tables in data lake
        benchmark: name of the benchmark
    '''
    BENCHMARK_PATH = f"../datasets/{benchmark}"
    DATALAKE_PATH = f"{BENCHMARK_PATH}/datalake"

    # Get all sources and data lake table in the benchmark
    all_dl_tables = glob.glob(f"{DATALAKE_PATH}/*.csv")
    # Get a source table
    src_table = f"{s_tab_path}"
    s_tab = pd.read_csv(src_table)
    s_key = s_tab[key_col_name]
    
    
    table_df_dict = preprocess_datalake(all_dl_tables, dl_tables)
    print(f"Processing {len(table_df_dict)} datalake tables.")
      
    total_relationship_related_tables = set()
    # dataframe with same dimensions as Source Table: each index contains confidence/evidence numerator score (# dl tables that has this value)
    raw_val_pair_conf_scores = pd.DataFrame(0, columns=s_tab.columns, index=s_key)
    annotated_conf_table = pd.DataFrame("", columns=s_tab.columns, index=s_key)
    for tbl_name, dl_tab in tqdm(table_df_dict.items()):
        common_columns = [col for col in s_tab.columns if col in dl_tab.columns]
        # check if there are at least 'col_threshold' common columns
        if len(common_columns) < col_threshold: continue
        projected_dl_tab = dl_tab[list(common_columns)]
        projected_s_tab = s_tab[list(common_columns)]
        # dictionary of key value: aligned values in source row
        aligned_vals = align_values(projected_s_tab, s_key, projected_dl_tab)
        if not aligned_vals:  continue
        if key_col_name not in projected_dl_tab: continue
        raw_val_pair_conf_scores, annotated_conf_table = assign_relationship_scores(aligned_vals, key_col_name, raw_val_pair_conf_scores, annotated_conf_table, tbl_name)
        total_relationship_related_tables.add(tbl_name)
    display(annotated_conf_table)
    
    
    # divide current scores (numerators) by total number of related tables (denominator) in dataframe
    val_pair_conf_scores = raw_val_pair_conf_scores.applymap(lambda x: x / len(dl_tables))
    
    print(f"There are {len(total_relationship_related_tables)} dl tables with shared value pairs with Source Table {src_table.split('/')[-1]}, out of {len(dl_tables)} candidate tables") 
    return raw_val_pair_conf_scores, val_pair_conf_scores, total_relationship_related_tables, annotated_conf_table

