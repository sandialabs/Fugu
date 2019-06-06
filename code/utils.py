import networkx as nx
import pandas as pd
import numpy as np
from collections import deque

def results_df_from_dict(results_dictionary, key, value):
    df = pd.DataFrame()
    for k in results_dictionary:
        df2 = pd.DataFrame()
        store = results_dictionary[k]
        extended_keys = [k]*len(store)
        df2[key] = extended_keys
        df2[value] = store
        df = df.append(df2, ignore_index=True, sort=False)
    return df

def fill_results_from_graph(results_df, scaffold, fields = ['time', 'neuron_number', 'name', 'brick'], unmatched_fields=['time']):
    field_values = {'name':deque()}
    fields.remove('name')
    for node in scaffold.graph.nodes:
        field_values['name'].append(node)
        for field in [field for field in fields if field not in unmatched_fields ]:
            if field not in field_values:
                field_values[field] = deque()
            if field in scaffold.graph.nodes[node]:
                field_values[field].append(scaffold.graph.nodes[node][field])
            else:
                field_values[field].append(np.NaN)
    field_atlas = pd.DataFrame(field_values)
    return results_df.merge(field_atlas, how='left')

