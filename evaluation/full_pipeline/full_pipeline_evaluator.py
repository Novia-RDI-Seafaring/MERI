import json
import sys
import os
base_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(base_path)

from evaluation.datasets import PopulatedSchemaDataset
import json
import glob
from pathlib import Path
from meri.meri import MERI
from jsonpath_ng.jsonpath import DatumInContext
from jsonpath_ng import parse
from typing import List
import fnmatch
import re

# iterate through gt to list of json queries and gt info for this query

def build_jsonpath_queries(d, current_path='$', ignore_keys=[], termination_keys = []):
    """Builds list of jsonqueries recursivly. Per default a query for each key and value for the entire dictionary will be created.

    Args:
        d (_type_): Dictionary
        current_path (str, optional): jsonpath query that points to the directory d in the original dictionary. Defaults to '$'.
        ignore_keys (list, optional): keys in the dictionary for which no queries should be generated. Queries for the following hierarchy will
            still be built. Defaults to [].
        termination_keys (list, optional): keys in the dictionary for which the following subdict in the hierarchy will be ignored. Defaults to [].

    Returns:
        _type_: _description_
    """
    queries = []
    
    if isinstance(d, dict):
        for key, value in d.items():               
            new_path = f"{current_path}.{key}"
            if key not in ignore_keys:
                queries.append(new_path)
            if key not in termination_keys: 
                queries.extend(build_jsonpath_queries(value, new_path, ignore_keys, termination_keys))
    elif isinstance(d, list):
        for index, value in enumerate(d):
            new_path = f"{current_path}[{index}]"
            queries.append(new_path)
            queries.extend(build_jsonpath_queries(value, new_path, ignore_keys, termination_keys))
    
    return queries


class ExtractionResultItem:

    def __init__(self, json_query, gt_value, pred_value=None) -> None:
        self.json_query = json_query
        self.gt_value = gt_value
        self.pred_value = pred_value
    
    def to_dict(self):
        return self.__dict__
    
class ExtractionResults:

    def __init__(self) -> None:
        self.elements: List[ExtractionResultItem] = []

    def group_by_path(self, jsonpath: str):
        #$.technicalSpecifications.HE16_DESIGN_POWER

        subset_results = ExtractionResults()
        for sub_element in list(filter(lambda d: fnmatch.fnmatch(d.json_query, jsonpath+"*"), self.elements)):
            subset_results.append(sub_element)

        return subset_results

    def children_as_group(self, hierarchy_jsonpath: str):
        # $.technicalSpecifications gives all results below the hierarch level as separate groups

        pattern = f"{hierarchy_jsonpath.replace('$', "\\$").replace('.', "\\.")}"+r"\.([^.]*)" #r"\$\.technicalSpecifications\.([^.]*)"

        paths = []
        for el in self.elements:
            match = re.search(pattern, el.json_query)
            if match and match.group(0) not in paths:
                paths.append(match.group(0))
        
        return [self.group_by_path(path) for path in paths]

    @property
    def longest_subquery(self):
        all_json_queries = [el.json_query for el in self.elements]

        # Start with the shortest string as the base
        shortest_string = min(all_json_queries, key=len)
        length = len(shortest_string)

        longest_substr = ""

        # Check all substrings of the shortest string
        for i in range(length):
            for j in range(i + 1, length + 1):
                candidate = shortest_string[i:j]
                if all(candidate in string for string in all_json_queries):
                    if len(candidate) > len(longest_substr):
                        longest_substr = candidate
        
        if longest_substr.endswith('.'):
            longest_subquery = longest_substr[:-1]
        else:
            longest_subquery = '.'.join(longest_substr.split('.')[:-1])
        
        return longest_subquery

    @property
    def n(self):
        return len(self.elements)

    def append(self, item: ExtractionResultItem):
        self.elements.append(item)
    
    def to_list(self):
        return [item.to_dict() for item in self.elements]