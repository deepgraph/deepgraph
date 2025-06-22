# Copyright (C) 2017-2025 by
# Dominik Traxl <dominik.traxl@posteo.org>
# All rights reserved.
# BSD-3-Clause License.

import inspect
import warnings
import ast
import textwrap
from collections import Counter

import numpy as np

from deepgraph._find_selected_indices import _find_selected_indices
from deepgraph.utils import _flatten


class CreatorFunction:
    # dict to store relations
    stored_relations = {}

    # Connector attributes
    c_instances = []
    c_input_features = []
    c_input_rs = []
    c_output_rs = []

    # Selector attributes
    s_instances = []
    s_input_features = []
    s_input_rs = []
    s_output_rs = []

    def __init__(self, fct):
        assert callable(fct), "{} is not callable.".format(fct)

        # make function accessible via self.fct, give self.name
        self.fct = fct
        self.name = fct.__name__

        # find all input arguments
        input_args = list(inspect.signature(fct).parameters.keys())
        self.input_features = [x for x in input_args if x.endswith("_s") or x.endswith("_t")]
        self.input_rs = [
            x for x in input_args if x not in self.input_features and not x == "sources" and not x == "targets"
        ]

        # find all output variables
        output = self._extract_return_variables()
        self.output_rs = [x for x in output if x != "sources" and x != "targets"]
        self.output = output

    @classmethod
    def assertions(cls, v, r_dtype_dic):
        # self.input_features of self.c_instances & self.s_instances
        #     must be in v.columns.values
        # set(cls.c_input_features).issubset(v.columns.values)

        # connectors and selectors must have exclusive output relations
        rs = cls.c_output_rs + cls.s_output_rs
        count_rs = Counter(rs)
        if not len(rs) == 0:
            msg = (
                "There are common output relations in "
                "connectors and/or selectors. \n"
                "[(relation, number of occurences)]: \n {}"
            )
            assert set(count_rs.values()) == {1}, msg.format([(r, nr) for r, nr in count_rs.items() if nr > 1])

        # dtypes for relations given which are not in any output
        unused_dtypes = set(r_dtype_dic.keys()).difference(rs)
        if len(unused_dtypes) != 0:
            warnings.warn(
                "There are dtypes given by 'r_dtype_dic' for which there is no"
                " output variable(s): \n {}".format(list(unused_dtypes)),
                UserWarning,
            )

    @classmethod
    def reset(cls, all_or_WS):
        if all_or_WS == "stored_relations":
            cls.stored_relations = {}
        elif all_or_WS == "all":
            cls.stored_relations = {}
            atrs = [
                atr
                for atr in dir(cls)
                if not atr.startswith("__") and not atr == "stored_relations" and not callable(getattr(cls, atr))
            ]
            for atr in atrs:
                setattr(cls, atr, [])

    @classmethod
    def flatten_variables(cls):
        atrs = [
            atr
            for atr in dir(cls)
            if not atr.startswith("__") and not atr == "stored_relations" and not callable(getattr(cls, atr))
        ]
        for atr in atrs:
            setattr(cls, atr, _flatten(cls.__dict__[atr]))

    def _extract_return_variables(self):
        source = textwrap.dedent(inspect.getsource(self.fct))
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                value = node.value
                if isinstance(value, ast.Tuple):
                    if all(isinstance(elt, ast.Name) for elt in value.elts):
                        return [elt.id for elt in value.elts]
                    else:
                        raise ValueError("Return tuple must contain only variable names.")
                elif isinstance(value, ast.Name):
                    return [value.id]
                else:
                    raise ValueError("Return must consist of variable names only.")

        raise ValueError("No return statement found in the function.")


class Connector(CreatorFunction):
    def __init__(self, fct):
        super(Connector, self).__init__(fct)

        # append to superclass attributes
        self.c_instances.append(self)
        self.c_input_features.append(self.input_features)
        self.c_input_rs.append(self.input_rs)
        self.c_output_rs.append(self.output_rs)

    def map(self, vi, sources, targets, dt_unit, ft_feature):
        # input value dict
        ivdic = {}

        # input features
        for feature in self.input_features:
            if feature == "ft_feature_s":
                ivdic[feature] = vi[ft_feature[0]].values[sources]
            elif feature == "ft_feature_t":
                ivdic[feature] = vi[ft_feature[0]].values[targets]
            else:
                if feature.endswith("_s"):
                    ivdic[feature] = vi[feature[:-2]].values[sources]
                elif feature.endswith("_t"):
                    ivdic[feature] = vi[feature[:-2]].values[targets]

        # input relations
        for r in self.input_rs:
            try:
                ivdic[r] = CreatorFunction.stored_relations[r]
            except KeyError:
                msg = (
                    "{} requests {}, which has not yet "
                    "been computed. Check the order of "
                    "your connectors and selectors.".format(self.name, r)
                )
                raise KeyError(msg)

        # evaluate
        output = self.fct(**ivdic)

        # store relations
        if not isinstance(output, tuple):
            output = (output,)
        for i, r in enumerate(self.output_rs):
            if r == "ft_r" and dt_unit is not None:
                CreatorFunction.stored_relations[r] = output[i] / np.timedelta64(1, dt_unit)
            else:
                CreatorFunction.stored_relations[r] = output[i]


class Selector(CreatorFunction):
    def __init__(self, fct):
        super(Selector, self).__init__(fct)

        if self.name == "_ft_selector":
            self.input_rs.remove("ftt")

        # append to superclass variables
        self.s_instances.append(self)
        self.s_input_features.append(self.input_features)
        self.s_input_rs.append(self.input_rs)
        self.s_output_rs.append(self.output_rs)

    def select_and_store(self, vi, sources, targets, ft_feature, dt_unit):
        # input value dict
        ivdic = {}

        # input features
        for feature in self.input_features:
            if feature.endswith("_s"):
                ivdic[feature] = vi[feature[:-2]].values[sources]
            elif feature.endswith("_t"):
                ivdic[feature] = vi[feature[:-2]].values[targets]

        # input relations
        for r in self.input_rs:
            if r not in CreatorFunction.stored_relations:
                self.request_r(r, vi, sources, targets, dt_unit, ft_feature)
            try:
                ivdic[r] = CreatorFunction.stored_relations[r]
            except KeyError:
                msg = (
                    "{} requests {}, which has not yet "
                    "been computed. Check the order of "
                    "your connectors and selectors.".format(self.name, r)
                )
                raise KeyError(msg)

        # input indices
        ivdic["sources"] = sources
        ivdic["targets"] = targets

        # for the fast track selector, we need the threshold value
        if self.name == "_ft_selector":
            ivdic["ftt"] = ft_feature[1]

        # select and return rs and new node indices
        output = self.fct(**ivdic)

        # output value dict
        ovdic = {}
        for i, name in enumerate(self.output):
            ovdic[name] = output[i]

        # assert that all output_rs have the same shape as the indices
        # PERFORMANCE
        for r in self.output_rs:
            assert len(ovdic[r]) == len(sources), "shape of {} has been modified in {}".format(r, self.name)

        # assert that new sources and target indices have same shape
        # PERFORMANCE
        assert len(ovdic["sources"]) == len(
            ovdic["targets"]
        ), "shape of reduced source and target indices must be the same."

        # store output rs of selectors in CreatorFunction.stored_relations
        for r in self.output_rs:
            CreatorFunction.stored_relations[r] = ovdic[r]

        # positional indices of selected pairs in the former indices
        if not len(ovdic["sources"]) == len(sources):
            index = _find_selected_indices(sources, targets, ovdic["sources"], ovdic["targets"])
        else:
            index = np.arange(len(sources))

        # shrink CreatorFunction.stored_relations by selected indices
        for r in CreatorFunction.stored_relations:
            CreatorFunction.stored_relations[r] = CreatorFunction.stored_relations[r][index]

        # return updated indices
        return ovdic["sources"], ovdic["targets"]

    @staticmethod
    def request_r(r, vi, sources, targets, dt_unit, ft_feature):
        # find the connector mapping to r, evaluate and store
        for connector in CreatorFunction.c_instances:
            if r in connector.output_rs:
                connector.map(vi, sources, targets, dt_unit, ft_feature)
