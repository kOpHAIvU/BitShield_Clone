import tvm.relay as r
import tvm.ir as ir
from tvm.ir import IRModule
from tvm.ir.transform import PassContext
import tvm
import numpy as np

from cgutils import *

def upruned_legacy(params_dict, prunef=None, prune_dict=None, threshold=None, random_frac=None, percentile=None):
    """Note that threshold and percentile work in terms of absolute values."""

    assert sum(1 for x in [prunef, prune_dict, threshold, random_frac, percentile] if x) == 1
    np_weights = {k: v.numpy() if isinstance(v, tvm.nd.NDArray) else v for k, v in params_dict.items() if len(v.shape) == 4}
    if prunef:
        prunef(np_weights)
    elif prune_dict:
        for k, v in prune_dict.items():
            np_weights[k][v] = 0
    elif threshold:
        for k, v in np_weights.items():
            abs_v = np.abs(v)
            v[abs_v < threshold] = 0
    elif random_frac:
        for k, v in np_weights.items():
            idxs = np.random.choice(np.arange(v.size), replace=False, size=int(v.size * random_frac))
            v[idxs] = 0
    elif percentile:
        for k, v in np_weights.items():
            abs_v = np.abs(v)
            v[abs_v < np.percentile(abs_v, percentile)] = 0
    else:
        assert False
    pruned_params = params_dict.copy()
    for k, v in np_weights.items():
        pruned_params[k] = tvm.nd.array(v)
    return pruned_params

def get_ignored_components_legacy(params, fi_frac, li_frac, nws=True, as_eps=False, irmod=None):
    """Given a dict of params (supposedly after unstructured pruning) and an irmod,
    returns a set of layer weight names to ignore, as well as a map from weight
    names to neuron indices to ignore (the map will be empty if nws is False).
    If (fi_frac * 100)% of the weights in a filter are zero, the filter is flagged.
    If (li_frac * 100)% of the filters in a layer are flagged, the layer is ignored.
    If as_eps is True, layer weight names will be converted to the names of their
    corresponding extra params. irmod is required in this case."""

    ignored_weights = set()
    ignored_neurons = {}

    for pname, param in params.items():
        if isinstance(param, tvm.nd.NDArray):
            param = param.numpy()
        if len(param.shape) != 4:
            continue
        for i, f in enumerate(param):
            if np.count_nonzero(f == 0) / f.size > fi_frac:
                if pname not in ignored_neurons:
                    ignored_neurons[pname] = set()
                ignored_neurons[pname].add(i)
        if len(ignored_neurons.get(pname, ())) / param.shape[0] > li_frac:
            ignored_weights.add(pname)

    if as_eps:
        assert irmod
        weight_names = []
        class WeightNamesFinder(ConvishVisitor):
            def __init__(self):
                super().__init__(post_order=True)
            def visit_convish(self, convish):
                weight_names.append(convish_weight(convish).name_hint)
        WeightNamesFinder().visit(irmod['main'])
        weights_to_eps = {wname: f'__ep_{i}' for i, wname in enumerate(weight_names)}
        ignored_weights = set([weights_to_eps[wname] for wname in ignored_weights])
        ignored_neurons = {
            weights_to_eps[wname]: ignored_neurons[wname] for wname in ignored_neurons
        }

    if not nws:
        ignored_neurons = {}

    print(f'Ignoring {len(ignored_weights)} weights; and also {sum(len(ignored_neurons[k]) for k in ignored_neurons)} neurons in {len(ignored_neurons)} weights.')
    return ignored_weights, ignored_neurons

def get_ignored_components(params, frac, nws=False, as_eps=False, irmod=None, is_random=False):
    """Given a dict of params and an irmod, returns a set of layer weight names
    to ignore, as well as a map from weight names to neuron indices to ignore
    (the map will be empty if nws is False).
    This function uses the MinWeight metrics on layers if nws is False and on
    filters if it's True.
    If as_eps is True, layer weight names will be converted to the names of their
    corresponding extra params. irmod is required in this case.
    If is_random, the components to ignore are randomly selected."""

    conv_params = {}
    for pname, param in params.items():
        if isinstance(param, tvm.nd.NDArray):
            param = param.numpy()
        if len(param.shape) != 4:
            continue
        conv_params[pname] = param

    ignored_weights = set()
    ignored_neurons = {}

    mw = lambda weights: (weights**2).sum() / weights.size
    sorted_scores_dict = lambda d: dict(sorted(d.items(), key=lambda x: x[1]))

    if not nws:
        if not is_random:
            param_scores = {pname: mw(param) for pname, param in conv_params.items()}
            param_scores = sorted_scores_dict(param_scores)
            ignored_weights = set(list(param_scores.keys())[:int(frac * len(param_scores))])
        else:
            pnames = list(conv_params.keys())
            ignored_weights = set(np.random.choice(pnames, size=int(frac * len(pnames)), replace=False))
    else:
        assert not is_random, NotImplemented
        neuron_scores = {}
        for pname, param in conv_params.items():
            for i, f in enumerate(param):
                neuron_scores[(pname, i)] = mw(f)
        neuron_scores = sorted_scores_dict(neuron_scores)
        ignored_neurons_list = list(neuron_scores.keys())[:int(frac * len(neuron_scores))]
        for pname, i in ignored_neurons_list:
            if pname not in ignored_neurons:
                ignored_neurons[pname] = set()
            ignored_neurons[pname].add(i)
        # If all neurons in a layer are ignored, use LWS to ignore the layer instead
        ignored_weights = set([
            pname
            for pname, neurons in ignored_neurons.items()
            if len(neurons) == conv_params[pname].shape[0]
        ])
        ignored_neurons = {k: v for k, v in ignored_neurons.items() if k not in ignored_weights}

    if as_eps:
        assert irmod
        weight_names = []
        class WeightNamesFinder(ConvishVisitor):
            def __init__(self):
                super().__init__(post_order=True)
            def visit_convish(self, convish):
                weight_names.append(convish_weight(convish).name_hint)
        WeightNamesFinder().visit(irmod['main'])
        weights_to_eps = {wname: f'__ep_{i}' for i, wname in enumerate(weight_names)}
        ignored_weights = set([weights_to_eps[wname] for wname in ignored_weights])
        ignored_neurons = {
            weights_to_eps[wname]: ignored_neurons[wname] for wname in ignored_neurons
        }

    print(f'Ignoring {len(ignored_weights)} weights; and also {sum(len(ignored_neurons[k]) for k in ignored_neurons)} neurons in {len(ignored_neurons)} weights.')
    return ignored_weights, ignored_neurons

def ignored_neurons_applied_to_extra_params(params, mod, ignored_neurons, mode, eps_mode=False):
    """Given the ignored_neurons generated by get_ignored_components,
    returns a dict of params with the ignored neurons applied to their
    corresponding extra params.
    If as_eps was True in get_ignored_components, enable eps_mode here (in
    which case mod is not needed)."""

    # We first use a ConvishVisitor to collect all conv weight names
    weight_names = []
    if eps_mode:
        weight_names = list(ignored_neurons.keys())
    else:
        class WeightNamesFinder(ConvishVisitor):
            def __init__(self):
                super().__init__(post_order=True)
            def visit_convish(self, convish):
                weight_names.append(convish_weight(convish).name_hint)
        WeightNamesFinder().visit(mod['main'])

    ret = params.copy()
    ndeleted = 0
    for i, wn in enumerate(weight_names):
        if wn not in ignored_neurons:
            continue
        epn = wn if eps_mode else f'__ep_{i}'
        epp = params[epn]
        if isinstance(epp, tvm.nd.NDArray):
            epp = epp.numpy()
        ndeleted += len(ignored_neurons[wn])
        epp = np.delete(epp, list(ignored_neurons[wn]), axis=-1)
        if mode == 'covar':
            epp = np.delete(epp, list(ignored_neurons[wn]), axis=-2)
        if isinstance(epp, np.ndarray):
            epp = tvm.nd.array(epp)
        ret[epn] = epp

    print(f'Deleted {ndeleted} ignored neurons in extra params.')

    return ret

def calc_uprune_stats(mod, ignored_weights, ignored_neurons, eps_mode=False):
    """Returns the number of ignored neurons (whether ignored by whole layer or
    individual neurons) and total neuron count of the model.
    Supports as_eps mode of get_ignored_components through the eps_mode param."""

    # TODO: Also count layers ignored

    total_ignored = 0
    total_neurons = 0
    convish_idx = 0
    class Counter(ConvishVisitor):
        def __init__(self):
            super().__init__(post_order=True)
        def visit_convish(self, convish):
            nonlocal total_ignored, total_neurons, convish_idx
            w = convish_weight(convish)
            wn = f'__ep_{convish_idx}' if eps_mode else w.name_hint
            wneurons = get_type(w).concrete_shape[0]
            total_neurons += wneurons
            if wn in ignored_weights:
                total_ignored += wneurons
            elif wn in ignored_neurons:
                total_ignored += len(ignored_neurons[wn])
            convish_idx += 1
    Counter().visit(mod['main'])
    return total_ignored, total_neurons

# vim: set fdm=marker:
