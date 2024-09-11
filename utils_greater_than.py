import time
from typing import Optional, Callable, List, Tuple, cast, Any, Set, Union, Literal, NamedTuple

import torch
import plotly.express as px
import numpy as np
from transformers import GPT2TokenizerFast

HeadOrMlpType = Union[int, Literal["mlp"]]
AttnSuffixForGpt = Union[Literal[""], Literal[".out"]]

def get_valid_years(
    tokenizer: GPT2TokenizerFast,
    start: int = 1000,
    end: int = 2150,
):
    """Get valid years (_abcd) between [start, end) that are tokenized into
    [_ab, cd] by the input tokenizer. Here _ denotes white space.
    """
    years = [" " + str(year) for year in range(start, end)]
    tokens = tokenizer(years)["input_ids"]
    detokenized = [tokenizer.convert_ids_to_tokens(year_toks) for year_toks in tokens]
    valid = torch.tensor([(len(detok) == 2 and len(detok[1]) == 2) for detok in detokenized])
    last_valid_index = None
    current_century = None
    for i, year in zip(range(len(valid)), range(start, end)):
        cent = year // 100
        if valid[i]:
            if current_century != cent:
                current_century = cent
                valid[i] = False
                if last_valid_index is not None:
                    valid[last_valid_index] = False
            last_valid_index = i
    if last_valid_index is not None:
        valid[last_valid_index] = False
    return torch.arange(start, end)[valid]


def collate(results: torch.Tensor, years: torch.Tensor) -> torch.Tensor:
    return torch.stack([results[years == y].mean(0) for y in range(2, 99)])


def show_mtx(mtx, title="NO TITLE :(", color_map_label="Logit diff variation", **kwargs):
    """Show a plotly matrix with a centered color map. Designed to display results of path patching experiments."""
    # we center the color scale on zero by defining the range (-max_abs, +max_abs)
    max_val = float(max(abs(mtx.min()), abs(mtx.max())))
    x_labels = [f"h{i}" for i in range(12)] + ["mlp"]
    fig = px.imshow(
        mtx,
        title=title,
        labels=dict(x="Head", y="Layer", color=color_map_label),
        color_continuous_scale="RdBu",
        range_color=(-max_val, max_val),
        x=x_labels,
        y=[str(i) for i in range(mtx.shape[0])],
        aspect="equal",
        **kwargs
    )
    fig.update_coloraxes(colorbar_title_side="right")
    return fig



def await_without_await(func: Callable[[], Any]):
    """We want solution files to be usable when run as a script from the command line (where a top level await would
    cause a SyntaxError), so we can do CI on the files. Avoiding top-level awaits also lets us use the normal Python
    debugger.
    Usage: instead of `await cui.init(port=6789)`, write `await_without_await(lambda: cui.init(port=6789))`
    """
    try:
        while True:
            func().send(None)
    except StopIteration:
        pass


def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (torch.Tensor, torch.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")


def imshow(tensor, center_zero=True, zrange=None, color_continuous_scale="RdBu", **kwargs):
    if center_zero:
        return px.imshow(
            to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale=color_continuous_scale, **kwargs
        )
    elif zrange is not None:
        zmin, zmax = zrange
        return px.imshow(
            to_numpy(tensor), zmin=zmin, zmax=zmax, color_continuous_scale=color_continuous_scale, **kwargs
        )
    else:
        return px.imshow(to_numpy(tensor), color_continuous_scale=color_continuous_scale, **kwargs)


def show_diffs(
    diffs, center_zero=True, zrange=None, title="", xlabel="predicted year", zlabel="logit change", dim=500, **kwargs
):
    return imshow(
        diffs,
        center_zero=center_zero,
        zrange=zrange,
        height=dim,
        width=dim,
        title=title,
        labels={"x": xlabel, "y": "YY", "color": zlabel},
        y=[str(i) for i in range(2, 99)],
        **kwargs,
    )

from pathlib import Path

#%%
def mean_logit_diff(logits: torch.Tensor, years: torch.Tensor) -> torch.Tensor:
    diffs = []
    for logit, year in zip(logits, years):
        diffs.append(logit[year + 1 :].sum() - logit[: year + 1].sum())
    return torch.tensor(diffs)

def cutoff_sharpness(logits: torch.Tensor, years: torch.Tensor = torch.arange(2, 99)) -> torch.Tensor:
    sharpness = logits[torch.arange(len(logits)), years + 1] - logits[torch.arange(len(logits)), years - 1]
    return sharpness

def prob_diff(probs: torch.Tensor, years: torch.Tensor) -> torch.Tensor:
    diffs = []
    for prob, year in zip(probs, years):
        diffs.append(prob[year + 1 :].sum() - prob[: year + 1].sum())
    return torch.tensor(diffs)