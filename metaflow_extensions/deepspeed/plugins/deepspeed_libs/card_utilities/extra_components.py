import os
from metaflow.cards import (
    Markdown,
    Table,
    VegaChart,
    ProgressBar,
    MetaflowCardComponent,
    Artifact,
)
import math
from metaflow.plugins.cards.card_modules.components import (
    with_default_component_id,
    TaskToDict,
    ArtifactsComponent,
    render_safely,
)
import datetime
from metaflow.metaflow_current import current
import json
from functools import wraps
from collections import defaultdict
from threading import Thread, Event
import time



def update_spec_data(spec, data):
    spec["data"]["values"].append(data)
    return spec


def update_data_object(data_object, data):
    data_object["values"].append(data)
    return data_object


def line_chart_spec(
    title=None,
    x_name="u",
    y_name="v",
    xtitle=None,
    ytitle=None,
    width=600,
    height=400,
    with_params=True,
    x_axis_temporal=False,
):
    parameters = [
        {
            "name": "interpolate",
            "value": "linear",
            "bind": {
                "input": "select",
                "options": [
                    "basis",
                    "cardinal",
                    "catmull-rom",
                    "linear",
                    "monotone",
                    "natural",
                    "step",
                    "step-after",
                    "step-before",
                ],
            },
        },
        {
            "name": "tension",
            "value": 0,
            "bind": {"input": "range", "min": 0, "max": 1, "step": 0.05},
        },
        {
            "name": "strokeWidth",
            "value": 2,
            "bind": {"input": "range", "min": 0, "max": 10, "step": 0.5},
        },
        {
            "name": "strokeCap",
            "value": "butt",
            "bind": {"input": "select", "options": ["butt", "round", "square"]},
        },
        {
            "name": "strokeDash",
            "value": [1, 0],
            "bind": {
                "input": "select",
                "options": [[1, 0], [8, 8], [8, 4], [4, 4], [4, 2], [2, 1], [1, 1]],
            },
        },
    ]
    parameter_marks = {
        "interpolate": {"expr": "interpolate"},
        "tension": {"expr": "tension"},
        "strokeWidth": {"expr": "strokeWidth"},
        "strokeDash": {"expr": "strokeDash"},
        "strokeCap": {"expr": "strokeCap"},
    }
    spec = {
        "title": title if title else "Line Chart",
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        # "width": width,
        # "height": height,
        "params": parameters if with_params else [],
        "data": {"name": "values", "values": []},
        "mark": {
            "type": "line",
            "tooltip": True,
            **(parameter_marks if with_params else {}),
        },
        "selection": {"grid": {"type": "interval", "bind": "scales"}},
        "encoding": {
            "x": {
                "field": x_name,
                "title": xtitle if xtitle else x_name,
                **({"timeUnit": "seconds"} if x_axis_temporal else {}),
                **({"type": "quantitative"} if not x_axis_temporal else {}),
            },
            "y": {
                "field": y_name,
                "type": "quantitative",
                "title": ytitle if ytitle else y_name,
            },
        },
    }
    data = {"values": []}
    return spec, data


class LineChart(MetaflowCardComponent):
    REALTIME_UPDATABLE = True

    def __init__(
        self,
        title,
        xtitle,
        ytitle,
        x_name,
        y_name,
        with_params=False,
        x_axis_temporal=False,
    ):
        super().__init__()

        self.spec, _ = line_chart_spec(
            title=title,
            xtitle=xtitle,
            ytitle=ytitle,
            x_name=x_name,
            y_name=y_name,
            with_params=with_params,
            x_axis_temporal=x_axis_temporal,
        )

    def update(self, data):  # Can take a diff
        self.spec = update_spec_data(self.spec, data)

    @with_default_component_id
    def render(self):
        vega_chart = VegaChart(self.spec, show_controls=True)
        vega_chart.component_id = self.component_id
        return vega_chart.render()


class ArtifactTable(Artifact):
    def __init__(self, data_dict):
        self._data = data_dict
        self._task_to_dict = TaskToDict(only_repr=True)

    @with_default_component_id
    @render_safely
    def render(self):
        _art_list = []
        for k, v in self._data.items():
            _art = self._task_to_dict.infer_object(v)
            _art["name"] = k
            _art_list.append(_art)

        af_component = ArtifactsComponent(data=_art_list)
        af_component.component_id = self.component_id
        return af_component.render()

