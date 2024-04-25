import os
from metaflow.cards import (
    Markdown,
    Table,
    VegaChart,
    ProgressBar,
    MetaflowCardComponent,
    Artifact
)
import math
from metaflow.plugins.cards.card_modules.components import with_default_component_id, TaskToDict , ArtifactsComponent , render_safely
import datetime
from metaflow.metaflow_current import current
import json
from functools import wraps
from collections import defaultdict
from threading import Thread, Event
import time

try:
    from transformers import (
        TrainerCallback
    )
except ImportError:
    class TrainerCallback:
        pass

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


class MetaflowHuggingFaceCardCallback(TrainerCallback):
    """
    A basic Huggingface trainer callback that updates a Metaflow card components.
    It updates progress bars, sets parameters, logs metrics, and final outputs.
    """

    MODEL_CARD_ID = "huggingface_model_card"

    PROFILING_CARD_ID = "huggingface_profiling_card"

    DEFAULT_FILE_NAME = "huggingface_model_card.json"

    _metrics = defaultdict(list)

    def _save(self):
        save_data = {
            "metrics": self._metrics,
            "model_config": self._model_config,
            "trainer_configuration": self._trainer_configuration,
            "runtime_info": self._runtime_info,
            "training_state": self._state,
            "created_on": datetime.datetime.now().isoformat(),
        }
        with open(self._save_file_name, "w") as f:
            json.dump(save_data, f)

    def _resolve_save_file_name(self, save_file_name):
        if save_file_name is not None:
            return save_file_name
        if os.environ.get("METAFLOW_HF_CARD_SAVE_FILE_NAME", None):
            return os.environ["METAFLOW_HF_CARD_SAVE_FILE_NAME"]
        save_file_name = self.DEFAULT_FILE_NAME
        return save_file_name

    def __init__(
        self, tracked_metrics=["loss", "grad_norm"], save_file_name=None
    ) -> None:
        super().__init__()
        self.tracked_metrics = tracked_metrics
        self._save_file_name = self._resolve_save_file_name(save_file_name)

    def on_train_begin(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs,
    ):
        self._model_config = None if model is None else model.config.to_dict()
        self._trainer_configuration = args.to_dict()
        self._runtime_info = {}
        self._state = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "max_steps": state.max_steps,
            "num_train_epochs": state.num_train_epochs,
        }
        self._training_state = {}

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not state.is_local_process_zero:
            return
        self._state = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "max_steps": state.max_steps,
            "num_train_epochs": state.num_train_epochs,
        }
        for metric in logs:
            if metric in self.tracked_metrics:
                data = {"step": state.global_step, "value": logs[metric]}
                self._metrics[metric].append(data)

        if "train_runtime" in logs:
            self._runtime_info = {
                "Train runtime": logs["train_runtime"],
                "Train samples / sec": logs["train_samples_per_second"],
                "Train steps / sec": logs["train_steps_per_second"],
                "Total Floating point operations": logs["total_flos"],
                "TFLOPs": logs["total_flos"] // 1e12 / logs["train_runtime"],
            }
        self._save()

class ArtifactTable(Artifact):
    def __init__(
        self, data_dict
    ):
        self._data = data_dict
        self._task_to_dict = TaskToDict(only_repr=True)

    @with_default_component_id
    @render_safely
    def render(self):
        _art_list = []
        for k,v in self._data.items():
            _art = self._task_to_dict.infer_object(v)
            _art["name"] = k
            _art_list.append(_art)

        af_component = ArtifactsComponent(data=_art_list)
        af_component.component_id = self.component_id
        return af_component.render()

def json_to_artifact_table(data):
    return ArtifactTable(data)
        

class InfoCollectorThread(Thread):
    def __init__(
        self,
        interval=1,
        file_name=None,
    ):
        super().__init__()
        self._exit_event = Event()
        self._interval = interval
        assert file_name is not None, "file_name must be provided"
        self._file_name = file_name
        self.daemon = True
        self._data = {}
        self._has_errored = False
        self._current_error = None

    def read(self):
        return self._data

    def has_errored(self):
        return self._has_errored

    def get_error(self):
        return self._current_error

    def _safely_load(self):
        try:
            with open(self._file_name, "r") as f:
                return json.load(f), None
        except FileNotFoundError as e:
            return {}, e
        except Exception as e:
            return {}, e

    def run(self):
        while self._exit_event.is_set() is False:
            data, self._current_error = self._safely_load()
            if not self._current_error:
                self._data = data
            self._has_errored = True if self._current_error else False
            time.sleep(self._interval)

    def stop(self):
        self._exit_event.set()
        self.join()


class CardRefresher:

    CARD_ID = None

    def on_startup(self, current_card):
        raise NotImplementedError("make_card method must be implemented")

    def on_error(self, current_card, error_message):
        raise NotImplementedError("error_card method must be implemented")

    def on_update(self, current_card, data_object):
        raise NotImplementedError("update_card method must be implemented")


class CardUpdaterThread(Thread):
    def __init__(
        self,
        card_refresher: CardRefresher,
        interval=1,
        file_name=None,
        collector_thread: InfoCollectorThread = None,
    ):
        super().__init__()
        self._exit_event = Event()
        self._interval = interval
        self._refresher = card_refresher
        self._file_name = file_name
        self._collector_thread = collector_thread
        self.daemon = True

    def run(self):
        if self._refresher.CARD_ID is None:
            raise ValueError("CARD_ID must be defined")
        current_card = current.card[self._refresher.CARD_ID]
        self._refresher.on_startup(current_card)
        while self._exit_event.is_set() is False:
            data = self._collector_thread.read()
            if self._collector_thread.has_errored():
                self._refresher.on_error(
                    current_card, self._collector_thread.get_error()
                )
            self._refresher.on_update(current_card, data)
            time.sleep(self._interval)

    def stop(self):
        self._exit_event.set()
        self._collector_thread.stop()
        self.join()


class HuggingfaceModelMetricsRefresher(CardRefresher):
    CARD_ID = "training_metrics"

    def __init__(self) -> None:
        self._metrics_charts = {}
        self._runtime_info_table = {

        }
        self._progress_bars = {}
        self._last_updated_on = None
        self._already_rendered = False
    
    
    def render_card_fresh(self, current_card, data):
        self._already_rendered = True
        current_card.clear()
        current_card.append(
            Markdown(
                "## Huggingface Model Training Metrics \n## %s [Attempt:%s]"
                % (current.pathspec, current.retry_count),
            )
        )
        self._last_updated_on = Markdown(
            f"_Last data-update on: {data['created_on']}_",
        )
        current_card.append(self._last_updated_on)
        current_card.append(
            Markdown(
                "## Training Progress",
            )
        )
        self._progress_bars["epoch"] = ProgressBar(
            label="Epoch",
            max=data["training_state"]["num_train_epochs"],
            value=round(data["training_state"]["epoch"], 2),
        )
        self._progress_bars["global_step"] = ProgressBar(
            label="Global Step",
            max=data["training_state"]["max_steps"],
            value=data["training_state"]["global_step"],
        )
        _steps_per_epoch = int(
            data["training_state"]["max_steps"]
            / data["training_state"]["num_train_epochs"]
        )
        _steps_in_epoch_current = int(data["training_state"]["global_step"] - (
            _steps_per_epoch * math.floor(data["training_state"]["epoch"])
        ))
        self._progress_bars["epoch_steps"] = ProgressBar(
            label="Steps in Epoch",
            max=_steps_per_epoch,
            value=_steps_in_epoch_current,
        )
        current_card.append(
            Table(
                data=[
                    [
                        self._progress_bars["epoch"],
                    ],
                    [
                        self._progress_bars["global_step"],
                    ],
                    [
                        self._progress_bars["epoch_steps"],
                    ],
                ]
            )
        )
        current_card.append(
            Markdown(
                "## Metrics",
            )
        )
        for metric in data["metrics"]:
            if metric not in self._metrics_charts:
                self._metrics_charts[metric] = LineChart(
                    title=metric,
                    xtitle="Step",
                    ytitle=metric,
                    x_name="step",
                    y_name="value",
                    with_params=False,
                    x_axis_temporal=False,
                )
            self._metrics_charts[metric].spec["data"]["values"] = data["metrics"][
                metric
            ]
            current_card.append(self._metrics_charts[metric], id=metric)
        if len(data["runtime_info"]) > 0:
            current_card.append(
                Markdown(
                    "## Runtime Information",
                )
            )
            for k in data["runtime_info"]:
                self._runtime_info_table[k] = Markdown(
                    str(data["runtime_info"][k])
                )
            current_card.append(
                Table(
                    data=[
                        [Markdown("**%s**" % k), self._runtime_info_table[k]]
                        for k in self._runtime_info_table
                    ]
                )
            )

        current_card.refresh()

    def on_startup(self, current_card):
        current_card.append(
            Markdown(
                "# Huggingface Model Training Metrics\n## %s [Attempt:%s]"
                % (current.pathspec, current.retry_count),
            )
        )
        current_card.append(
            Markdown(
                "_waiting for data to appear_",
            )
        )
        current_card.refresh()

    def on_error(self, current_card, error_message):
        if isinstance(error_message, FileNotFoundError):
            return
        
        if not self._already_rendered:
            current_card.clear()
            current_card.append(
                Markdown(
                    f"## Error: {str(error_message)}",
                )
            )
            current_card.refresh()

    def update_only_components(self, current_card, data_object):
        self._last_updated_on.update(
            f"_Last data-update on: {data_object['created_on']}_"
        )
        if len(data_object["runtime_info"]) > 0:
            for k in data_object["runtime_info"]:
                self._runtime_info_table[k].update(
                    str(data_object["runtime_info"][k])
                )

        for metric in data_object["metrics"]:
            self._metrics_charts[metric].spec["data"]["values"] = data_object[
                "metrics"
            ][metric]

        self._progress_bars["epoch"].update(
            round(data_object["training_state"]["epoch"], 2),
        )
        self._progress_bars["global_step"].update(
            data_object["training_state"]["global_step"],
        )
        _steps_per_epoch = int(
            data_object["training_state"]["max_steps"]
            / data_object["training_state"]["num_train_epochs"]
        )
        _steps_in_epoch_current = int(data_object["training_state"]["global_step"] - (
            _steps_per_epoch * math.floor(data_object["training_state"]["epoch"])
        ))
        self._progress_bars["epoch_steps"].update(
            _steps_in_epoch_current
        )
        current_card.refresh()

    def on_update(self, current_card, data_object):
        data_object_keys = set(data_object.keys())
        if len(data_object_keys) == 0:
            return
        if len(self._metrics_charts) == 0:
            self.render_card_fresh(current_card, data_object)
            return
        elif len(data_object["metrics"]) != len(self._metrics_charts):
            self.render_card_fresh(current_card, data_object)
            return
        elif (len(data_object["runtime_info"]) > 0 and len(self._runtime_info_table) == 0):
            self.render_card_fresh(current_card, data_object)
            return
        else:
            self.update_only_components(current_card, data_object)
            return


class HuggingfaceModelCardRefresher(CardRefresher):
    CARD_ID = "training_variables"

    def __init__(self) -> None:
        self._rendered = False

    def render_card_fresh(self, current_card, data):        
        self._rendered = True
        current_card.clear()
        current_card.append(
            Markdown(
                "# Huggingface Model Training Configuration \n## %s [Attempt:%s]"
                % (current.pathspec, current.retry_count),
            )
        )
        current_card.append(
            Markdown(
                "## Trainer Configuration",
            )
        )
        current_card.append(
            json_to_artifact_table(data["trainer_configuration"])
        )
        current_card.append(
            Markdown(
                "## Model Configuration",
            )
        )
        current_card.append(json_to_artifact_table(data["model_config"]))
        current_card.refresh()

    def on_startup(self, current_card):
        current_card.append(
            Markdown(
                "# Huggingface Model Training Config\n## %s[Attempt:%s]"
                % (current.pathspec, current.retry_count),
            )
        )
        current_card.append(
            Markdown(
                "_waiting for data to appear_",
            )
        )
        current_card.refresh()

    def on_error(self, current_card, error_message):
        if isinstance(error_message, FileNotFoundError):
            return
        if not self._rendered:
            current_card.clear()
            current_card.append(
                Markdown(
                    f"## Error: {str(error_message)}",
                )
            )
            current_card.refresh()

    def on_update(self, current_card, data_object):
        if self._rendered:
            return
        data_object_keys = set(data_object.keys())
        if len(data_object_keys) == 0:
            return
        required_keys = set([
            "trainer_configuration",
            "model_config",
        ])
        if not data_object_keys.issuperset(required_keys):
            return

        self.render_card_fresh(current_card, data_object)


class AsyncPeriodicRefresher:

    def __init__(self, card_referesher: CardRefresher, updater_interval=1, collector_interval=1, file_name=None):
        assert card_referesher.CARD_ID is not None, "CARD_ID must be defined"
        self._collector_thread = InfoCollectorThread(
            interval=collector_interval, file_name=file_name
        )
        self._collector_thread.start()
        self._updater_thread = CardUpdaterThread(
            card_refresher=card_referesher,
            interval=updater_interval,
            file_name=file_name,
            collector_thread=self._collector_thread,
        )

    def start(self):
        self._updater_thread.start()

    def stop(self):
        self._updater_thread.stop()
        self._collector_thread.stop()


def huggingface_card(func):
    from metaflow import card

    @wraps(func)
    def wrapper(*args, **kwargs):
        async_refresher_model_card = AsyncPeriodicRefresher(
            HuggingfaceModelCardRefresher(), 
            updater_interval=3, 
            collector_interval=2,
            file_name=MetaflowHuggingFaceCardCallback.DEFAULT_FILE_NAME
        )
        async_refresher_metrics = AsyncPeriodicRefresher(
            HuggingfaceModelMetricsRefresher(),
            updater_interval=1, 
            collector_interval=0.5,
            file_name=MetaflowHuggingFaceCardCallback.DEFAULT_FILE_NAME
        )
        try:
            async_refresher_model_card.start()
            async_refresher_metrics.start()
            func(*args, **kwargs)
        finally:
            async_refresher_model_card.stop()
            async_refresher_metrics.stop()

    _model_card = card(id=HuggingfaceModelCardRefresher.CARD_ID, type="blank", refresh_interval=0.5)
    _metrics_card = card(id=HuggingfaceModelMetricsRefresher.CARD_ID, type="blank", refresh_interval=0.5)
    return _metrics_card(_model_card(wrapper))