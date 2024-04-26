import os
from metaflow.cards import (
    Markdown,
    Table,
    ProgressBar,
)
from metaflow.decorators import StepDecorator
import math
from metaflow.metaflow_current import current
from ..card_utilities.async_cards import CardRefresher, AsyncPeriodicRefresher
from ..card_utilities.extra_components import ArtifactTable, LineChart
from ..card_utilities.injector import CardDecoratorInjector
from .constants import DEFAULT_FILE_NAME, DEFAULT_PROFILER_FILE_NAME


def json_to_artifact_table(data):
    return ArtifactTable(data)


class HuggingfaceModelMetricsRefresher(CardRefresher):
    CARD_ID = "training_metrics"

    def __init__(self) -> None:
        self._metrics_charts = {}
        self._runtime_info_table = {}
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
        _steps_in_epoch_current = int(
            data["training_state"]["global_step"]
            - (_steps_per_epoch * math.floor(data["training_state"]["epoch"]))
        )
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
                self._runtime_info_table[k] = Markdown(str(data["runtime_info"][k]))
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
                self._runtime_info_table[k].update(str(data_object["runtime_info"][k]))

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
        _steps_in_epoch_current = int(
            data_object["training_state"]["global_step"]
            - (_steps_per_epoch * math.floor(data_object["training_state"]["epoch"]))
        )
        self._progress_bars["epoch_steps"].update(_steps_in_epoch_current)
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
        elif (
            len(data_object["runtime_info"]) > 0 and len(self._runtime_info_table) == 0
        ):
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
        current_card.append(json_to_artifact_table(data["trainer_configuration"]))
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
        required_keys = set(
            [
                "trainer_configuration",
                "model_config",
            ]
        )
        if not data_object_keys.issuperset(required_keys):
            return

        self.render_card_fresh(current_card, data_object)


class HuggingFaceCardDecorator(StepDecorator, CardDecoratorInjector):

    name = "huggingface_card"

    def step_init(self, flow, graph, step_name, decorators, environment, flow_datastore, logger):
        self.attach_card_decorator(
            flow,
            step_name,
            HuggingfaceModelCardRefresher.CARD_ID,
            "blank",
            refresh_interval=10,
        )
        self.attach_card_decorator(
            flow,
            step_name,
            HuggingfaceModelMetricsRefresher.CARD_ID,
            "blank",
            refresh_interval=0.5
        )
    
    def task_decorate(self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context):
        def _wrapped_step_func(*args, **kwargs):
            async_refresher_model_card = AsyncPeriodicRefresher(
                HuggingfaceModelCardRefresher(),
                updater_interval=3,
                collector_interval=2,
                file_name=DEFAULT_FILE_NAME,
            )
            async_refresher_metrics = AsyncPeriodicRefresher(
                HuggingfaceModelMetricsRefresher(),
                updater_interval=1,
                collector_interval=0.5,
                file_name=DEFAULT_FILE_NAME,
            )
            try:
                async_refresher_model_card.start()
                async_refresher_metrics.start()
                return step_func(*args, **kwargs)
            finally:
                async_refresher_model_card.stop()
                async_refresher_metrics.stop()

        return _wrapped_step_func
        