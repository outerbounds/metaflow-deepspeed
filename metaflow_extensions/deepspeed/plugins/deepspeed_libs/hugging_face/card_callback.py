import datetime
import json
from functools import wraps
from collections import defaultdict
import tempfile
import os
from .constants import DEFAULT_FILE_NAME, DEFAULT_PROFILER_FILE_NAME

try:
    from transformers import TrainerCallback
except ImportError:

    class TrainerCallback:
        pass

class MetaflowHuggingFaceCardCallback(TrainerCallback):
    """
    A basic Huggingface trainer callback that updates a Metaflow card components.
    It updates progress bars, sets parameters, logs metrics, and final outputs.
    """

    MODEL_CARD_ID = "huggingface_model_card"

    PROFILING_CARD_ID = "huggingface_profiling_card"

    DEFAULT_FILE_NAME = DEFAULT_FILE_NAME

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
        self, 
        tracked_metrics=["loss", "grad_norm"], 
        save_file_name=None,
        run_profiling=False,
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


POSSIBLE_METRICS = [
    "cpu_time", 
    "cuda_time", 
    "xpu_time",                
    "cpu_time_total", "cuda_time_total", "xpu_time_total",
    "cpu_memory_usage", "cuda_memory_usage", "xpu_memory_usage",
    "self_cpu_memory_usage", "self_cuda_memory_usage",
    "self_xpu_memory_usage", "count"]


class MetaflowHuggingFaceProfilerCallback(TrainerCallback):

    DEFAULT_FILE_NAME = DEFAULT_PROFILER_FILE_NAME

    def __init__(self, max_steps=100, save_file_name=None, profiler_kwargs=None, tracking_metrics=None):
        super().__init__()
        self._save_file_name = save_file_name or self.DEFAULT_FILE_NAME
        import torch.profiler as profiler_module
        self._profiler_module = profiler_module
        self._profiler_kwargs = profiler_kwargs
        self._max_steps = max_steps
        self._tracking_metrics = tracking_metrics
        self._profiling_finished = False

    def defaults(self):
        _activities = [
            self._profiler_module.ProfilerActivity.CPU,
            self._profiler_module.ProfilerActivity.CUDA,
        ]
        return dict(
            activities=_activities,
            record_shapes=True, 
            use_cuda=True,
            with_stack=True,
            profile_memory=True
        )

    def on_train_begin(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs,
    ):
        if not state.is_local_process_zero:
            return
        if self._profiler_kwargs is None:
            self._profiler_kwargs = self.defaults()
        if self._tracking_metrics is None:
            self._tracking_metrics = POSSIBLE_METRICS

        self._profiler = self._profiler_module.profile(
            **self._profiler_kwargs
        )
        self._profiler.start()

    def _save(self, memory_profile_html_page, metrics_info_dict):
        save_data = {
            "memory_profile_html_page": memory_profile_html_page,
            "metrics_info_dict": metrics_info_dict,
            "created_on": datetime.datetime.now().isoformat(),
        }
        print("Saving Profiling Information to file: ", self._save_file_name)
        with open(self._save_file_name, "w") as f:
            json.dump(save_data, f)
    
    def _save_profiling_information(self):
        _html_file = None
        if getattr(self._profiler, "export_memory_timeline", None):
            # export_memory_timeline is only available in torch>=2.2.0
            with tempfile.NamedTemporaryFile(suffix=".html") as f:
                self._profiler.export_memory_timeline(f.name)
                f.seek(0)
                _html_file = f.read()
        _metrics_info_dict = {}
        for metric in self._tracking_metrics:
            _metrics_info_dict[metric] = self._profiler.key_averages().table(sort_by=metric, row_limit=20)
        # TODO Add Traces over here too!
        self._save(_html_file, _metrics_info_dict)

    def on_step_end(
        self, args, state, control, **kwargs
    ):
        if not state.is_local_process_zero:
            return
        if self._profiling_finished:
            return
        
        if state.global_step > self._max_steps:
            self._profiler.__exit__(None, None, None)
            self._save_profiling_information()
            self._profiling_finished = True
        else:
            self._profiler.step()


    

