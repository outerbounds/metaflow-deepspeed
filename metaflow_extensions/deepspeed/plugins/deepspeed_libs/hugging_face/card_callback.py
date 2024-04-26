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
                # "Total Floating point operations": logs["total_flos"],
                # "TFLOPs": logs["total_flos"] // 1e12 / logs["train_runtime"],
            }
        self._save()