import wandb
import numpy as np
from typing import Dict, Union
from detectron2.utils.events import EventWriter, get_event_storage
from detectron2.config import CfgNode

class WandbWriter(EventWriter):
    """
    Write all scalars to a wandb tool.
    """

    def __init__(
        self,
        project: str = "detectron2",
        entity: str = "briannlongzhao",
        config: Union[Dict, CfgNode] = {},
        window_size: int = 20,
        **kwargs,
    ):
        """
        Args:
            project (str): W&B Project name
            entity (str): W&B username
            config Union[Dict, CfgNode]: the project level configuration object
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `wandb.init(...)`
        """
        self._window_size = window_size
        self._run = (
            wandb.init(project=project, entity=entity, config=config, **kwargs) if not wandb.run else wandb.run
        )
        self._run._label(repo="detectron2")
        self.mot_count_metrics = [
            "Dets","GT_Dets","IDs","GT_IDs","IDTP","IDFN","IDFP","CLR_TP","CLR_FN","CLR_FP","IDSW","MT","PT","ML",
            "Frag","HOTA_TP","HOTA_FN","HOTA_FP","CLR_Frames"
        ]

    def write(self, step):
        storage = get_event_storage()
        log_dict = {}
        for k, (v, _) in storage.latest_with_smoothing_hint(self._window_size).items():
            log_dict[k] = v
        self._run.log(log_dict, step=step)

    def log_video(self, video, caption):
        self._run.log({"video": wandb.Video(video, caption=caption, fps=4)})

    def log_results(self, results):
        for task, result in results.items():
            columns, data = [], []
            if task == "bbox":
                for ap, value in result.items():
                    columns.append(ap)
                    data.append(value)
                table = wandb.Table(columns=columns, data=[data])
                self._run.log({task: table})
            elif task == "MotChallenge2DBox":
                rows, columns, data = [], [], []
                num_metrics = 0
                for metric_group, metrics in result["pred"]["COMBINED_SEQ"]["pedestrian"].items():
                    num_metrics += len(metrics)
                    columns += metrics.keys()
                for seq_name, seq_res in result["pred"].items():
                    rows.append(seq_name)
                    data_row = [0] * num_metrics
                    for metric_group, metrics in seq_res["pedestrian"].items():
                        for metric, values in metrics.items():
                            if metric not in columns:
                                continue
                            value = np.average(values) if type(values) is np.ndarray else values
                            value = value*100 if metric not in self.mot_count_metrics else value
                            data_row[columns.index(metric)] = value
                    data.append(data_row)
                table = wandb.Table(rows=rows, columns=columns, data=data)
                table.add_column(name="Sequence", data=rows)
                self._run.log({task: table})

    def watch(self, model, log="all", log_graph=False):
        self._run.watch(model, log=log, log_graph=log_graph)

    def close(self):
        self._run.finish()