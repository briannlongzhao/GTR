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
        import wandb

        self._window_size = window_size
        self._run = (
            wandb.init(project=project, entity=entity, config=config, **kwargs) if not wandb.run else wandb.run
        )
        self._run._label(repo="detectron2")

    def write(self, step):
        storage = get_event_storage()
        log_dict = {}
        for k, (v, _) in storage.latest_with_smoothing_hint(self._window_size).items():
            log_dict[k] = v
        self._run.log(log_dict, step=step)

    def close(self):
        self._run.finish()