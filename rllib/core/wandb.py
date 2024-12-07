from abc import ABC
import wandb


class WandB(ABC):
    def __init__(
        self,
        project: str | None,
        run_name: str | None,
        reinit: bool | None,
        tags: list[str] | None,
        dir: str | None,
    ):
        if project is None:
            self._api = None
            return

        self._api = wandb.Api()
        wandb.init(
            project=project,
            name=run_name,
            reinit=reinit,
            tags=tags,
            dir=dir,
        )
