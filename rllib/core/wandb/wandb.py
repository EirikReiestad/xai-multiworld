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

        self._log = {}

    def log(self, data: dict):
        if self._api is None:
            return
        wandb.log(data)

    def add_log(self, key: str, value: float, cumulative: bool = False):
        if self._api is None:
            return

        if cumulative:
            self._log[key] = self._log.get(key, 0) + value
        else:
            self._log[key] = value

    def commit_log(self):
        if self._api is None:
            return
        self.log(self._log)
        self._log = {}
