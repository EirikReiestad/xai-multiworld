from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum
from utils.common.model_artifact import ModelArtifact


def create_environment(
    artifact: ModelArtifact,
    width: int | None = None,
    height: int | None = None,
    agents: int | None = None,
    static: bool = True,
):
    _height = artifact.metadata.get("height") or height
    _width = artifact.metadata.get("width") or width
    _agents = artifact.metadata.get("agents") or agents

    height = height or _height
    width = width or _width
    agents = agents or _agents

    environment_type = artifact.metadata.get("environment_type") or "go-to-goal"
    preprocessing = artifact.metadata.get("preprocessing") or "none"

    preprocessing = PreprocessingEnum(preprocessing)

    if environment_type:
        return GoToGoalEnv(
            width=width,
            height=height,
            agents=agents,
            preprocessing=preprocessing,
            static=static,
            render_mode="rgb_array",
        )
    else:
        raise ValueError(
            f"Sorry but environment type of {environment_type} is not supported."
        )
