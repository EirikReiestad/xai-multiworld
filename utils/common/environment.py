from multiworld.multigrid.envs.go_to_goal import GoToGoalEnv
from multiworld.multigrid.utils.preprocessing import PreprocessingEnum
from utils.common.model_artifact import ModelArtifact


def create_environment(artifact: ModelArtifact):
    height = artifact.metadata.get("height") or 10
    width = artifact.metadata.get("width") or 10
    agents = artifact.metadata.get("agents") or 1
    environment_type = artifact.metadata.get("environment_type") or "go-to-goal"
    preprocessing = artifact.metadata.get("preprocessing") or "none"

    preprocessing = PreprocessingEnum(preprocessing)

    if environment_type:
        return GoToGoalEnv(
            width=width,
            height=height,
            agents=agents,
            preprocessing=preprocessing,
            static=True,
            render_mode="rgb_array",
        )
    else:
        raise ValueError(
            f"Sorry but environment type of {environment_type} is not supported."
        )
