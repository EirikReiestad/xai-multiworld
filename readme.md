# xai-multiworld

## Description
`xai-multiworld` is a Python-based library for multi-agent reinforcement learning environments. This library includes functionalities for creating, training, and evaluating multi-agent systems with a focus on explainable AI (XAI) techniques.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [License](#license)
- [Contact](#contact)
- [Naming conventions for git](#naming-conventions-for-git)

## Installation
To install the dependencies, use the following commands:
```sh
pip install poetry
poetry install
```
## Usage
To run an example using the DQN algorithm, use the following command:
```sh
poetry run python examples/**example_name**
```

- WandB is used for logging. If you do not have an account or do not want to use it, the `wandb()` methods should not be called.
- `render_mode` should be set to `rgb_array` if you do not want to render the environment.
  
For more examples and usage instructions, refer to the examples directory.

## Features
- Multi-Agent Environments: Support for creating and managing multi-agent environments.
- Reinforcement Learning Algorithms: Integrated with various RL algorithms like DQN.
- Explainable AI: Tools for generating explanations for agent behaviors.
- Visualization: Utilities for visualizing training progress and agent interactions.

## License
This project is licensed under the MIT License.

## Contact
For any questions or suggestions, please contact the repository owner Eirik Reiestad (https://github.com/EirikReiestad).

---

You can view more details and explore the repository here: https://github.com/EirikReiestad/xai-multiworld.

## Naming conventions for git
### Git branch prefixes
- `feature/` - these branches are used to develop new features 
- `bugfix/` - these branches are used to make fixes 
- `release/` - these branches prepare the codebase for new releases
- `hotfix/` - these branches addresses urgent issues in production

### Issues examples
feature/add-user-authentication

bugfix/fix-login-page

release/v1.0

hotfix/fix-login-bug

### Commit messages
- `feat:` - new feature
- `fix:` - bug fix
- `refactor:` - code refactoring
- `docs:` - changes in documentation
- `model:` - changes to the model parameters
- `idun:` - changes to IDUN 
