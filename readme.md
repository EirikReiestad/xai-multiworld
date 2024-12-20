## Run
### Install Poetry
pip install poetry

### Install dependencies
poetry install

### Run the project
#### Run examples 
`poetry run python examples **example_name**`

- WandB is used for logging. If you do not have an account or do not want to use it, the `wandb()` methods should not be called.
- `render_mode` should be set to `rgb_array` if you do not want to render the environment.

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
