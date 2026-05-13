# Configuration Conventions

Training steps keep YAML, a human-readable configuration format, next to the code. Patterns are shared across many steps, but each manifest remains authoritative.

## Layout

For a step directory `src/nemotron/steps/<category>/<name>/`, expect the following files:

- `step.toml` holds the identifier, human title, tags, `[[consumes]]`, `[[produces]]`, `[[parameters]]`, optional `[[strategies]]`, `[[errors]]`, and `[[models]]` blocks.
- `config/default.yaml` holds primary configuration tuned for real workloads.
- `config/tiny.yaml` holds reduced settings for short sample runs and plumbing validation.
- Extra files such as `config/nemo_gym.yaml` appear only on steps that need alternate method profiles.

## CLI Selection

The command `nemotron step run <step_id> -c tiny` passes the config name `tiny`, which resolves to `config/tiny.yaml` inside that step unless the run specification changes the default. Omit `-c` to pick the run specification default, commonly named `default`.

## Overrides

Unknown arguments after the Typer options are treated as dotlist overrides and passthrough material according to the NeMo Run integration described in [Execution through NeMo Run](../../nemo_runspec/nemo-run.md) and in the Nemotron CLI documentation.

## Environment Variables

Profiles in `env.toml`, or in whichever file your workflow selects, carry cluster-specific environment variables and startup commands. Those values are not duplicated inside each step YAML file.

## Related Reading

- [Getting Started](../getting-started.md)
- [Step Catalog (Training)](step-catalog.md)
