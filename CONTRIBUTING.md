# Contributing

Thank you for helping improve `crowd-detection`. Contributions should be focused, reproducible, and honest about the project’s experimental status.

## Before opening a pull request

Please open an issue first for substantial behavior changes. For bug fixes and documentation improvements, explain the problem and intended result in the pull request. Do not include real camera footage, credentials, phone numbers, email passwords, private webhook URLs, generated model files, runtime databases, logs, recordings, or reports.

## Local setup

```bash
git clone https://github.com/vincenzo-afk/crowd-detection.git
cd crowd-detection
./setup.sh
```

Use simulation mode when a camera is unavailable:

```bash
./run.sh --sim --no-window
```

## Validation

The repository currently has no automated test suite. Run the same static checks used by continuous integration:

```bash
python3 -m compileall -q config src web main.py launch.py
bash -n setup.sh install-quick.sh run.sh start.sh
```

If your change affects runtime behavior, also describe a manual validation path, such as a simulation run or a local dashboard request.

## Branches and commits

Use a short branch name that describes the change, such as `fix/status-endpoint`, `feat/simulation-control`, or `docs/setup-guide`. Write imperative commit subjects, for example `Improve dashboard status response`. Keep unrelated refactors out of feature or bug-fix branches.

## Pull requests

A pull request should include the change summary, validation commands and results, configuration changes, security or privacy implications, and any breaking behavior. Update `README.md` when commands, routes, configuration keys, or user-visible behavior change.

The repository owner reviews changes before merging. Pull requests should not assume that a production deployment, authentication layer, or safety certification exists unless the implementation and documentation add it explicitly.
