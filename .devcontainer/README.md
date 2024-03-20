# GitHub Codespace with OpenROAD GUI

The goal is to provide a web browser experience of the OpenROAD GUI. Solution works with Firefox and Chrome.

## User

1. First time users are encourage to read the tutorial [creating a codespace for a repository](https://docs.github.com/en/codespaces/developing-in-a-codespace/creating-a-codespace-for-a-repository). Follow this tutorial to create a codespace from the ORFS repository. Select `ubuntu2204-gui` in the `Dev container configuration` field.
2. Currently to run the OpenROAD GUI in the web browser, you need to [install VSCode](https://code.visualstudio.com/docs/setup/setup-overview).
3. Open VSCode and install the [GitHub Codespaces](https://marketplace.visualstudio.com/items?itemName=GitHub.codespaces) extension
4. Connect to the generated codespace
5. Open a web browser and connect to url: `localhost:6080`
6. Alternatively, you can use a VNC client to connect to `localhost:5901`

## Developer

OpenROAD Flow Scripts provide a docker image generation script `etc/DockerHelper.sh`, which is used here to build and publish the image to GH registry.

Useful resources:
* [Development loop](https://code.visualstudio.com/docs/devcontainers/create-dev-container#_full-configuration-edit-loop)
* [devcontainer JSON reference](https://containers.dev/implementors/json_reference/)
* [Desktop lite feature](https://github.com/microsoft/vscode-dev-containers/blob/main/script-library/docs/desktop-lite.md)
