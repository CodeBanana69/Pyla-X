# PylaAI

[![CC BY-NC 4.0 License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/xUusk3fw4A)
[![Trello](https://img.shields.io/badge/Trello-0079BF?logo=trello&logoColor=white)](https://trello.com/b/SAz9J6AA/public-pyla-trello)

> [!WARNING]
> **Warning**: There are two versions of PylaAI, you are currently browsing the source code for developers. Please visit our [Discord](https://discord.gg/xUusk3fw4A) to use the compiled version, which comes as a ready-to-use `.exe`.

PylaAI is currently the best external Brawl Stars bot.

## Requirements

- **NVIDIA GPUs**
  - `setup.py` installs `onnxruntime-gpu` when an NVIDIA GPU is detected
  - Falls back to DirectML, then CPU, if CUDA does not load

- **AMD / Intel / iGPU**
  - DirectML on Windows (`onnxruntime-directml`)
  - CPU fallback if DirectML does not load

## Installation

You will need [Python 3.11.9](https://www.python.org/downloads/release/python-3119/).

### Windows

```sh
python setup.py install
```

### Other Platforms

The official PylaAI does **NOT** support other platforms such as Linux or Mac, but you can visit [Unofficial Ports](https://github.com/4D1-TooFarGone/Pyla-Ports) for cross-platform support.

## Using PylaAI

> [!NOTE]
> **Note**: This open-source version runs in localhost mode. The cloud features have been disabled by default.

Run the bot:

```sh
python main.py
```

### Startup options

PylaAI can run as its own desktop window (pywebview), in the system browser, or headless. A command-line flag overrides `interface_mode` in `cfg/general_config.toml` (`desktop`, `browser`, or `headless`).

| Flag | Effect |
| --- | --- |
| *(none)* | Console visible. Opens the native desktop window when pywebview is available, otherwise the system browser. |
| `--desktop` | Force the integrated pywebview desktop window. |
| `--web`, `--browser`, `--no-webapp` | Open the UI in the system browser instead of the desktop window. |
| `--headless` | Serve the local web UI without opening a window or browser. Open the printed localhost URL yourself. |
| `--no-console` | Hides the console window, output goes to `pyla.log` in the current folder. Ignored when Pyla is started from an existing terminal, so your own terminal is never hidden. |

Examples:

```sh
python main.py --desktop
python main.py --headless --no-console
python main.py --no-webapp
```


## License

This project is **not permitted to be sold or monetized** under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Maintainer

### Developers

- **ivanyordanovgt**
- **AngelFireLA**
- **awarzu**

### Contributors

- **Maayan080**
- **simonrejzek**
- **bocchi-the-cat**
- **Ariko842**
- **Nauwk07**
- **aetherwtff**
- **fazelukario**
- **k00shi**
- **mydd7**
