# Vexa Programming Language — Installation Guide

Vexa is a programming language with support for machine learning.

This guide explains how to install Vexa on macOS, set up the VS Code extension, and run your first Vexa program.

## 1. Requirements

Before installing Vexa, make sure you have:

- macOS
- Python 3
- Visual Studio Code (version 1.85 or newer)
- The Vexa interpreter (`interpreter.py`)
- The Vexa VS Code extension (`.vsix` file)

### Install Python

Download Python from:

https://www.python.org/downloads/

Verify the installation:

```bash
python3 --version
```

## 2. Install Machine Learning Dependencies

Vexa uses Python libraries for machine learning and data processing.

Install the required packages:

```bash
python3 -m pip install numpy pandas torch
```

These packages provide:

| Package | Purpose |
|---------|---------|
| NumPy | Numerical computing and arrays |
| pandas | Data processing and analysis |
| PyTorch (`torch`) | Machine learning and neural networks |

Verify the installation:

```bash
python3 -c "import numpy, pandas, torch; print('All ML dependencies installed!')"
```

If you receive an `externally-managed-environment` error, create a Python virtual environment instead.

## 3. Install the Vexa Interpreter

### Step 1: Download Vexa

Download `interpreter.py` from the Vexa GitHub repository.

### Step 2: Create the Vexa folder

Open Terminal and run:

```bash
mkdir -p ~/Desktop/Vexa
```

Move `interpreter.py` into the `Vexa` folder.

### Step 3: Test the interpreter

Open Terminal in the Vexa folder:

```bash
cd ~/Desktop/Vexa
```

Create a file named `test.vexa` containing a valid Vexa program.

Run:

```bash
python3 interpreter.py test.vexa
```

If the program executes successfully, the interpreter is working.

## 4. Enable the `vexa` Terminal Command

This allows you to run Vexa programs using:

```bash
vexa test.vexa
```

Instead of:

```bash
python3 interpreter.py test.vexa
```

### Step 1: Create the executable directory

```bash
mkdir -p ~/.local/bin
```

### Step 2: Create the Vexa executable

Run this command from the folder containing `interpreter.py`:

```bash
{
  printf '#!/usr/bin/env python3\n'
  cat interpreter.py
} > ~/.local/bin/vexa
```

### Step 3: Make it executable

```bash
chmod +x ~/.local/bin/vexa
```

### Step 4: Add Vexa to PATH

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Step 5: Test Vexa

```bash
vexa --help
```

Then run your program:

```bash
vexa test.vexa
```

You can now run Vexa programs from any directory.

**Important:** The `vexa` executable uses the Python interpreter found by `/usr/bin/env python3`. Make sure NumPy, pandas, and PyTorch are installed in that same Python environment.

## 5. Install the VS Code Extension

The Vexa extension provides editor support for `.vexa` files.

### Step 1: Download the extension

Download the Vexa `.vsix` file from the GitHub repository.

Example filename:

`vexa-language-0.1.0.vsix`

### Step 2: Open VS Code

Make sure VS Code is installed in your Applications folder.

### Step 3: Install the extension

1. Open VS Code.
2. Press `Cmd + Shift + P`.
3. Search for `Extensions: Install from VSIX...`.
4. Select the downloaded `.vsix` file.
5. Reload VS Code if prompted.

### Step 4: Verify installation

Create or open a file ending in `.vexa`.

VS Code should recognize it as a Vexa file.

## 6. Run Your First Vexa Program

Create a file named:

`test.vexa`

Write a valid Vexa program in the file.

Open the VS Code terminal and run:

```bash
vexa test.vexa
```

Vexa will execute your program.

## 7. Troubleshooting

### Error: `command not found: vexa`

Make sure the executable exists:

```bash
ls -l ~/.local/bin/vexa
```

Reload your terminal configuration:

```bash
source ~/.zshrc
```

Try again:

```bash
vexa --help
```

### Error: VSIX is not compatible

Your VS Code version may be outdated.

1. Open VS Code.
2. Select `Code > Check for Updates`.
3. Install the update.
4. Restart VS Code.
5. Try installing the `.vsix` file again.

### Error: Cannot update on a read-only volume

This usually means VS Code is running from a downloaded disk image.

1. Quit VS Code.
2. Open the downloaded `.dmg` file.
3. Drag Visual Studio Code into Applications.
4. Eject the disk image.
5. Launch VS Code from Applications.
6. Try updating again.

### Error: `No module named numpy`

Install NumPy:

```bash
python3 -m pip install numpy
```

### Error: `No module named pandas`

Install pandas:

```bash
python3 -m pip install pandas
```

### Error: `No module named torch`

Install PyTorch:

```bash
python3 -m pip install torch
```

### Error: `externally-managed-environment`

Your Python installation may require a virtual environment.

Create one:

```bash
python3 -m venv ~/.venvs/vexa
```

Activate it:

```bash
source ~/.venvs/vexa/bin/activate
```

Install dependencies:

```bash
python -m pip install numpy pandas torch
```

To run Vexa with this environment, change the first line of `~/.local/bin/vexa` to the absolute path of the virtual environment's Python, preceded by `#!`.

For example, if your home directory is `/Users/example`, use:

```python
#!/Users/example/.venvs/vexa/bin/python
```

This ensures Vexa uses the environment containing its ML dependencies.

## 8. Verify Everything

Check Python:

```bash
python3 --version
```

Check Vexa:

```bash
vexa --help
```

Check the machine learning libraries:

```bash
python3 -c "import numpy, pandas, torch; print('ML ready!')"
```

Check the VS Code extension by opening a `.vexa` file.

If all checks pass, Vexa is ready to use!

---

## About Vexa

Vexa is a programming language with its own Python-based interpreter, VS Code editor support, and machine learning capabilities.

Happy coding!
