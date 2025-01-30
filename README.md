# Docproc

A Python-based document region analyzer and content extraction tool.

> [!WARNING]  
> Project is under active development so most of the features aren't implemented, The readme is written to understand project scope.

## Overview

Docproc is an opinionated document region analyzer that helps extract text, equations, images and handwriting from documents. It provides both a library interface and a command-line tool.

## Installation

```bash
poetry install docproc
```

## Usage

### As a Command-line Tool

```bash
docproc analyze <file>
docproc extract --class <filename>
```

### As a Library

```python
from docproc import DocumentAnalyzer

analyzer = DocumentAnalyzer()
regions = analyzer.parse_file("example.md")
```

## Development

```bash
poetry install
poetry run test
```

## Contributing

Pull requests are welcome. Please ensure tests pass before submitting.

## Contact

For any questions, feedback or suggestions, please contact the author @ [hi@rithul.dev](mailto:hi@rithul.dev)
