#!/bin/bash
set -e

echo "Building main.tex..."
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "Build complete. Opening main.pdf..."
open main.pdf
