#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/src"
OUT="$ROOT/builds"
TMP="$ROOT/.build_tmp"

QUICK=false
if [ "$1" = "--quick" ]; then
    QUICK=true
fi

mkdir -p "$OUT"
rm -rf "$TMP"
mkdir -p "$TMP"

cp "$SRC"/*.tex "$TMP"/
cp "$SRC"/*.bib "$TMP"/
cp -r "$SRC"/diagrams "$TMP"/

cd "$TMP"

if [ "$QUICK" = true ]; then
    pdflatex -interaction=nonstopmode main.tex
else
    pdflatex -interaction=nonstopmode main.tex
    bibtex main
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex
fi

cp main.pdf "$OUT"/main.pdf

cd "$ROOT"
rm -rf "$TMP"

echo "Done: builds/main.pdf"
open "$OUT/main.pdf"
