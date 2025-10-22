#!/bin/bash

# Commit modified files
for file in $(git diff --name-only); do
  git add "$file"
  git commit -m "Update $file"
done

# Commit new files
for file in $(git ls-files --others --exclude-standard); do
  git add "$file"
  git commit -m "Add $file"
done

# Commit deleted files
for file in $(git diff --name-only --diff-filter=D); do
  git rm "$file"
  git commit -m "Remove $file"
done