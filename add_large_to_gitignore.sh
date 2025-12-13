#!/bin/bash

# Find files > 25MB and add to .gitignore
find . -type f -size +25M | sed 's|^\./||' | while read file; do
    if ! grep -Fxq "$file" .gitignore; then
        echo "Adding to .gitignore: $file"
        echo "$file" >> .gitignore
    fi
done

echo "Done! Added large files to .gitignore"