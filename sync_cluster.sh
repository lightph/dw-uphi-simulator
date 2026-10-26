#!/bin/bash

# Configuration
LOCAL_DIR="$HOME/Documents/Investigación 1/proyecto-v3/"
REMOTE_USER="luz.perez"
REMOTE_HOST="10.73.25.233"
REMOTE_DIR="/home/luz.perez/inv1/proyecto-v3/"

echo "Syncing $LOCAL_DIR to cluster..."

# Execute rsync
# -a : Archive mode (preserves permissions, times, symlinks)
# -v : Verbose
# -z : Compress data during transfer
# -P : Show progress and keep partially transferred files if interrupted
# --filter=":- .gitignore" : Tells rsync to parse and respect .gitignore rules
# --exclude=".git/" : Explicitly ignores the hidden git directory itself
rsync -avzP \
  --filter=":- .gitignore" \
  --exclude=".git/" \
  "$LOCAL_DIR" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

echo "Sync complete!"
