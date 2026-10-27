#!/bin/bash
echo "=== POSTGRES DATABASES ==="
sudo -u postgres psql -c "\l"
echo "=== POSTGRES USERS ==="
sudo -u postgres psql -c "\du"
