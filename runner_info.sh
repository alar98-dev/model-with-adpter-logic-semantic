#!/bin/bash
echo "=== SYSTEM INFO ==="
uname -a
uptime
id
echo "=== POSTGRESQL CHECK ==="
pg_ctl status || true
ps aux | grep -i postgres || true
echo "=== PATHS CHECK ==="
ls -la /sdcard/SRV/DEVELOPMENT 2>/dev/null || echo "No /sdcard/SRV/DEVELOPMENT"
ls -la ~
