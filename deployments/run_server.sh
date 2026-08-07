#!/usr/bin/env bash
set -euo pipefail

# Simple helper to run the Uvicorn server using the project venv.
# Usage: ./run_server.sh start|stop|restart

ROOT_DIR="/services/model-with-adpter-logic-semantic"
VENV="$ROOT_DIR/venv"
PIDFILE="$ROOT_DIR/server.pid"
LOGFILE="$ROOT_DIR/server.log"

start() {
  echo "Starting SWA REST server..."
  cd "$ROOT_DIR"
  export PYTHONPATH="$ROOT_DIR"
  source "$VENV/bin/activate"
  nohup python -m uvicorn swa.rest_server:app --host 0.0.0.0 --port 20000 > "$LOGFILE" 2>&1 &
  echo $! > "$PIDFILE"
  echo "Started: PID=$(cat $PIDFILE)"
}

stop() {
  if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    echo "Stopping PID $PID"
    kill "$PID" || true
    rm -f "$PIDFILE"
  else
    echo "No PID file found"
  fi
}

restart() {
  stop
  sleep 1
  start
}

case ${1:-start} in
  start) start ;;
  stop) stop ;;
  restart) restart ;;
  *) echo "Usage: $0 {start|stop|restart}"; exit 2 ;;
esac
