#!/usr/bin/env bash
set -euo pipefail

mpirun="${MPIRUN:-mpirun}"
gdbserver="${GDBSERVER:-gdbserver}"
host="127.0.0.1"
port="2345"
dry_run=0
mpirun_args=()

while (($#)); do
  case "$1" in
    -h|--help)
      echo "usage: mpi_gdbserver.sh [mpirun args...] [--host H] [--port P] [--dry-run] -- <program> [args...]" >&2
      exit 0
      ;;
    --host) host="${2:?missing value for --host}"; shift 2 ;;
    --port) port="${2:?missing value for --port}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --) shift; break ;;
    *) mpirun_args+=("$1"); shift ;;
  esac
done

prog="${1:?missing program (use -- <program> [args...])}"
shift || true
prog_args=("$@")

if [[ ! -x "$prog" && -x "build/$prog" ]]; then
  prog="build/$prog"
fi
[[ -x "$prog" ]] || { echo "error: program not found or not executable: $prog" >&2; exit 2; }

rank_wrapper='r=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-${SLURM_PROCID:-${MV2_COMM_WORLD_RANK:-0}}}}}; p=$1; shift; if [ "$r" = 0 ]; then exec "$MPI_GDBSERVER" "$MPI_GDB_HOST:$MPI_GDB_PORT" "$p" "$@"; else exec "$p" "$@"; fi'

cmd=(
  "$mpirun" "${mpirun_args[@]}"
  -x LD_LIBRARY_PATH
  env MPI_GDBSERVER="$gdbserver" MPI_GDB_HOST="$host" MPI_GDB_PORT="$port"
  sh -c "$rank_wrapper" -- "$prog" "${prog_args[@]}"
)

echo "[mpi_gdbserver] rank0 gdbserver listening on ${host}:${port}" >&2
echo "[mpi_gdbserver] attach: gdb ${prog} -ex 'target remote ${host}:${port}'" >&2
printf '[mpi_gdbserver] launch:' >&2; printf ' %q' "${cmd[@]}" >&2; echo >&2

(( dry_run == 1 )) && exit 0
exec "${cmd[@]}"
