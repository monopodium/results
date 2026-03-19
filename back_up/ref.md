作者：张宇杭
链接：https://zhuanlan.zhihu.com/p/1949862513010272038
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

编译 NCCL 源代码首先，克隆 nccl 源代码，并进入到 nccl 目录中：$ git clone https://github.com/NVIDIA/nccl.git
$ cd nccl
在该仓库的 README.md 文件中，能够很快找到编译 NCCL 需要运行的命令：$ make -j src.build
-j 选项指定 make 并行地执行编译过程。这里并没有指定 job 的数量，也就是不对最大并行度做限制。src.build 则是需要构建的目标。从 Makefile 中可以发现构建 src.build 的这一条 recipe:...

src.%:
	${MAKE} -C src $* BUILDDIR=${ABSBUILDDIR}

...
构建 src.build 时，实际上执行的 make -C src build BUILDDIR=${ABSBUILDDIR}那么接下来查看 src 文件夹下面的 Makefile，该文件还 include 了文件 ../makefiles/common.mk 以及 ../makefiles/version.mk。可以发现 common.mk 里面定义了调试时开启的编译选项：ifeq ($(DEBUG), 0)
NVCUFLAGS += -O3
CXXFLAGS  += -O3 -g
else
NVCUFLAGS += -O0 -G -g
CXXFLAGS  += -O0 -g -ggdb3
endif
也就是说，为了调试 nccl，我们在编译之前需要设置环境变量 DEBUG=1。common.mk 中还有一个变量 TRACE，这个变量会控制编译 NCCL 时是否定义宏 ENABLE_TRACE，从而改变NCCL的执行过程，让它能输出更多调试信息，本文选择开启它。此外，为了加快编译过程，可以指定变量 NVCC_GENCODE，只为某一特定架构下生成二进制。下面是以 A100 为例的编译命令。$ export DEBUG=1
$ export TRACE=1
$ make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
Optional： 使用 bear 捕捉编译过程，生成一份 compile_commands.json。我们可以为 VSCode 的 cpp 扩展配置 compile commands，以获得更好的代码跳转等功能，从而帮助我们阅读 NCCL 的源代码。推荐此时不能把并行度开得太大，否则 bear 会占用过多的操作系统资源导致故障。可以使用如下的样例：$ export DEBUG=1
$ export TRACE=1
$ bear -- make -j8 src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
使用 CMake除了使用 make 来构建 NCCL，我们还可以使用 cmake 来构建 NCCL。构建的命令如下：$ mkdir build
$ cd build
$ export DEBUG=1
$ cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CUDA_ARCHITECTURES="80" -DTRACE=ON ..
$ make -j
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON 指示 cmake 生成一份 compile_commands.json。编译结束后，build 文件夹下面会有 include/ 目录包含 NCCL 头文件，lib/目录包含 NCCL 库文件。此外，还有一个 compile_commands.json。编译 nccl-testsnccl-tests 是 NVIDIA 官方提供的工具，用于检查 NCCL 的性能和正确性。我们可以将其作为我们调试与运行 NCCL 的入口。在运行调试之前，我们需要知道NCCL 支持多种并发模型:单线程多GPU多线程，每一个线程控制一个GPU多进程从这一信息中可以推断，NCCL 尽管和 MPI 有着相似的集合通信 API，但是它们实际上不是一个层面的东西，NCCL 可以使用 MPI 来支撑自己的多进程并发模型，二者之间不是一个互相代替的关系。本文将以 NCCL+MPI 为例，编译 nccl-tests 并运行调试。编译的命令如下：$ export DEBUG=1
$ export MPI_HOME=YOUR_MPI_HOME
$ export NCCL_HOME=YOUR_NCCL_HOME
$ make -j MPI=1 NAME_SUFFIX=_mpi NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
NAME_SUFFIX 变量的作用是让最后编译生成的可执行文件后缀带上 _mpi，比如 all_gather_perf_mpi调试示例程序由于使用 MPI 运行 NCCL 的时候有多个进程，此时推荐调试 NCCL 的方式是调试 rank 0 所在的进程：运行程序时，用 gdbserver 包裹 rank0，其它的进程正常运行。此时 rank0 会等待一个 gdb 进程的连接调试的时候连接特定的 host 和 port 即可。使用下面这个脚本，假设其名为 mpi_gdbserver.sh。#!/usr/bin/env bash
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
      echo "usage: scripts/mpi_gdbserver.sh [mpirun args...] [--host H] [--port P] [--dry-run] -- <program> [args...]" >&2
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
  env MPI_GDBSERVER="$gdbserver" MPI_GDB_HOST="$host" MPI_GDB_PORT="$port"
  sh -c "$rank_wrapper" -- "$prog" "${prog_args[@]}"
)

echo "[mpi_gdbserver] rank0 gdbserver listening on ${host}:${port}" >&2
echo "[mpi_gdbserver] attach: gdb ${prog} -ex 'target remote ${host}:${port}'" >&2
printf '[mpi_gdbserver] launch:' >&2; printf ' %q' "${cmd[@]}" >&2; echo >&2

(( dry_run == 1 )) && exit 0
exec "${cmd[@]}"
运行程序：export NCCL_TESTS_DIR=YOUR_NCCL_TESTS_DIR
#修改 LD_LIBRARY_PATH，确保动态链接库能找到
export NCCL_HOME=YOUR_NCCL_HOME
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
mpi_gdbserver.sh -n 2 -- $NCCL_TESTS_DIR/build/all_gather_perf_mpi -b 1K -e 1G -f 2
然后使用 gdb 来连接这个进程：$ gdb YOUR_NCCL_TESTS_DIR/build/all_gather_perf_mpi -ex 'target remote 127.0.0.1:2345'
然后就可以正常调试了。在 VSCode/Neovim 中调试使用 VSCode 进行调试时，需要做的事情很简单，只需要在 NCCL 的目录下面编写一个配置文件 .vscode/launch.json：{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "(gdbserver) NCCL rank0",
      "type": "cppdbg",
      "request": "launch",
      "program": "YOUR_NCCL_TESTS_DIR/build/all_gather_perf_mpi",
      "MIMode": "gdb",
      "cwd": "${workspaceFolder}",
      "miDebuggerServerAddress": "127.0.0.1:2345",
      "miDebuggerPath": "/usr/bin/gdb",
      "setupCommands": [
        {
          "description": "Enable pretty-printing for gdb",
          "text": "-enable-pretty-printing",
          "ignoreFailures": true
        }
      ]
    }
  ]
}
这个配置文件指定了 gdb 启动的时候连接 127.0.0.1:2345，并且调试文件 all_gather_perf_mpi。记得修改 YOUR_NCCL_TESTS_DIR 为你的机器上的 nccl-tests 的目录路径。运行调试后的效果如下