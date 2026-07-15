#!/usr/bin/env bash
#
# sysbench-cpu.sh — 在 StarryOS/QEMU 里复现 sysbench CPU 多线程并行性测试。
#
# 用法示例：
#   ./scripts/sysbench-cpu.sh --arch x86_64 --smp 4
#   ./scripts/sysbench-cpu.sh --arch x86_64 --smp 4,8
#   ./scripts/sysbench-cpu.sh --arch x86_64 --smp 8 --threads 1,2,4,8
#   ./scripts/sysbench-cpu.sh --arch x86_64 --smp 4 --threads 1,2,4 --runs 3
#   ./scripts/sysbench-cpu.sh --arch x86_64 --smp 4 --fresh-rootfs
#
# 说明：
#   - 默认从已下载的 Alpine rootfs 复制一份临时 rootfs 到 tmp/sysbench/。
#   - sysbench 只安装到这份临时 rootfs，不污染正式 rootfs。
#   - QEMU 配置也生成在 tmp/sysbench/，日志默认写入 .agents/log/sysbench/。
#   - 输出中的 SYSBENCH_CPU_RESULT 行可直接用于计算多线程加速比。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ARCH="x86_64"
SMP_LIST_RAW="4"
THREADS="1,2,4"
BENCH_TIME=5
CPU_MAX_PRIME=20000
WORK_DIR="tmp/sysbench"
LOG_DIR=".agents/log/sysbench"
ROOTFS_IMG=""
FRESH_ROOTFS="false"
REINSTALL_SYSBENCH="false"
RUNS=1
WARMUP_RUNS=1

info() {
    printf "[sysbench-cpu] %s\n" "$*"
}

error() {
    printf "[sysbench-cpu] ERROR: %s\n" "$*" >&2
    exit 1
}

usage() {
    cat <<'USAGE'
Usage:
  ./scripts/sysbench-cpu.sh [--workspace DIR] [--arch x86_64] [--smp N|N,N] [--threads 1,2,4] \
    [--time SEC] [--cpu-max-prime N] [--warmup N] [--runs N] [--fresh-rootfs] [--reinstall-sysbench]

Options:
  --workspace <DIR>      构建并测试指定源码树；默认是脚本所在仓库，可用于干净 baseline worktree
  --arch <arch>          目前支持 x86_64（默认：x86_64）
  --smp <N|LIST>         StarryOS/QEMU vCPU 数；可用 4,8 生成对比表（默认：4）
  --threads <LIST>       sysbench 线程列表，逗号或空格分隔（默认：1,2,4）
  --time <SEC>           每组 sysbench 运行秒数（默认：5）
  --cpu-max-prime <N>    sysbench cpu --cpu-max-prime 参数（默认：20000）
  --warmup <N>           不计入统计的完整矩阵预热次数（默认：1；0 表示跳过）
  --runs <N>             每个 SMP/线程组合重复次数，汇总输出平均值和范围（默认：1）
  --rootfs <IMAGE>       使用指定临时 rootfs；默认 tmp/sysbench/rootfs-x86_64-sysbench.img
  --fresh-rootfs         重新从 Alpine rootfs 复制临时 rootfs，并重新安装 sysbench
  --reinstall-sysbench   即使 rootfs 里已有 sysbench，也重新 apk add
  --log-dir <DIR>        日志目录（默认：.agents/log/sysbench）
  --help, -h             显示帮助
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace)
            WORKSPACE_ROOT="${2:-}"
            shift 2
            ;;
        --arch)
            ARCH="${2:-}"
            shift 2
            ;;
        --smp)
            SMP_LIST_RAW="${2:-}"
            shift 2
            ;;
        --threads)
            THREADS="${2:-}"
            shift 2
            ;;
        --time)
            BENCH_TIME="${2:-}"
            shift 2
            ;;
        --cpu-max-prime)
            CPU_MAX_PRIME="${2:-}"
            shift 2
            ;;
        --runs)
            RUNS="${2:-}"
            shift 2
            ;;
        --warmup)
            WARMUP_RUNS="${2:-}"
            shift 2
            ;;
        --rootfs)
            ROOTFS_IMG="${2:-}"
            shift 2
            ;;
        --fresh-rootfs)
            FRESH_ROOTFS="true"
            shift
            ;;
        --reinstall-sysbench)
            REINSTALL_SYSBENCH="true"
            shift
            ;;
        --log-dir)
            LOG_DIR="${2:-}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            error "未知参数：$1"
            ;;
    esac
done

[[ -f "$WORKSPACE_ROOT/Cargo.toml" ]] || error "源码树不存在或缺少 Cargo.toml：$WORKSPACE_ROOT"
WORKSPACE_ROOT="$(cd "$WORKSPACE_ROOT" && pwd)"
cd "$WORKSPACE_ROOT"

[[ "$ARCH" == "x86_64" ]] || error "当前 sysbench CPU 复现脚本只支持 --arch x86_64"
[[ "$BENCH_TIME" =~ ^[0-9]+$ ]] && [[ "$BENCH_TIME" -gt 0 ]] || error "--time 必须是正整数"
[[ "$CPU_MAX_PRIME" =~ ^[0-9]+$ ]] && [[ "$CPU_MAX_PRIME" -gt 0 ]] || error "--cpu-max-prime 必须是正整数"
[[ "$RUNS" =~ ^[0-9]+$ ]] && [[ "$RUNS" -gt 0 ]] || error "--runs 必须是正整数"
[[ "$WARMUP_RUNS" =~ ^[0-9]+$ ]] || error "--warmup 必须是非负整数"
command -v debugfs >/dev/null 2>&1 || error "找不到 debugfs，请安装 e2fsprogs"

THREAD_LIST="${THREADS//,/ }"
declare -A SEEN_THREADS=()
for thread in $THREAD_LIST; do
    [[ "$thread" =~ ^[0-9]+$ ]] && [[ "$thread" -gt 0 ]] || error "--threads 中包含非法线程数：$thread"
    [[ -z "${SEEN_THREADS[$thread]+set}" ]] || error "--threads 中不能重复线程数：$thread"
    SEEN_THREADS[$thread]=1
done

SMP_LIST="${SMP_LIST_RAW//,/ }"
declare -A SEEN_SMPS=()
for smp in $SMP_LIST; do
    [[ "$smp" =~ ^[0-9]+$ ]] && [[ "$smp" -gt 0 ]] || error "--smp 中包含非法 CPU 数：$smp"
    [[ -z "${SEEN_SMPS[$smp]+set}" ]] || error "--smp 中不能重复 CPU 数：$smp"
    SEEN_SMPS[$smp]=1
done
FIRST_SMP="$(printf '%s\n' $SMP_LIST | sed -n '1p')"
THREAD_VALUES=($THREAD_LIST)
SMP_VALUES=($SMP_LIST)
[[ "${#THREAD_VALUES[@]}" -gt 0 ]] || error "--threads 不能为空"
[[ "${#SMP_VALUES[@]}" -gt 0 ]] || error "--smp 不能为空"

mkdir -p "$WORK_DIR" "$LOG_DIR"

if [[ -z "$ROOTFS_IMG" ]]; then
    ROOTFS_IMG="$WORK_DIR/rootfs-${ARCH}-sysbench.img"
fi

if [[ "$FRESH_ROOTFS" == "true" || ! -f "$ROOTFS_IMG" ]]; then
    BASE_ROOTFS="tmp/axbuild/rootfs/rootfs-${ARCH}-alpine.img"
    if [[ -d "$BASE_ROOTFS" ]]; then
        BASE_ROOTFS="$BASE_ROOTFS/rootfs-${ARCH}-alpine.img"
    fi

    if [[ ! -f "$BASE_ROOTFS" ]]; then
        info "未找到 Alpine rootfs，先下载/准备：cargo xtask starry rootfs --arch $ARCH"
        cargo xtask starry rootfs --arch "$ARCH"
    fi
    [[ -f "$BASE_ROOTFS" ]] || error "Alpine rootfs 不存在：$BASE_ROOTFS"

    info "复制临时 rootfs：$BASE_ROOTFS -> $ROOTFS_IMG"
    cp -f "$BASE_ROOTFS" "$ROOTFS_IMG"
fi
[[ -f "$ROOTFS_IMG" ]] || error "sysbench 临时 rootfs 不存在：$ROOTFS_IMG"

rootfs_has_sysbench() {
    debugfs -R 'stat /usr/bin/sysbench' "$ROOTFS_IMG" 2>/dev/null | grep -q 'Type: regular'
}

write_install_qemu_config() {
    local config="$1"
    cat > "$config" <<EOF
args = [
  "-m",
  "2G",
  "-nographic",
  "-machine",
  "q35",
  "-device",
  "virtio-blk-pci,drive=disk0",
  "-drive",
  "id=disk0,if=none,format=raw,file=\${workspace}/$ROOTFS_IMG",
  "-device",
  "virtio-net-pci,netdev=net0",
  "-netdev",
  "user,id=net0",
]
uefi = true
to_bin = true
shell_prefix = "root@starry:"
shell_init_cmd = '''
set -eu
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

echo SYSBENCH_INSTALL_BEGIN
apk update
apk add --no-cache sysbench
sync
sysbench --version
echo SYSBENCH_INSTALL_DONE
'''
success_regex = ["(?m)^SYSBENCH_INSTALL_DONE\\\\s*$"]
fail_regex = [
  "(?i)\\\\bpanic(?:ked)?\\\\b",
  "(?i)page fault",
  "(?i)segmentation fault",
  "(?m)^SYSBENCH_INSTALL_FAILED\\\\s*$",
]
timeout = 600
EOF
}

write_run_qemu_config() {
    local config="$1"
    local thread_list="$2"
    cat > "$config" <<EOF
args = [
  "-m",
  "2G",
  "-nographic",
  "-machine",
  "q35",
  "-device",
  "virtio-blk-pci,drive=disk0",
  "-drive",
  "id=disk0,if=none,format=raw,file=\${workspace}/$ROOTFS_IMG",
  "-device",
  "virtio-net-pci,netdev=net0",
  "-netdev",
  "user,id=net0",
  "-snapshot",
]
uefi = true
to_bin = true
shell_prefix = "root@starry:"
shell_init_cmd = '''
set -eu
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

echo SYSBENCH_CPU_BEGIN
sysbench --version

for t in $thread_list; do
    echo SYSBENCH_CPU_THREADS=\$t
    out="/tmp/sysbench-cpu-\$t.log"
    sysbench cpu --threads="\$t" --cpu-max-prime=$CPU_MAX_PRIME --time=$BENCH_TIME run | tee "\$out"
    eps="\$(awk '/events per second:/ { print \$4 }' "\$out")"
    echo "SYSBENCH_CPU_RESULT threads=\$t eps=\$eps"
done

echo SYSBENCH_CPU_DONE
'''
success_regex = ["(?m)^SYSBENCH_CPU_DONE\\\\s*$"]
fail_regex = [
  "(?i)\\\\bpanic(?:ked)?\\\\b",
  "(?i)page fault",
  "(?i)segmentation fault",
  "(?m)^SYSBENCH_CPU_FAILED\\\\s*$",
]
timeout = 600
EOF
}

run_and_log() {
    local log="$1"
    shift

    info "日志：$log"
    set +e
    if command -v stdbuf >/dev/null 2>&1; then
        stdbuf -oL -eL "$@" 2>&1 | tee "$log"
    else
        "$@" 2>&1 | tee "$log"
    fi
    local status=${PIPESTATUS[0]}
    set -e
    return "$status"
}

summarize_cpu_log() {
    local log="$1"
    local base_thread="$2"
    awk -v base_thread="$base_thread" '
        /^SYSBENCH_CPU_RESULT[[:space:]]/ {
            split($2, t, "=");
            split($3, e, "=");
            gsub(/\r/, "", e[2]);
            order[++n] = t[2];
            eps[t[2]] = e[2];
        }
        END {
            if (n == 0) {
                exit 0;
            }
            base = eps[base_thread];
            for (i = 1; i <= n; i++) {
                thread = order[i];
                scale = eps[thread] / base;
                printf("threads=%s eps=%.2f scale=%.3fx\n", thread, eps[thread], scale);
            }
        }
    ' "$log"
}

rotate_values() {
    local offset="$1"
    shift
    local values=("$@")
    local count="${#values[@]}"
    local rotated=()

    for ((index = 0; index < count; index++)); do
        rotated+=("${values[(index + offset) % count]}")
    done
    printf '%s' "${rotated[*]}"
}

validate_cpu_log() {
    local log="$1"
    local expected_threads="$2"
    local expected_count=0

    for thread in $expected_threads; do
        local count
        count="$(awk -v expected_thread="$thread" '
            /^SYSBENCH_CPU_RESULT[[:space:]]/ {
                split($2, t, "=");
                split($3, e, "=");
                gsub(/\r/, "", e[2]);
                if (t[2] == expected_thread && e[2] ~ /^[0-9]+(\.[0-9]+)?$/) {
                    count++;
                }
            }
            END { print count + 0 }
        ' "$log")"
        [[ "$count" == "1" ]] || error "sysbench 结果不完整：threads=$thread 在 $log 中出现 $count 次"
        ((expected_count += 1))
    done

    local total_count
    total_count="$(awk '
        /^SYSBENCH_CPU_RESULT[[:space:]]/ { count++ }
        END { print count + 0 }
    ' "$log")"
    [[ "$total_count" == "$expected_count" ]] || error "sysbench 结果包含重复或未知线程：$log"
}

append_summary_tsv() {
    local run="$1"
    local smp="$2"
    local log="$3"
    local output="$4"

    awk -v run="$run" -v smp="$smp" '
        /^SYSBENCH_CPU_RESULT[[:space:]]/ {
            split($2, t, "=");
            split($3, e, "=");
            gsub(/\r/, "", e[2]);
            printf("%s\t%s\t%s\t%s\n", run, smp, t[2], e[2]);
        }
    ' "$log" >> "$output"
}

aggregate_summary_tsv() {
    local raw_summary_tsv="$1"
    local output="$2"

    awk -F '\t' -v smp_order="$SMP_LIST" -v thread_order="$THREAD_LIST" '
        NR == 1 { next }
        {
            key = $2 SUBSEP $3;
            sum[key] += $4;
            count[key]++;
            if (!(key in min) || $4 < min[key]) {
                min[key] = $4;
            }
            if (!(key in max) || $4 > max[key]) {
                max[key] = $4;
            }
        }
        END {
            smp_count = split(smp_order, smps, " ");
            thread_count = split(thread_order, threads, " ");
            print "smp\tthreads\truns\tavg_eps\tmin_eps\tmax_eps";
            for (s = 1; s <= smp_count; s++) {
                for (t = 1; t <= thread_count; t++) {
                    key = smps[s] SUBSEP threads[t];
                    if (count[key] != 0) {
                        printf("%s\t%s\t%d\t%.2f\t%.2f\t%.2f\n", smps[s], threads[t], count[key], sum[key] / count[key], min[key], max[key]);
                    }
                }
            }
        }
    ' "$raw_summary_tsv" > "$output"
}

print_matrix_summary() {
    local summary_tsv="$1"

    awk -v smp_order="$SMP_LIST" -v thread_order="$THREAD_LIST" '
        BEGIN {
            smp_count = split(smp_order, smps, " ");
            thread_count = split(thread_order, threads, " ");
            first_thread = threads[1];
            last_thread = threads[thread_count];
        }
        NR == 1 { next }
        {
            eps[$1, $2] = $4;
        }
        END {
            print "";
            print "CPU - sysbench cpu --cpu-max-prime=" prime " --time=" bench_time " (events/sec, higher = better)";

            printf("%-10s", "threads");
            for (i = 1; i <= smp_count; i++) {
                printf(" %-18s", "StarryOS smp" smps[i]);
            }
            print "";

            for (t = 1; t <= thread_count; t++) {
                thread = threads[t];
                printf("%-10s", thread);
                for (i = 1; i <= smp_count; i++) {
                    smp = smps[i];
                    value = eps[smp, thread];
                    if (value == "") {
                        printf(" %-18s", "-");
                    } else {
                        printf(" %-18.2f", value);
                    }
                }
                print "";
            }

            printf("\nScaling (%s->%s threads):", first_thread, last_thread);
            for (i = 1; i <= smp_count; i++) {
                smp = smps[i];
                base = eps[smp, first_thread];
                last = eps[smp, last_thread];
                if (base == "" || last == "" || base == 0) {
                    printf(" StarryOS smp%s -", smp);
                } else {
                    printf(" StarryOS smp%s %.2fx", smp, last / base);
                }
                if (i < smp_count) {
                    printf(";");
                }
            }
            print "";

            if (smp_count >= 2) {
                left_smp = smps[1];
                right_smp = smps[smp_count];
                left = eps[left_smp, last_thread];
                right = eps[right_smp, last_thread];
                if (left != "" && right != "" && left != 0) {
                    printf("StarryOS smp%s vs smp%s at %s threads: %.2fx\n", left_smp, right_smp, last_thread, right / left);
                }
            }
        }
    ' prime="$CPU_MAX_PRIME" bench_time="$BENCH_TIME" "$summary_tsv"
}

run_cpu_matrix() {
    local phase="$1"
    local ordinal="$2"
    local record_results="$3"
    local rotation_offset="$4"
    local smp_order
    local thread_order

    smp_order="$(rotate_values "$rotation_offset" "${SMP_VALUES[@]}")"
    thread_order="$(rotate_values "$rotation_offset" "${THREAD_VALUES[@]}")"
    write_run_qemu_config "$RUN_CONFIG" "$thread_order"

    for smp in $smp_order; do
        local run_log
        if [[ "$record_results" == "true" ]]; then
            run_log="$LOG_DIR/sysbench-cpu-smp${smp}-run${ordinal}-${TIMESTAMP}.log"
        else
            run_log="$LOG_DIR/sysbench-cpu-warmup-smp${smp}-run${ordinal}-${TIMESTAMP}.log"
        fi

        info "运行 sysbench CPU 测试：phase=$phase run=$ordinal smp=$smp threads=[$thread_order] time=${BENCH_TIME}s prime=$CPU_MAX_PRIME"
        run_and_log "$run_log" cargo xtask starry qemu \
            --arch "$ARCH" \
            --smp "$smp" \
            --qemu-config "$RUN_CONFIG" \
            --rootfs "$ROOTFS_IMG"

        validate_cpu_log "$run_log" "$thread_order"
        info "本轮结果摘要："
        summarize_cpu_log "$run_log" "${THREAD_VALUES[0]}"
        if [[ "$record_results" == "true" ]]; then
            append_summary_tsv "$ordinal" "$smp" "$run_log" "$RAW_SUMMARY_TSV"
        fi
        info "完整日志：$run_log"
    done
}

INSTALL_CONFIG="$WORK_DIR/qemu-install-sysbench-${ARCH}.toml"
RUN_CONFIG="$WORK_DIR/qemu-run-sysbench-cpu-${ARCH}.toml"
write_install_qemu_config "$INSTALL_CONFIG"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
RAW_SUMMARY_TSV="$LOG_DIR/sysbench-runs-${TIMESTAMP}.tsv"
SUMMARY_TSV="$LOG_DIR/sysbench-summary-${TIMESTAMP}.tsv"
printf 'run\tsmp\tthreads\teps\n' > "$RAW_SUMMARY_TSV"

if [[ "$REINSTALL_SYSBENCH" == "true" ]] || ! rootfs_has_sysbench; then
    INSTALL_LOG="$LOG_DIR/sysbench-install-smp${FIRST_SMP}-${TIMESTAMP}.log"
    info "临时 rootfs 中没有 sysbench，启动 StarryOS 安装 sysbench..."
    run_and_log "$INSTALL_LOG" cargo xtask starry qemu \
        --arch "$ARCH" \
        --smp "$FIRST_SMP" \
        --qemu-config "$INSTALL_CONFIG" \
        --rootfs "$ROOTFS_IMG"
else
    info "临时 rootfs 已包含 /usr/bin/sysbench，跳过安装阶段。"
fi

for warmup in $(seq 1 "$WARMUP_RUNS"); do
    run_cpu_matrix "warmup" "$warmup" "false" "$((warmup - 1))"
done

for run in $(seq 1 "$RUNS"); do
    run_cpu_matrix "measure" "$run" "true" "$((WARMUP_RUNS + run - 1))"
done

aggregate_summary_tsv "$RAW_SUMMARY_TSV" "$SUMMARY_TSV"

info "汇总表："
print_matrix_summary "$SUMMARY_TSV"
info "逐轮 TSV：$RAW_SUMMARY_TSV"
info "汇总 TSV：$SUMMARY_TSV"
