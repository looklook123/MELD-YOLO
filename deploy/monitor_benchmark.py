#!/usr/bin/env python3

import os
import sys
import time
import signal
import subprocess

INTERVAL = 0.5
NUM_CPUS = os.cpu_count() or 4


def read_proc_stat():
    with open("/proc/stat", "r") as f:
        line = f.readline().strip()

    parts = line.split()

    if parts[0] != "cpu":
        raise RuntimeError("Cannot read /proc/stat")

    values = list(map(int, parts[1:]))

    idle = values[3] + values[4]  # idle + iowait
    total = sum(values)

    return total, idle


def read_process_cpu_time(pid):
    with open(f"/proc/{pid}/stat", "r") as f:
        data = f.read()

    # comm can contain spaces/parentheses, so split after last ")"
    right = data.rfind(")")
    fields = data[right + 2:].split()

    # After removing pid + comm, field index:
    # original field 14 -> fields[11]
    # original field 15 -> fields[12]
    utime = int(fields[11])
    stime = int(fields[12])

    return utime + stime


def read_rss_mb(pid):
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except FileNotFoundError:
        return 0.0

    return 0.0


def main():

    if len(sys.argv) < 2:
        print("Usage:")
        print("python3 monitor_benchmark.py <command> [args...]")
        sys.exit(1)

    command = sys.argv[1:]

    print("=" * 60)
    print("Resource Monitor")
    print("=" * 60)
    print("Command :", " ".join(command))
    print("CPU cores:", NUM_CPUS)
    print("Sample interval:", INTERVAL, "s")
    print()

    start_time = time.time()

    process = subprocess.Popen(command)

    samples_cpu = []
    samples_rss = []

    prev_total, prev_idle = read_proc_stat()
    prev_proc = read_process_cpu_time(process.pid)

    clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])

    try:

        while process.poll() is None:

            time.sleep(INTERVAL)

            try:
                total, idle = read_proc_stat()
                proc_time = read_process_cpu_time(process.pid)
                rss = read_rss_mb(process.pid)
            except FileNotFoundError:
                break

            delta_total = total - prev_total
            delta_idle = idle - prev_idle
            delta_proc = proc_time - prev_proc

            prev_total = total
            prev_idle = idle
            prev_proc = proc_time

            if delta_total > 0:

                # Process CPU usage relative to whole machine.
                #
                # Process CPU time is measured in clock ticks.
                # System total CPU time is summed over all cores.
                #
                # Therefore this value is already normalized:
                # 100% = all CPU capacity.
                cpu_percent = (
                    delta_proc / delta_total
                ) * 100.0

            else:
                cpu_percent = 0.0

            samples_cpu.append(cpu_percent)
            samples_rss.append(rss)

    except KeyboardInterrupt:

        print("\nInterrupted.")

        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=5)
        except Exception:
            process.kill()

    return_code = process.wait()

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("Resource Usage Result")
    print("=" * 60)

    if samples_cpu:

        avg_cpu = sum(samples_cpu) / len(samples_cpu)
        peak_cpu = max(samples_cpu)

        avg_rss = sum(samples_rss) / len(samples_rss)
        peak_rss = max(samples_rss)

        print(f"Average CPU utilization : {avg_cpu:.2f}%")
        print(f"Peak CPU utilization    : {peak_cpu:.2f}%")
        print(f"Average Memory RSS      : {avg_rss:.2f} MB")
        print(f"Peak Memory RSS         : {peak_rss:.2f} MB")
        print(f"Samples                 : {len(samples_cpu)}")

    else:

        print("No monitoring samples collected.")

    print(f"Process elapsed time    : {elapsed:.2f} s")
    print(f"Process return code     : {return_code}")

    print("=" * 60)

    return return_code


if __name__ == "__main__":
    sys.exit(main())
