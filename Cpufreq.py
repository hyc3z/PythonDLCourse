import subprocess
import sys

def RunShellWithReturnCode(command,print_output=True,universal_newlines=True):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=universal_newlines)
    if print_output:
        output_array = []
        while True:
            line = p.stdout.readline()
            if not line:
                break
            print(line.strip("/n"))
            output_array.append(line)
        output ="".join(output_array)
    else:
        output = p.stdout.read()
    p.wait()
    errout = p.stderr.read()
    if print_output and errout:
        print(sys.stderr, errout)
    p.stdout.close()
    p.stderr.close()
    return output, p.returncode


if __name__ == '__main__':
    RunShellWithReturnCode('cat /proc/cpuinfo |grep MHz')

