import subprocess
import time as t

for i in range(100):
    # # Switch from performance to powersave ~ 25-35 ms
    # subprocess.check_call(['sudo','sh','-c','echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor'])
    # start_t = t.time()
    # subprocess.check_call(['sudo','sh','-c','echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor'])
    # end_t = t.time()
    # subprocess.check_call(['sudo','sh','-c','echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor'])

    # Switch from powersave to performance ~ 48-65 ms
    subprocess.check_call(['sudo','sh','-c','echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor'])
    start_t = t.time()
    subprocess.check_call(['sudo','sh','-c','echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor'])
    end_t = t.time()
    subprocess.check_call(['sudo','sh','-c','echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor'])

    #subprocess.Popen(['cat', '/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq'])
    #subprocess.Popen(['cat', '/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq'])
    
    # if round((end_t-start_t)*1000,2) > 28:
    if round((end_t-start_t)*1000,2) > 50:
        print(f"time at iteration {i} in ms: {round((end_t-start_t)*1000,2)}")