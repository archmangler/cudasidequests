# Nvidia CUDA API and related tools and instrumentation


```
root@e8375681cc9a:~/cudasidequests/cu04# nvidia-smi --query-gpu=gpu_name,driver_version,temperature.gpu --format=csv
name, driver_version, temperature.gpu
NVIDIA GeForce RTX 3070, 580.65.06, 23
```


Setting the power limits:

```
nvidia-smi -i 0 -pl 200

```


Checking the clocks :

```
root@e8375681cc9a:~/cudasidequests/cu04# nvidia-smi -q -d CLOCK

==============NVSMI LOG==============

Timestamp                                 : Sun Nov 22 05:39:43 2026
Driver Version                            : 580.65.06
CUDA Version                              : 13.0

Attached GPUs                             : 1
GPU 00000000:01:00.0
    Clocks
        Graphics                          : 210 MHz
        SM                                : 210 MHz
        Memory                            : 405 MHz
        Video                             : 555 MHz
    Applications Clocks
        Graphics                          : N/A
        Memory                            : N/A
    Default Applications Clocks
        Graphics                          : N/A
        Memory                            : N/A
    Deferred Clocks
        Memory                            : N/A
    Max Clocks
        Graphics                          : 2100 MHz
        SM                                : 2100 MHz
        Memory                            : 7001 MHz
        Video                             : 1950 MHz
    Max Customer Boost Clocks
        Graphics                          : N/A
    SM Clock Samples
        Duration                          : Not Found
        Number of Samples                 : Not Found
        Max                               : Not Found
        Min                               : Not Found
        Avg                               : Not Found
    Memory Clock Samples
        Duration                          : Not Found
        Number of Samples                 : Not Found
        Max                               : Not Found
        Min                               : Not Found
        Avg                               : Not Found
    Clock Policy
        Auto Boost                        : N/A
        Auto Boost Default                : N/A

```
