EFA:

two p5en.48xlarge nodes connected by efa nics, each with:

```
nvidia-sminvidia-smi
Mon Mar  9 04:35:35 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H200                    On  |   00000000:59:00.0 Off |                    0 |
| N/A   62C    P0            434W /  700W |   87321MiB / 143771MiB |     68%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA H200                    On  |   00000000:5A:00.0 Off |                    0 |
| N/A   49C    P0            412W /  700W |   97399MiB / 143771MiB |     65%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA H200                    On  |   00000000:72:00.0 Off |                    0 |
| N/A   59C    P0            414W /  700W |   90717MiB / 143771MiB |     51%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA H200                    On  |   00000000:73:00.0 Off |                    0 |
| N/A   53C    P0            420W /  700W |   90237MiB / 143771MiB |     52%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA H200                    On  |   00000000:8B:00.0 Off |                    0 |
| N/A   32C    P0            116W /  700W |  135846MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA H200                    On  |   00000000:8C:00.0 Off |                    0 |
| N/A   29C    P0            111W /  700W |  135852MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA H200                    On  |   00000000:A4:00.0 Off |                    0 |
| N/A   35C    P0            116W /  700W |  136106MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA H200                    On  |   00000000:A5:00.0 Off |                    0 |
| N/A   30C    P0            116W /  700W |  135872MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A          969852      C   ray::MegatronTrainRayActor.train      87186MiB |
|    1   N/A  N/A          970554      C   ray::MegatronTrainRayActor.train      97264MiB |
|    2   N/A  N/A          970556      C   ray::MegatronTrainRayActor.train      90582MiB |
|    3   N/A  N/A          970553      C   ray::MegatronTrainRayActor.train      90102MiB |
|    4   N/A  N/A          972057      C   sglang::scheduler_TP0                 13583... |
|    5   N/A  N/A          972058      C   sglang::scheduler_TP1                 13584... |
|    6   N/A  N/A          972027      C   sglang::scheduler_TP0                 13609... |
|    7   N/A  N/A          972028      C   sglang::scheduler_TP1                 13586... |
+-----------------------------------------------------------------------------------------+

 lscpu
Architecture:                x86_64
  CPU op-mode(s):            32-bit, 64-bit
  Address sizes:             46 bits physical, 48 bits virtual
  Byte Order:                Little Endian
CPU(s):                      192
  On-line CPU(s) list:       0-191
Vendor ID:                   GenuineIntel
  Model name:                Intel(R) Xeon(R) Platinum 8488C
    CPU family:              6
    Model:                   143
    Thread(s) per core:      2
    Core(s) per socket:      48
    Socket(s):               2
    Stepping:                8
    BogoMIPS:                4800.00
    Flags:                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_goo
                             d nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq monitor ssse3 fma cx16 pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes x
                             save avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch ssbd ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rds
                             eed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx_vnni avx512_bf16 wbnoinvd ida arat avx512vbmi umip pku os
                             pke waitpkg avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq rdpid cldemote movdiri movdir64b md_clear serialize amx_bf16 avx512_fp16 
                             amx_tile amx_int8 flush_l1d arch_capabilities
Virtualization features:     
  Hypervisor vendor:         KVM
  Virtualization type:       full
Caches (sum of all):         
  L1d:                       4.5 MiB (96 instances)
  L1i:                       3 MiB (96 instances)
  L2:                        192 MiB (96 instances)
  L3:                        210 MiB (2 instances)
NUMA:                        
  NUMA node(s):              2
  NUMA node0 CPU(s):         0-47,96-143
  NUMA node1 CPU(s):         48-95,144-191
Vulnerabilities:             
  Gather data sampling:      Not affected
  Ghostwrite:                Not affected
  Indirect target selection: Not affected
  Itlb multihit:             Not affected
  L1tf:                      Not affected
  Mds:                       Not affected
  Meltdown:                  Not affected
  Mmio stale data:           Not affected
  Reg file data sampling:    Not affected
  Retbleed:                  Not affected
  Spec rstack overflow:      Not affected
  Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; PBRSB-eIBRS SW sequence; BHI BHI_DIS_S
  Srbds:                     Not affected
  Tsa:                       Not affected
  Tsx async abort:           Not affected
  Vmscape:                   Not affected

 numactl --hardware
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
node 0 size: 1023918 MB
node 0 free: 56650 MB
node 1 cpus: 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191
node 1 size: 1024023 MB
node 1 free: 616110 MB
node distances:
node   0   1 
  0:  10  21 
  1:  21  10 

a$ ibv_devinfo
hca_id: rdmap85s0
        transport:                      unspecified (4)
        fw_ver:                         0.0.0.0
        node_guid:                      dd28:fea6:0000:2500
        sys_image_guid:                 0000:0000:0000:0000
        vendor_id:                      0x1d0f
        vendor_part_id:                 61346
        hw_ver:                         0xEFA2
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x01
                        link_layer:             Unspecified

hca_id: rdmap86s0
        transport:                      unspecified (4)
        fw_ver:                         0.0.0.0
        node_guid:                      b726:9b4e:0000:1c00
        sys_image_guid:                 0000:0000:0000:0000
        vendor_id:                      0x1d0f
        vendor_part_id:                 61346
        hw_ver:                         0xEFA2
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x01
                        link_layer:             Unspecified

hca_id: rdmap87s0
        transport:                      unspecified (4)
        fw_ver:                         0.0.0.0
        node_guid:                      1db4:ebe8:0000:1400
        sys_image_guid:                 0000:0000:0000:0000
        vendor_id:                      0x1d0f
        vendor_part_id:                 61346
        hw_ver:                         0xEFA2
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x01
                        link_layer:             Unspecified

hca_id: rdmap88s0
        transport:                      unspecified (4)
        fw_ver:                         0.0.0.0
        node_guid:                      d9b8:635c:0000:0b00
        sys_image_guid:                 0000:0000:0000:0000
        vendor_id:                      0x1d0f
        vendor_part_id:                 61346
        hw_ver:                         0xEFA2
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x01
                        link_layer:             Unspecified

hca_id: rdmap110s0
        transport:                      unspecified (4)
        fw_ver:                         0.0.0.0
        node_guid:                      9762:ff72:0000:8d00
        sys_image_guid:                 0000:0000:0000:0000
        vendor_id:                      0x1d0f
        vendor_part_id:                 61346
        hw_ver:                         0xEFA2
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x01
                        link_layer:             Unspecified

hca_id: rdmap111s0
        transport:                      unspecified (4)
        fw_ver:                         0.0.0.0
        node_guid:                      a958:fabd:0000:8300
        sys_image_guid:                 0000:0000:0000:0000
        vendor_id:                      0x1d0f
        vendor_part_id:                 61346
        hw_ver:                         0xEFA2
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x01
                        link_layer:             Unspecified

hca_id: rdmap112s0
        transport:                      unspecified (4)
        fw_ver:                         0.0.0.0
        node_guid:                      bd13:f671:0000:7c00
        sys_image_guid:                 0000:0000:0000:0000
        vendor_id:                      0x1d0f
        vendor_part_id:                 61346
        hw_ver:                         0xEFA2
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x01
                        link_layer:             Unspecified

hca_id: rdmap113s0
        transport:                      unspecified (4)
        fw_ver:                         0.0.0.0
        node_guid:                      672e:6485:0000:7300
        sys_image_guid:                 0000:0000:0000:0000
        vendor_id:                      0x1d0f
        vendor_part_id:                 61346
        hw_ver:                         0xEFA2
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x01
                        link_layer:             Unspecified

hca_id: rdmap135s0
        transport:                      unspecified (4)
        fw_ver:                         0.0.0.0
        node_guid:                      7d18:d061:0001:2a00
        sys_image_guid:                 0000:0000:0000:0000
        vendor_id:                      0x1d0f
        vendor_part_id:                 61346
        hw_ver:                         0xEFA2
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x01
                        link_layer:             Unspecified

hca_id: rdmap136s0
        transport:                      unspecified (4)
        fw_ver:                         0.0.0.0
        node_guid:                      6b03:04cd:0001:2100
        sys_image_guid:                 0000:0000:0000:0000
        vendor_id:                      0x1d0f
        vendor_part_id:                 61346
        hw_ver:                         0xEFA2
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x01
                        link_layer:             Unspecified

hca_id: rdmap137s0
        transport:                      unspecified (4)
        fw_ver:                         0.0.0.0
        node_guid:                      0349:cae0:0001:1400
        sys_image_guid:                 0000:0000:0000:0000
        vendor_id:                      0x1d0f
        vendor_part_id:                 61346
        hw_ver:                         0xEFA2
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x01
                        link_layer:             Unspecified

hca_id: rdmap138s0
        transport:                      unspecified (4)
        fw_ver:                         0.0.0.0
        node_guid:                      1911:016b:0001:0b00
        sys_image_guid:                 0000:0000:0000:0000
        vendor_id:                      0x1d0f
        vendor_part_id:                 61346
        hw_ver:                         0xEFA2
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x01
                        link_layer:             Unspecified

hca_id: rdmap160s0
        transport:                      unspecified (4)
        fw_ver:                         0.0.0.0
        node_guid:                      e575:9872:0001:8d00
        sys_image_guid:                 0000:0000:0000:0000
        vendor_id:                      0x1d0f
        vendor_part_id:                 61346
        hw_ver:                         0xEFA2
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x01
                        link_layer:             Unspecified

hca_id: rdmap161s0
        transport:                      unspecified (4)
        fw_ver:                         0.0.0.0
        node_guid:                      f34e:8a37:0001:8300
        sys_image_guid:                 0000:0000:0000:0000
        vendor_id:                      0x1d0f
        vendor_part_id:                 61346
        hw_ver:                         0xEFA2
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x01
                        link_layer:             Unspecified

hca_id: rdmap162s0
        transport:                      unspecified (4)
        fw_ver:                         0.0.0.0
        node_guid:                      a78e:88cd:0001:7c00
        sys_image_guid:                 0000:0000:0000:0000
        vendor_id:                      0x1d0f
        vendor_part_id:                 61346
        hw_ver:                         0xEFA2
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x01
                        link_layer:             Unspecified

hca_id: rdmap163s0
        transport:                      unspecified (4)
        fw_ver:                         0.0.0.0
        node_guid:                      79e8:6b13:0001:7300
        sys_image_guid:                 0000:0000:0000:0000
        vendor_id:                      0x1d0f
        vendor_part_id:                 61346
        hw_ver:                         0xEFA2
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x01
                        link_layer:             Unspecified
```

AMD, 2 nodes, each with :
```
amd-smi
+------------------------------------------------------------------------------+
| AMD-SMI 26.2.0+021c61fc      amdgpu version: 6.16.6   ROCm version: 7.1.1    |
| VBIOS version: 00165724                                                      |
| Platform: Linux Baremetal                                                    |
|-------------------------------------+----------------------------------------|
| BDF                        GPU-Name | Mem-Uti   Temp   UEC       Power-Usage |
| GPU  HIP-ID  OAM-ID  Partition-Mode | GFX-Uti    Fan               Mem-Usage |
|=====================================+========================================|
| 0000:05:00.0    AMD Instinct MI355X | 0 %      50 °C   0          240/1400 W |
|   0       1       6        SPX/NPS1 | 0 %        N/A           283/294896 MB |
|-------------------------------------+----------------------------------------|
| 0000:15:00.0    AMD Instinct MI355X | 0 %      48 °C   0          250/1400 W |
|   1       3       7        SPX/NPS1 | 0 %        N/A           283/294896 MB |
|-------------------------------------+----------------------------------------|
| 0000:65:00.0    AMD Instinct MI355X | 0 %      51 °C   0          246/1400 W |
|   2       2       5        SPX/NPS1 | 0 %        N/A           283/294896 MB |
|-------------------------------------+----------------------------------------|
| 0000:75:00.0    AMD Instinct MI355X | 0 %      49 °C   0          242/1400 W |
|   3       0       4        SPX/NPS1 | 0 %        N/A           283/294896 MB |
|-------------------------------------+----------------------------------------|
| 0000:85:00.0    AMD Instinct MI355X | 0 %      51 °C   0          248/1400 W |
|   4       5       2        SPX/NPS1 | 0 %        N/A           283/294896 MB |
|-------------------------------------+----------------------------------------|
| 0000:95:00.0    AMD Instinct MI355X | 0 %      50 °C   0          245/1400 W |
|   5       7       3        SPX/NPS1 | 0 %        N/A           283/294896 MB |
|-------------------------------------+----------------------------------------|
| 0000:e5:00.0    AMD Instinct MI355X | 0 %      51 °C   0          246/1400 W |
|   6       6       1        SPX/NPS1 | 0 %        N/A           283/294896 MB |
|-------------------------------------+----------------------------------------|
| 0000:f5:00.0    AMD Instinct MI355X | 0 %      49 °C   0          240/1400 W |
|   7       4       0        SPX/NPS1 | 0 %        N/A           283/294896 MB |
+-------------------------------------+----------------------------------------+
+------------------------------------------------------------------------------+
| Processes:                                                                   |
|  GPU        PID  Process Name          GTT_MEM  VRAM_MEM  MEM_USAGE     CU % |
|==============================================================================|
|    0     326601  gpuagent                0.0 B     0.0 B      0.0 B    0.0 % |
|    1     326601  gpuagent                0.0 B     0.0 B      0.0 B    0.0 % |
|    2     326601  gpuagent                0.0 B     0.0 B      0.0 B    0.0 % |
|    3     326601  gpuagent                0.0 B     0.0 B      0.0 B    0.0 % |
|    4     326601  gpuagent                0.0 B     0.0 B      0.0 B    0.0 % |
|    5     326601  gpuagent                0.0 B     0.0 B      0.0 B    0.0 % |
|    6     326601  gpuagent                0.0 B     0.0 B      0.0 B    0.0 % |
|    7     326601  gpuagent                0.0 B     0.0 B      0.0 B    0.0 % |
+------------------------------------------------------------------------------+

lscpu
Architecture:                x86_64
  CPU op-mode(s):            32-bit, 64-bit
  Address sizes:             52 bits physical, 57 bits virtual
  Byte Order:                Little Endian
CPU(s):                      256
  On-line CPU(s) list:       0-127
  Off-line CPU(s) list:      128-255
Vendor ID:                   AuthenticAMD
  BIOS Vendor ID:            Advanced Micro Devices, Inc.
  Model name:                AMD EPYC 9575F 64-Core Processor
    BIOS Model name:         AMD EPYC 9575F 64-Core Processor                Unknown CPU @ 3.3GHz
    BIOS CPU family:         107
    CPU family:              26
    Model:                   2
    Thread(s) per core:      1
    Core(s) per socket:      64
    Socket(s):               2
    Stepping:                1
    Frequency boost:         enabled
    CPU(s) scaling MHz:      66%
    CPU max MHz:             5008.0068
    CPU min MHz:             0.0000
    BogoMIPS:                6589.96
    Flags:                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl n
                             onstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_leg
                             acy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp 
                             ibrs_enhanced vmmcall fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveo
                             pt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx_vnni avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc amd_ibpb_ret arat npt lb
                             rv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfn
                             i vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid bus_lock_detect movdiri movdir64b overflow_recov succor smca fsrm avx512_vp2intersect flush_l1d debug_swap
Virtualization features:     
  Virtualization:            AMD-V
Caches (sum of all):         
  L1d:                       6 MiB (128 instances)
  L1i:                       4 MiB (128 instances)
  L2:                        128 MiB (128 instances)
  L3:                        512 MiB (16 instances)
NUMA:                        
  NUMA node(s):              2
  NUMA node0 CPU(s):         0-63
  NUMA node1 CPU(s):         64-127
Vulnerabilities:             
  Gather data sampling:      Not affected
  Indirect target selection: Not affected
  Itlb multihit:             Not affected
  L1tf:                      Not affected
  Mds:                       Not affected
  Meltdown:                  Not affected
  Mmio stale data:           Not affected
  Reg file data sampling:    Not affected
  Retbleed:                  Not affected
  Spec rstack overflow:      Not affected
  Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; STIBP always-on; PBRSB-eIBRS Not affected; BHI Not affected
  Srbds:                     Not affected
  Tsa:                       Not affected
  Tsx async abort:           Not affected
  Vmscape:                   Not affected

root@chi2766:~# numactl --hardware
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
node 0 size: 1547791 MB
node 0 free: 1484025 MB
node 1 cpus: 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
node 1 size: 1548128 MB
node 1 free: 1483555 MB
node distances:
node   0   1 
  0:  10  32 
  1:  32  10 

   ibv_devices
    device                 node GUID
    ------              ----------------
    ionic_0             069081fffe36b328
    ionic_1             069081fffe36a5a8
    ionic_2             069081fffe369648
    ionic_3             069081fffe365880
    ionic_4             069081fffe369420
    ionic_5             069081fffe366360
    ionic_6             069081fffe36aec0
    ionic_7             069081fffe3686d0

root@chi2766:~# ibv_devices
    device                 node GUID
    ------              ----------------
    ionic_0             069081fffe36b328
    ionic_1             069081fffe36a5a8
    ionic_2             069081fffe369648
    ionic_3             069081fffe365880
    ionic_4             069081fffe369420
    ionic_5             069081fffe366360
    ionic_6             069081fffe36aec0
    ionic_7             069081fffe3686d0
root@chi2766:~# ibv_devinfo
hca_id: ionic_0
        transport:                      InfiniBand (0)
        fw_ver:                         1.117.1-a-63
        node_guid:                      0690:81ff:fe36:b328
        sys_image_guid:                 0690:81ff:fe36:b328
        vendor_id:                      0x1dd8
        vendor_part_id:                 4098
        hw_ver:                         0x0
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

hca_id: ionic_1
        transport:                      InfiniBand (0)
        fw_ver:                         1.117.1-a-63
        node_guid:                      0690:81ff:fe36:a5a8
        sys_image_guid:                 0690:81ff:fe36:a5a8
        vendor_id:                      0x1dd8
        vendor_part_id:                 4098
        hw_ver:                         0x0
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

hca_id: ionic_2
        transport:                      InfiniBand (0)
        fw_ver:                         1.117.1-a-63
        node_guid:                      0690:81ff:fe36:9648
        sys_image_guid:                 0690:81ff:fe36:9648
        vendor_id:                      0x1dd8
        vendor_part_id:                 4098
        hw_ver:                         0x0
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

hca_id: ionic_3
        transport:                      InfiniBand (0)
        fw_ver:                         1.117.1-a-63
        node_guid:                      0690:81ff:fe36:5880
        sys_image_guid:                 0690:81ff:fe36:5880
        vendor_id:                      0x1dd8
        vendor_part_id:                 4098
        hw_ver:                         0x0
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

hca_id: ionic_4
        transport:                      InfiniBand (0)
        fw_ver:                         1.117.1-a-63
        node_guid:                      0690:81ff:fe36:9420
        sys_image_guid:                 0690:81ff:fe36:9420
        vendor_id:                      0x1dd8
        vendor_part_id:                 4098
        hw_ver:                         0x0
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

hca_id: ionic_5
        transport:                      InfiniBand (0)
        fw_ver:                         1.117.1-a-63
        node_guid:                      0690:81ff:fe36:6360
        sys_image_guid:                 0690:81ff:fe36:6360
        vendor_id:                      0x1dd8
        vendor_part_id:                 4098
        hw_ver:                         0x0
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

hca_id: ionic_6
        transport:                      InfiniBand (0)
        fw_ver:                         1.117.1-a-63
        node_guid:                      0690:81ff:fe36:aec0
        sys_image_guid:                 0690:81ff:fe36:aec0
        vendor_id:                      0x1dd8
        vendor_part_id:                 4098
        hw_ver:                         0x0
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

hca_id: ionic_7
        transport:                      InfiniBand (0)
        fw_ver:                         1.117.1-a-63
        node_guid:                      0690:81ff:fe36:86d0
        sys_image_guid:                 0690:81ff:fe36:86d0
        vendor_id:                      0x1dd8
        vendor_part_id:                 4098
        hw_ver:                         0x0
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

```