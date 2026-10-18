# CUDA 方向环境搭建与复现说明

==说明：本文是技术报告的附录材料，覆盖进阶阶段（A1–A4 与最终关 6-1 GEMM）代码的**全部环境搭建与复现方法**。原先分散在「虚拟机环境配置说明」「环境与复现说明」「复现包 README」三处的环境内容已合并到这一份，按"先选路线 → 再照做"的顺序组织，照着重做一遍应当得到同样的结论。==

---

## 0. 先选路线

不同的人需要的东西不一样，先对号入座，不必从头读。

| 你的身份 / 目的 | 走哪条 | 花多久 |
| --- | --- | --- |
| 只想确认**代码对不对**，不想折腾环境 | **路线 A**（第 3 章）：解压复现包跑 `run.sh` | 1 分钟 |
| 想**从零搭一台**和开发时一样的无卡虚拟机 | **路线 B**（第 4 章）：VMware + Ubuntu 24.04 | 40 分钟 |
| 手上**有 NVIDIA 显卡**（WSL2 / 云 GPU / 物理机） | **路线 C**（第 5 章）：真机验证包 | 10 分钟 |
| 已经有一台能编译的 Linux，只想核对命令 | 第 6、7、12 章 | 5 分钟 |
| 跑出来一堆 `FAIL`，想知道是不是代码错了 | 第 9.3 节（先看数学指纹，别急着改代码） | 2 分钟 |
| 报错了，找原因 | 第 11 章报错对照表 | — |

一句话概括三条路线的关系：

- **路线 A** 用 Python 逐字复刻 kernel 语义，不需要任何显卡，就能把"算得对不对"钉死；
- **路线 B** 补上"平台能不能编过、`solve` 符号对不对、shared 超没超限"这三项；
- **路线 C** 才能给出真实耗时——而成绩里的 `performanceMedianUs` 本来就只能由平台或真卡给出。

---

## 1. 十秒判断这台机器有没有 GPU

装任何东西之前先做这一步。整段贴进终端：

```bash
echo "===== 系统 ====="; grep -E '^(NAME|VERSION)=' /etc/os-release
echo "===== 显示设备 ====="; lspci | grep -iE 'nvidia|vga|display|3d'
echo "===== 设备节点 ====="; ls /dev/nvidia* 2>/dev/null || echo "无 /dev/nvidia*"
echo "===== 驱动 ====="; nvidia-smi 2>&1 | head -3
echo "===== 已有工具 ====="; command -v nvcc gcc make || true
```

开发机上的实测输出（一台 VMware 虚拟机）：

```text
Ubuntu 24.04.5 LTS (Noble Numbat)
00:0f.0 VGA compatible controller: VMware SVGA II Adapter
无 /dev/nvidia*
nvidia-smi: command not found
（nvcc / gcc / make 均无输出，说明都没装）
```

判读只有三种情况：

| 看到什么 | 结论 | 后面怎么做 |
| --- | --- | --- |
| `nvidia-smi` 打出显卡表格 | 有可用 GPU | 走路线 C，先补驱动再装 Toolkit |
| `lspci` 里有 NVIDIA，但 `nvidia-smi` 报错 | 卡在，驱动没装 | 补驱动 |
| **只有 `VMware SVGA II Adapter`** | **没有 GPU** | 走路线 A + B，只做编译验证与 Python 对拍 |

---

## 2. 交付物与验证工具的分工

先把"哪些要交、哪些只是自测"分清，后面所有命令都建立在这个区分上。

| 文件 | 作用 | 是否提交 |
| --- | --- | --- |
| `gemm_opt.cu` | 6-1 的交付文件，入口 `extern "C" void solve(...)` | **是** |
| `report.md` | 6-1 的技术报告 | **是** |
| `test_gemm.cu` | 自测程序：与 CPU 参考结果对拍 + CUDA event 计时 | 否 |
| `verify_gemm.py` | 用 Python 逐字复刻 kernel 语义，无 GPU 也能验正确性 | 否 |
| `run.sh` | 复现包的一键脚本 | 否 |
| `expected_verify_output.txt` | `verify_gemm.py` 的黄金输出（80 组） | 否 |
| `MANIFEST.txt` | 四个文件的 SHA-256 | 否 |

容易搞混的一点：`test_*.cu` 不是"给平台跑的东西"，它们只是自己造一个 CPU 参考实现来对拍。**平台只认 `solve` 这个入口。**

---

## 3. 路线 A：最省事——跑复现包

### 3.1 一条命令

解压 `gemm_repro.zip`（根目录是 `gemm_repro/`），然后：

```bash
cd gemm_repro
bash run.sh
```

不需要改任何代码，不需要装任何库。

### 3.2 `run.sh` 会做什么

它先探测本机有什么，能做多少做多少，**任何机器都能跑出一个结论**：

| 本机条件 | 会做什么 | 结论强度 |
| --- | --- | --- |
| 只有 `python3` | CPU 逐字复刻 kernel 逻辑，80 组尺寸对拍，并与黄金输出逐字比对 | **能证明算法与边界正确** |
| `python3` + `nvcc` + NVIDIA GPU | 追加：编译、链接、`-Xptxas -v` 资源用量、16 组尺寸端到端 + 耗时 | 最完整 |
| 连 `python3` 都没有 | 打印环境信息，指出缺什么 | 无法下结论 |

脚本分为四段：`0) 环境` → `1) 逻辑正确性` → `2) 编译与端到端` → `3) 文件指纹`。

退出码：`0` = 已做的检查全部通过；`1` = 有检查未通过。

> 无卡时 `test_gemm` 返回 `2`，这被 `run.sh` 计为"跳过"而非失败，不会把总分拉成 `1`。

### 3.3 判据

**CPU 侧（必做）**：`verify_gemm.py` 的输出与 `expected_verify_output.txt` **逐字相同**即通过。关键看两列——「误差判定」全 `PASS`、「逐位相同」全是「是」。逐位相同说明复刻的不是"另一个实现"，而是同一个执行过程。

**GPU 侧（有卡时）**：与题面同一条公式，逐元素

```text
|a − e| ≤ 1e-3 + 1e-3·|e|      （占容差比例 ≤ 1.0 即合规）
```

期望看到 16 行 `PASS`，末尾一行 `512^3 占容差比例 = ...（<=1 即合规）`。

### 3.4 文件指纹

```bash
sha256sum -c MANIFEST.txt
```

四项全 `OK` 说明手上的 `gemm_opt.cu` 就是被验证过的那一份，中间没有经过手工改动或转贴失真。完整指纹见附录 B。

---

## 4. 路线 B：从零搭一台无卡虚拟机

这一章是开发时实际走过的路，包括踩过的坑。

### 4.1 为什么用虚拟机，以及它为什么只能编译不能跑

1、进阶阶段的代码由平台统一编译、统一启动，本地没有可以反复试跑的入口，交上去才知道过不过，代价太大。

2、开发机是笔记本，独显 RTX 5060 Laptop 走 **Optimus 架构**，物理上无法把整块 PCI 设备直通给虚拟机。VMware 的「3D 加速」只加速 OpenGL，不提供 CUDA，所以虚拟机里永远看不到显卡。

3、但 `nvcc` 编译只依赖 CUDA Toolkit，**不依赖驱动**。虚拟机虽然跑不了 kernel，却能把「编译」这一层完整走通。

4、于是分工是这样的：

| 谁 | 负责验证什么 |
| --- | --- |
| 虚拟机 | 语法对不对、头文件全不全、`solve` 符号对不对、shared 超没超限 |
| Python 复刻脚本 | 算得对不对、边界处理对不对 |
| 平台评测 | 真实耗时 `performanceMedianUs` |

5、一句话：**虚拟机是安检口，不是跑道。** 它不负责跑得快，只负责盖章。

### 4.2 虚拟机规格

| 项目 | 值 |
| --- | --- |
| 宿主 | Windows，NVIDIA GeForce RTX 5060 Laptop GPU（Blackwell，`sm_120`，8 GB，26 个 SM），驱动 591.91 |
| 虚拟化软件 | VMware® Workstation 17 Pro |
| 客户机 | Ubuntu 24.04.3 LTS Server（`ubuntu-24.04.3-live-server-amd64.iso`），默认设置安装 |
| 初始磁盘 | 20 GB，其中 `/` 只有 9.8 GB |
| 扩容后 | `/` 18 GB，可用约 10 GB |
| 主机编译器 | gcc 13.3.0 |
| CUDA Toolkit | 12.0（Ubuntu 24.04 仓库的 `nvidia-cuda-toolkit`） |
| GPU 状态 | **无**。`lspci` 只有 `VMware SVGA II Adapter`，没有 `/dev/nvidia*` |

> 装的是 24.04.3 镜像，装完随包更新后 `os-release` 显示 24.04.5，不影响任何步骤。

### 4.3 步骤 1：体检

即第 1 章那段脚本。确认"只有 VMware SVGA II Adapter"之后再往下走。

### 4.4 步骤 2：先把磁盘扩到 18 GB（必须最先做）

#### 为什么必须先做

默认安装 `nvidia-cuda-toolkit` 时 `/` 只有 9.8 GB，装到一半 `df` 就是 100%，`dpkg` 直接报 `No space left on device`，`nvidia-cuda-dev` 解包失败，留下一堆半装状态的包。之后再装任何东西都会连锁报错。

所以顺序必须是：**先扩容，再装包**。反过来做，后面每一步都在填前面的坑。

#### 先看清容量卡在哪一层

```bash
lsblk -f
sudo pvs; sudo vgs; sudo lvs; sudo parted -l
```

实测输出：

```text
NAME                       SIZE TYPE FSTYPE      MOUNT
sda                         20G disk
├─sda1                       1M part
├─sda2                     1.8G part ext4        /boot
└─sda3                    18.2G part LVM2_member
  └─ubuntu--vg-ubuntu--lv   10G lvm  ext4        /

PV         VG        Fmt  Attr PSize  PFree
/dev/sda3  ubuntu-vg lvm2 a--  18.22g 8.22g
VG        #PV #LV #SN Attr   VSize  VFree
ubuntu-vg   1   1   0 wz--n- 18.22g 8.22g
LV        VG        Attr       LSize
ubuntu-lv ubuntu-vg -wi-ao---- 10.00g
```

一眼就能看出问题：分区 `sda3` 已经占满整盘（后面只剩 1 MB 对齐空隙），**但卷组里有 8.22 GB 没分配给逻辑卷**。

打个比方：卷组是一大桶水（18.22 L），逻辑卷是手里的杯子（10 L）。桶里有水，杯子装不下。

所以 `growpart` 和 `pvresize` **都不用做**，只差最后两步。

#### 执行扩容

```bash
# ① 把卷组里剩下的空间全部给逻辑卷
sudo lvextend -l +100%FREE /dev/ubuntu-vg/ubuntu-lv

# ② 让 ext4 认识新尺寸（在线扩容，不用卸载、不用重启）
sudo resize2fs /dev/ubuntu-vg/ubuntu-lv

# ③ 确认
df -h /
```

预期：`/` 从 `9.8G` 变成 `18G`，可用从 `2.0G` 变成约 `10G`。

> 也可以合并成 `sudo lvextend -r -l +100%FREE ...`（`-r` 自动调 resize2fs）。建议分开敲，报错时能看出是哪一层没成功。

#### 清理之前装坏的半装状态

```bash
sudo dpkg --configure -a
sudo apt-get -f install
```

这两条把「解包失败」留下的未配置包补齐。跑完不报 `dpkg: error` 再往下走。

### 4.5 步骤 3：装编译链

三个包，各有各的用处，不要少装：

```bash
sudo apt update

# ① 主机编译器与 make（nvcc 编译主机代码要靠 gcc）
sudo apt install -y build-essential

# ② CUDA 编译器（Ubuntu 自带源，版本 12.0）
sudo apt install -y nvidia-cuda-toolkit

# ③ 头文件（cuda_runtime.h 在 nvidia-cuda-dev 里，不在 toolkit 元包里）
sudo apt install -y --no-install-recommends nvidia-cuda-dev
```

为什么用 Ubuntu 自带源而不是 NVIDIA 官方源：

1、虚拟机没有显卡，装 12.8 也跑不起来，版本新不带来任何收益。

2、官方源体积大得多（完整 `cuda-toolkit-12-8` 约 3.5 GB），自带源这套装完约 2 GB，`/` 只有 18 GB，要留余量。

3、本机显卡需要 12.8 才能直编 `sm_120`，但虚拟机只做「语法与符号」验证，用 `sm_75` 一样能验（见 4.6）。

装完确认三件事：

```bash
nvcc --version | tail -2
gcc --version | head -1
dpkg -L nvidia-cuda-dev | grep cuda_runtime.h
```

期望输出：

```text
Cuda compilation tools, release 12.0, V12.0.xxx
gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
/usr/include/cuda_runtime.h
```

第三条尤其重要。之前 `cuda_runtime.h: No such file or directory` 就是这行没输出，说明 `nvidia-cuda-dev` 还在半装状态，回 4.4 收拾干净再装。

### 4.6 步骤 4：三条编译验证命令

#### 为什么要加 `-allow-unsupported-compiler`

CUDA 12.0 支持的 gcc 上限低于 Ubuntu 24.04 自带的 13.3，直接编会报：

```text
error: unsupported GNU version! gcc versions later than 12 are not supported!
```

加这个参数绕过版本检查：

```bash
ALLOW="-allow-unsupported-compiler"
```

它只放宽版本号判断，**不改任何代码语义**。我们用的都是 CUDA 1.0 时代就有的基础特性（`__global__`、`__shared__`、`__syncthreads()`、`<<<>>>`），gcc 13 编出来没有兼容问题。

#### ① 语法检查：`nvcc -c`

```bash
cd ~/cuda
nvcc -O3 $ALLOW -c gemm_opt.cu -o /dev/null && echo "[OK] gemm_opt.cu 语法通过"
```

`-c` 只编译不链接，**不需要驱动**，这是无卡机器能用的关键。四关的交付文件同样处理：

```bash
nvcc -O3 $ALLOW -c hello_cuda.cu   -o /dev/null && echo "A1 OK"
nvcc -O3 $ALLOW -c vector_add.cu   -o /dev/null && echo "A2 OK"
nvcc -O3 $ALLOW -c matrix_add.cu   -o /dev/null && echo "A3 OK"
nvcc -O3 $ALLOW -c tiled_matmul.cu -o /dev/null && echo "A4 OK"
```

> 不要用 `nvcc -ptx` 代替 `-c`。`-ptx` 只走设备那一半，`solve()` 里的 `dim3` 参数、启动配置、函数名拼错它一概发现不了；`-c` 两半都编。

#### ② 链接检查：这一步价值最高

```bash
nvcc -O3 $ALLOW -o test_gemm test_gemm.cu gemm_opt.cu && echo "[OK] 链接成功"
```

能链上，说明 `extern "C" void solve(const float*, const float*, float*, int, int, int)` 这个符号名和签名与测试程序声明的**完全一致**。这是最容易错、后果最重的一项：平台找不到 `solve` 会直接判零分，而 `-c` 是发现不了的。

#### ③ 资源用量：`-Xptxas -v`

```bash
nvcc -O3 $ALLOW -arch=sm_75 -Xptxas -v -c gemm_opt.cu -o /dev/null
```

期望看到类似：

```text
ptxas info    : Used NN registers, 8256 bytes smem, 384 bytes cmem[0]
```

要核对两个数：

| 指标 | 期望 | 超了会怎样 |
| --- | --- | --- |
| `bytes smem` | **8256** | 平台上限 48 KiB（49152 B），超了拒绝启动 |
| `registers` | 40–64 | 超过 255 会溢出到 local memory，性能崩 |
| `spill` 字样 | **不出现** | 出现说明寄存器不够，要调 `TM/TN` |

8256 这个数可以手算核对：

```text
As = BK × (BM + 1) = 16 × 65  = 1040 个 float
Bs = BK × BN       = 16 × 64  = 1024 个 float
合计 2064 个 float × 4 B = 8256 B
```

> 关于 `-arch=sm_75`：CUDA 12.0 不认识 `sm_120`，所以这里指定一个它认识的架构。**shared 用量是编译期常量，与架构无关**；寄存器数会随架构略有浮动，但量级一致，够用来判断有没有 spill。

#### 可选：确认循环真的展开了

```bash
nvcc $ALLOW -arch=sm_75 -ptx gemm_opt.cu -o gemm_opt.ptx
grep -c "fma.rn.f32" gemm_opt.ptx
```

`fma.rn.f32` 是融合乘加指令。数目很大（几十到上百）说明 `#pragma unroll` 生效、累加被编成硬件 FMA；只有个位数说明循环没展开，性能会差，但**正确性不受影响**。

### 4.7 步骤 5：运行与判读

```bash
./test_gemm; echo "退出码=$?"
```

无卡时的正常输出：

```text
[SKIP] 本机没有可用的 NVIDIA 显卡，跳过运行阶段。
       cudaGetDeviceCount -> no CUDA-capable device is detected (count = 0)

       这不是代码错误：能走到这里，说明 **编译和链接都已成功**，
       extern "C" void solve(...) 的符号对得上，
       只是 kernel / cudaMalloc / cudaMemcpy 没有被真正执行。

       正确性验证请改用：python3 verify_gemm.py
退出码=2
```

退出码含义见 9.1。

> A2/A3/A4 的 `test_*.cu` 没有加这个前置保护，在虚拟机里**不要跑**它们，只做 `-c`；要看运行结果必须上真卡。

### 4.8 文件怎么传进虚拟机

三种方式，按省事程度排：

1、**拖拽**。VM 里先装 `open-vm-tools-desktop`，装完必须重启才生效：

```bash
sudo apt install -y open-vm-tools open-vm-tools-desktop
sudo reboot
```

2、**共享文件夹**。虚拟机关机状态下，设置 → 选项 → 共享文件夹 → 总是启用 → 添加，主机路径指向代码目录，名称填 `share`。开机后：

```bash
sudo mkdir -p /mnt/hgfs
sudo vmhgfs-fuse .host:/ /mnt/hgfs -o allow_other
ls /mnt/hgfs/share/
```

改代码只要在 Windows 侧改，虚拟机里立刻能看到。

3、**heredoc 直接粘**。文件小的时候最快，也不依赖 VMware Tools：

```bash
cat > gemm_opt.cu <<'EOF'
...（整份代码）
EOF
```

### 4.9 一条命令替代上面全部手敲

`CUDA_VM一键部署.sh` 把 11 个文件（5 个交付源码 + 5 个自测程序 + `verify_gemm.py`）和六项检查全打包在一个 bash 文件里。**它不装驱动、不改系统、不联网、不申请 sudo**，只做落盘与检查。

用法（二选一）：

```bash
# ① 整段粘贴进终端，回车
# ② 把文件放进虚拟机后
bash CUDA_VM一键部署.sh            # 默认落到 ~/cuda-work
bash CUDA_VM一键部署.sh ~/mywork   # 也可以指定目录
```

六项检查的顺序：

| 段 | 做什么 | 无卡时 |
| --- | --- | --- |
| 1 | 打印 nvcc / gcc 版本，确认 `cuda_runtime.h` 就位 | 正常 |
| 2 | 逐关 `nvcc -c`（语法与头文件） | 正常 |
| 3 | 链接 `test_gemm`（验证 `solve` 符号） | 正常 |
| 4 | `-Xptxas -v` 打印 shared / 寄存器 / spill | 正常 |
| 5 | 运行 `test_gemm` | 得到 `[SKIP]` + 退出码 2 |
| 6 | `python3 verify_gemm.py` 逻辑对拍 | 正常 |

它会**先试编一次，命中 `unsupported GNU version` 才自动补** `-allow-unsupported-compiler` 重试，不需要事先判断本机 gcc 版本。这是旧脚本的教训——旧的 `CUDA测试_一键粘贴.sh` 没带这个参数，在新版 Ubuntu 上会 5 个文件全 FAIL。

末尾会打印结论：编译与链接全过 = 代码可以提交；有 `[FAIL]` 就照提示看。

---

## 5. 路线 C：有真显卡

### 5.1 三种可用环境

| 环境 | 适合 | 注意 |
| --- | --- | --- |
| **WSL2（Ubuntu 24.04）** | 本机就有 NVIDIA 显卡，最省事 | 需 Windows 侧装 NVIDIA 驱动，**WSL 内不要再装 Linux 驱动**；Toolkit 单独装 |
| **云 GPU（AutoDL / 矩池云等）** | 本机无卡 | 镜像多已带 CUDA，先确认 `nvcc --version` 与 `nvidia-smi` 的版本是否匹配 |
| **物理 Linux** | 长期使用 | 驱动 + Toolkit 一起装，注意 `sm_120` 需 CUDA ≥ 12.8 |

### 5.2 真机验证包

`CUDA真机验证_一键粘贴.sh` 适用于以上三处。整份粘贴进终端，或存成 `run.sh` 后 `bash run.sh`。它做五件事：

1、把 `gemm_opt.cu`（交付）和 `test_gemm.cu`（自测，不提交）写到 `~/gemm_test/`；

2、打印显卡信息，用 `nvidia-smi --query-gpu=compute_cap` 自动探测计算能力；

3、用**实际架构**编译（RTX 50 系是 `sm_120`，必须 CUDA ≥ 12.8）；

4、跑 16 组尺寸：正确性对拍 + CUDA event 中位耗时；

5、打印寄存器 / shared 用量。

任何一步失败都会打印原因并退出，不会留下半截环境。

### 5.3 架构对照表

手工指定 `-arch` 时用：

| 显卡 | 架构 | `-arch` | 所需 CUDA 版本 |
| --- | --- | --- | --- |
| GTX 9xx | Maxwell | `sm_52` | ≥ 6.5 |
| GTX 10xx | Pascal | `sm_61` | ≥ 8.0 |
| RTX 20xx / T4 | Turing | `sm_75` | ≥ 10.0 |
| A100 | Ampere | `sm_80` | ≥ 11.0 |
| RTX 30xx | Ampere | `sm_86` | ≥ 11.1 |
| RTX 40xx | Ada | `sm_89` | ≥ 11.8 |
| H100 | Hopper | `sm_90` | ≥ 11.8（推荐 12.x） |
| **RTX 50xx** | **Blackwell** | **`sm_120`** | **≥ 12.8** |

不确定就写 `-arch=native`（CUDA 11.5+ 自动匹配本机显卡，最省事）；再不确定就省略该参数，nvcc 用默认架构——能跑，只是性能非最优。

> ⚠️ RTX 50 系列（Blackwell）要求 CUDA **12.8 及以上**，低版本会报
> `nvcc fatal : Unsupported gpu architecture 'compute_120'`。
> 若复现环境是 RTX 50 系列且 CUDA 版本较低，请先升级 Toolkit，
> 或改用 `-arch=sm_90` 让驱动做 PTX JIT（能跑，但不是最佳性能）。

---

## 6. 代码依赖：为什么一条 `nvcc` 就够

### 6.1 一句话结论

`gemm_opt.cu` **除 CUDA Runtime 外没有任何第三方依赖**，不链接 cuBLAS / cuDNN / NPP，不使用 Driver API、不使用 Thrust、不使用 C++17 及以上特性。任何装好 CUDA Toolkit（≥ 11.0）且有可用 NVIDIA 显卡的 Linux 或 Windows 环境，用一条 `nvcc` 命令即可完成编译与运行。

### 6.2 头文件

| 文件 | 用到的头文件 | 说明 |
| --- | --- | --- |
| `gemm_opt.cu`（交付） | `#include <cuda_runtime.h>` | **仅此一个** |
| `test_gemm.cu`（自测） | `cuda_runtime.h` + `<cstdio> <cstdlib> <cmath> <vector> <algorithm> <random>` | 全部为标准库 |

`verify_gemm.py` 只用 Python 标准库，**不需要 numpy**，Ubuntu 自带 `python3` 直接能跑。

### 6.3 用到的 CUDA 特性（均为早期版本即支持）

| 特性 | 首次支持 | 用途 |
| --- | --- | --- |
| `__global__` / `<<<grid, block>>>` | CUDA 1.0 | kernel 定义与启动 |
| `__shared__` 静态共享内存 | CUDA 1.0 | 分块缓存 A、B 的 tile |
| `__syncthreads()` | CUDA 1.0 | 装载与计算之间的屏障（共 2 处） |
| 三维内置变量 `threadIdx/blockIdx/blockDim` | CUDA 1.0 | 线程映射 |
| `__restrict__` | CUDA 2.0 | 帮助编译器做别名分析 |
| `#pragma unroll` | CUDA 2.0 | 展开小循环 |
| `cudaDeviceSynchronize()` / `cudaGetLastError()` | CUDA 1.0 | 结尾同步、清错误状态 |

**没有使用**：动态共享内存、`__shfl_*` warp shuffle、Tensor Core（`wmma`）、`__ldg`、原子操作、协作组（cooperative groups）、异步拷贝（`cp.async`）。

因此代码**不绑定任何特定架构代次**。

### 6.4 接口约定

```c
extern "C" void solve(const float* A, const float* B, float* C,
                      int M, int K, int N);
```

- `extern "C"`：避免 C++ 名称修饰（name mangling），保证从 `test_gemm.cu` 直接链接可得。
- 三个指针都是**平台已分配好的显存指针**，`solve` 内不做任何分配、拷贝、释放。
- `M / K / N` 为 32 位 `int`，符合题面对标量参数的要求。
- 矩阵均为**行主序**：`A[m*K + k]`、`B[k*N + n]`、`C[m*N + n]`。

对应到题面的硬性约束：128 次启动/例（本例只用 1 次）、32 个参数/kernel（本例 6 个）、1024 线程/block（本例 256）、动态 shared ≤ 48 KiB（本例 0，全部静态）、禁用 `cudaMalloc` / 主机传输 / 自定义 stream / Driver API / host 端解引用平台指针——以上在本实现中均未触碰。

### 6.5 编译期常量

| 宏 | 值 | 含义 |
| --- | --- | --- |
| `BM` / `BN` | 64 / 64 | block tile 尺寸 |
| `BK` | 16 | K 方向分块的厚度 |
| `TM` / `TN` | 4 / 4 | 每线程负责的输出块 |
| 派生 | `THREADS_X = BN/TN = 16`，`THREADS_Y = BM/TM = 16`，共 256 线程 | |

修改这四个宏即可改变配置；`THREADS_X/Y`、`A_LOADS/B_LOADS` 会自动跟着变，无需手工同步。

### 6.6 编译命令

```bash
# 最小检查：只编译，不生成可执行文件（不需要显卡，CI 里也能跑）
nvcc -O3 -c gemm_opt.cu -o /dev/null

# 完整构建：链接自测程序
nvcc -O3 -o test_gemm test_gemm.cu gemm_opt.cu

# 资源用量
nvcc -O3 -arch=native -Xptxas -v -c gemm_opt.cu -o /dev/null
```

---

## 7. 资源用量与平台上限核对

| 项 | 值 | 平台上限 | 余量 |
| --- | --- | --- | --- |
| 动态 shared memory | 0（全部为静态分配） | 48 KiB | — |
| 静态 shared memory | **8256 B ≈ 8.06 KiB** | 48 KiB | 6 倍余量 |
| 每 block 线程数 | 256 | 1024 | 4 倍余量 |
| grid 维度（512³） | (8, 8) = 64 blocks | 65535 | 充裕 |
| 每例 kernel 启动次数 | **1** | 128 | — |
| kernel 参数个数 | 6 | 32 | — |

shared 明细：

```c
__shared__ float As[BK][BM + 1];   // 16 × 65 × 4 = 4160 B
__shared__ float Bs[BK][BN];       // 16 × 64 × 4 = 4096 B
```

`As` 的第二维写成 `BM + 1` 而非 `BM`，是为了在**转置写入**时把 bank 冲突打散（不填充时相邻线程会集中落在同一个 bank 上）。

寄存器用量用 4.6 的 `-Xptxas -v` 命令查看，不同 CUDA 版本通常在 40~64 个/线程。**应确认 `spill` 为 0**——若发生寄存器溢出到 local memory，性能会明显下降。

---

## 8. 无 GPU 下怎么证明算对了

### 8.1 为什么复刻能代替跑一遍

`gemm_opt.cu` 的 kernel 是**确定性**的：没有原子操作、没有随机、没有跨线程写同一地址，每个输出元素沿着 `k = 0 → K-1` 严格顺序累加。既然执行顺序唯一，就可以用一段 Python 把这个顺序原样走一遍，得到的浮点结果与真机**逐位相同**。

这不是「另写一个实现」，是「照抄它的执行过程」。

### 8.2 复刻了哪些语义

1、`As[k][m]` 的**转置**写入下标，以及 `+1` 填充列的位置。

2、越界装载时**无条件填 0**（不是跳过），以及全局地址 `A[(by*BM + m)*K + t*BK + k]` 的完整算式。

3、两次 `__syncthreads()` 的位置语义：装完才准读、读完才准覆盖。

4、`K` 方向分批时 shared 被覆盖后的取值，确保不会读到上一轮的脏数据。

5、写回时 `row < M && col < N` 的判断，以及与 `acc[4][4]` 对应的下标映射。

### 8.3 跑法与期望输出

```bash
python3 verify_gemm.py
```

```text
    M    K    N   误差判定    逐位相同      最大绝对误差    占容差比例
------------------------------------------------------------------------
    1    1    1   PASS        是            0.000e+00     0.000e+00
    2    3    4   PASS        是            0.000e+00     0.000e+00
...
------------------------------------------------------------------------
随机尺寸 60 例：60/60 通过
========================================================================
总结论： 全部通过，逻辑与边界处理正确 ✔
```

判据看两处：**「逐位相同」全是「是」**，以及最后一行 `总结论：全部通过`。

覆盖范围：20 组固定尺寸 + 60 组随机尺寸 = **80 组**。固定尺寸里特意混了不整除 tile 的（`17×15×16`、`63×64×65`、`127×129×131`、`511×17×33`）和某维为 1 的极端形状（`1×256×1`、`512×1×512`）——这两类是边界 bug 的主要暴露点。

### 8.4 复刻的局限

它测不出数据竞争、漏写 `__syncthreads()` 会不会错、bank 冲突、occupancy 和真实耗时。这些只有真卡能暴露。

不过这个 kernel 的同步结构是静态可见的（每轮两个同步，装载段和累加段之间没有交叉写），所以风险主要落在性能上，而性能本来就不由本地测。

---

## 9. 结果判读

### 9.1 `test_gemm` 退出码

| 退出码 | 含义 |
| --- | --- |
| 0 | 全部通过 |
| 1 | 有 FAIL（真错） |
| **2** | **没有显卡，跳过（不算失败）** |
| 3 | kernel 运行期出错 |

### 9.2 正常期望输出

```text
GPU              : <显卡型号>
SM count         : <SM 数量>
global mem       : <显存> MiB
shared / block   : <每 block 静态 shared 上限> B
warmup=10 repeat=50

M=1    K=1    N=1    PASS  最大误差=...  占容差=...  中位耗时=      ... us     ... GFLOP/s
M=2    K=3    N=4    PASS  ...
M=15   K=16   N=17   PASS  ...
...
M=512  K=512  N=512  PASS  ...
全部通过 ✔
512^3 占容差比例 = ... （<=1 即合规）
```

计时方式：预热 10 次 → CUDA event 包裹 `solve()` 重复 50 次 → 取**中位数**（转为微秒），与平台评测口径一致。

### 9.3 「16 行 FAIL」是没有显卡的指纹

早期版本的测试程序没加前置检查，在虚拟机里跑出来的输出是 16 行 `FAIL`、`占容差` 全在 400~1000 之间、`中位耗时 0.00 us`、`inf GFLOP/s`。看着像全盘算错，其实不是。

原因：没有显卡时 `cudaMalloc` / `cudaMemcpy` / kernel 启动**全部静默失败**（返回了错误码但没人检查），结果缓冲区 `hC` 保持初始值全 0，于是每个元素的误差就是 `|0 - e|`。代入题目的容差公式：

```text
占容差 = |0 - e| / (1e-3 + 1e-3·|e|) = 1000·|e| / (1 + |e|)
```

这个式子的上限正好是 1000。拿两组实测数据验算：

| 用例 | 最大误差 \|e\| | 公式 | 实际输出 |
| --- | --- | --- | --- |
| `1×1×1` | 0.6705 | 1000×0.6705/1.6705 = 401.4 | 4.014e+02 |
| `512³` | 33.94 | 1000×33.94/34.94 = 971.4 | 9.714e+02 |

精确到小数点后一位都对得上。如果是真的算错，误差会是随机的、分散的，不可能整齐地挤在 1000 附近。

**这组数字本身就是「没有显卡」的指纹。** 所以后来给 `test_gemm.cu` 加了 `cudaGetDeviceCount` 前置检查，无卡时直接 `[SKIP]` 并返回 2，不再刷 16 行假 FAIL。

---

## 10. 这套环境能证明什么、不能证明什么

| 项目 | 能验证吗 | 说明 |
| --- | --- | --- |
| 语法、类型、头文件 | ✅ 完全能 | `nvcc -c` 过 = 平台也编得过 |
| `solve` 符号名与签名 | ✅ 完全能 | 链接过就等于平台能找到入口 |
| shared 用量、寄存器溢出 | ✅ 完全能 | `-Xptxas -v` 是编译期信息，与是否有卡无关 |
| 算法与边界正确性 | ✅ 能（等价验证） | Python 复刻 80 组逐位对拍 |
| 数据竞争、同步漏写 | ❌ 不能 | 复刻只能模拟「按语义执行」，测不出硬件乱序 |
| bank 冲突、occupancy | ❌ 不能 | 必须真卡 |
| 中位耗时 / GFLOP/s | ❌ 不能 | 由平台返回的 `performanceMedianUs` 为准 |

所以本地测试的定位是：**交卷前把「编译不过」和「逻辑写错」两类错误清零**，性能交给平台。虚拟机给的是「交上去不会被编译错误和符号错误卡住」这个保证；耗时和真机行为，必须由平台或真卡给出。

---

## 11. 报错对照表

| 报错 / 现象 | 真实原因 | 处理 |
| --- | --- | --- |
| `No space left on device` | `/` 只有 9.8 G 且已满 | 回 4.4 扩容，再 `dpkg --configure -a` |
| `dpkg: error processing ... (--unpack)` | 半装状态 | `sudo dpkg --configure -a` + `apt-get -f install` |
| `cuda_runtime.h: No such file or directory` | 头文件包没装上（`nvidia-cuda-dev` 与 toolkit 是分开的） | `apt install nvidia-cuda-dev`，用 `dpkg -L` 确认 `/usr/include/cuda_runtime.h` |
| `unsupported GNU version! gcc 13 is not supported` | gcc 13.3 高于 CUDA 12.0 的支持范围 | 加 `-allow-unsupported-compiler`，或 `-ccbin g++-12` |
| `nvcc fatal: Unsupported gpu architecture 'sm_120'` / `'compute_120'` | CUDA 版本低于显卡要求 | 升级 Toolkit 到 12.8+；或在无卡机上改用 `-arch=sm_75` 只做资源检查；或 `-arch=sm_90` 走 JIT |
| `no CUDA-capable device is detected` | 无显卡 / 驱动未装 / 虚拟机没有直通 | **正常结果**（虚拟机），改做编译 + Python 对拍；真机则补驱动 |
| `undefined reference to 'solve(...)'` | 链接时漏了实现文件，或误把 `extern "C"` 去掉 | 编译命令里同时给 `test_gemm.cu gemm_opt.cu` |
| `nvcc: command not found` | PATH 没配 | `export PATH=/usr/local/cuda-12.0/bin:$PATH` |
| 运行输出全是 FAIL、占容差接近 1000 | 无卡导致结果缓冲区全 0 | 不是代码错，见 9.3 的公式验算 |
| 某个尺寸 `FAIL`，其余 PASS | 边界处理出错 | 先看是否集中在不整除 tile 的用例 |
| 可执行文件名冲突 | Windows 上 `test_gemm` 与目录同名 | 输出改名为 `bench.exe` |

---

## 12. 完整命令清单

### 12.1 无卡路径（路线 A + B）

```bash
# ① 体检：确认有没有显卡
lspci | grep -iE 'nvidia|vga|display'; ls /dev/nvidia*; nvidia-smi

# ② 扩容：卷组里的空闲空间给逻辑卷
sudo lvextend -l +100%FREE /dev/ubuntu-vg/ubuntu-lv
sudo resize2fs /dev/ubuntu-vg/ubuntu-lv
df -h /

# ③ 清理半装状态
sudo dpkg --configure -a
sudo apt-get -f install

# ④ 装编译链
sudo apt update
sudo apt install -y build-essential
sudo apt install -y nvidia-cuda-toolkit
sudo apt install -y --no-install-recommends nvidia-cuda-dev

# ⑤ 确认工具链
nvcc --version | tail -2
gcc --version | head -1
dpkg -L nvidia-cuda-dev | grep cuda_runtime.h

# ⑥ 语法检查（四个基础关 + 最终关）
cd ~/cuda
for f in hello_cuda vector_add matrix_add tiled_matmul gemm_opt; do
    nvcc -O3 -allow-unsupported-compiler -c $f.cu -o /dev/null && echo "$f OK"
done

# ⑦ 链接检查（验证 solve 符号）
nvcc -O3 -allow-unsupported-compiler -o test_gemm test_gemm.cu gemm_opt.cu

# ⑧ 资源用量
nvcc -O3 -allow-unsupported-compiler -arch=sm_75 -Xptxas -v -c gemm_opt.cu -o /dev/null

# ⑨ 运行（无卡应为 [SKIP] + 退出码 2）
./test_gemm; echo "退出码=$?"

# ⑩ 逻辑对拍（80 组）
python3 verify_gemm.py
```

> ⑥–⑩ 如果不想手敲，直接跑 `bash CUDA_VM一键部署.sh`（4.9），一次做完。

### 12.2 有卡路径（路线 C）

```bash
# ① 确认显卡与 Toolkit
nvidia-smi
nvcc --version
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# ② 用实际架构编译（以 sm_120 为例，按 5.3 替换）
nvcc -O3 -arch=sm_120 -o test_gemm test_gemm.cu gemm_opt.cu

# ③ 查看用量
nvcc -O3 -arch=sm_120 -Xptxas -v -c gemm_opt.cu -o /dev/null

# ④ 端到端 16 组 + 中位耗时
./test_gemm; echo "退出码=$?"
```

或直接整份粘贴 `CUDA真机验证_一键粘贴.sh`（5.2）。

---

## 13. 交卷前自检

- [ ] `nvcc -c gemm_opt.cu` 无 error、无 warning
- [ ] 链接 `test_gemm` 成功（`solve` 符号对得上）——**这一项错了直接零分**
- [ ] `-Xptxas -v`：`smem = 8256 B`，无 `spill`
- [ ] `verify_gemm.py` 输出与黄金文件逐字相同，80/80
- [ ] 有卡的话：16 行 `PASS`，`512^3 占容差 ≤ 1`，记录 `performanceMedianUs`
- [ ] `sha256sum -c MANIFEST.txt` 四项 OK
- [ ] 只提交 `gemm_opt.cu` 与 `report.md`，不带 `test_*.cu` / `verify_gemm.py`

---

## 附录 A：一键体检脚本

整段粘贴，一次性确认环境是否可用：

```bash
#!/usr/bin/env bash
echo "===== 1. 系统 ====="
grep -E '^(NAME|VERSION)=' /etc/os-release

echo "===== 2. 磁盘 ====="
df -h / | tail -1

echo "===== 3. 显示设备与驱动 ====="
lspci 2>/dev/null | grep -iE 'nvidia|vga|display' || echo "缺 pciutils"
ls /dev/nvidia* 2>/dev/null || echo "无 /dev/nvidia*"
nvidia-smi 2>&1 | head -3

echo "===== 4. 编译链 ====="
command -v nvcc gcc make || true
nvcc --version 2>/dev/null | tail -2
gcc --version 2>/dev/null | head -1
dpkg -L nvidia-cuda-dev 2>/dev/null | grep cuda_runtime.h || echo "缺 cuda_runtime.h"

echo "===== 5. 一句话结论 ====="
if nvidia-smi >/dev/null 2>&1; then
    echo "有 GPU：可以真跑与测速"
else
    echo "无 GPU：只做编译验证 + python3 verify_gemm.py"
fi
```

---

## 附录 B：文件指纹

`复现包/MANIFEST.txt` 内容（`sha256sum -c` 直接可用）：

```text
655c206b21109a38239f0b2290885a3d2c1cf93cd83f7ac09317e821447aeb94  gemm_opt.cu
17277930374b4a1c1b84781c8fe0239c2e3f8c906f57460510b4e9f0df64d4af  test_gemm.cu
e74ea047feda713e88d236626e3529632cde0ff3eac4aa0f94b84a7bc00e41f8  verify_gemm.py
a213bc8fbd1b2a41c8e60480e26828ea08f81ca0d2fa8aa1d547d9536c67dd9d  expected_verify_output.txt
```

---

## 附录 C：与报告 §1 环境表格的对应

填 `report.md` 第 1 节时可直接用下表：

| 项 | 值 |
| --- | --- |
| GPU | `<复现环境显卡型号>`；开发机为 RTX 5060 Laptop（`sm_120`，8 GiB，26 SM，驱动 591.91） |
| 显存 | `<例如 8 GiB>` |
| CUDA 版本 | `<nvcc --version 输出的 Release 号>`；虚拟机内为 12.0 |
| 测试尺寸 (M,K,N) | `512×512×512`（主测）、`256³`、`128³`；另加边界尺寸 `1×1×1`、`17×15×16`、`33×511×17`、`255×32×64`、`127×129×131` 等 |
| 数据范围 | 输入 A、B 均在 `[-1, 1]`（符合题面） |
| 编译选项 | `nvcc -O3 -arch=native`；无卡虚拟机为 `nvcc -O3 -allow-unsupported-compiler -arch=sm_75` |

> 开发机的独显为 Optimus 架构，无法直通虚拟机，因此端到端 GPU 实测请在复现环境中完成。本地能给出的确定性证据是 Python 复刻的 80 组逐位对拍。

---

## 附：原文件归属

合并前这些内容分散在三处，本文件是它们的唯一来源：

| 原文件 | 内容并入 |
| --- | --- |
| `虚拟机环境配置说明.md` | 第 1、4、8、9、10、12.1、附录 A |
| `环境与复现说明.md` | 第 2、3.3、6、7、9.2、11、附录 B、附录 C |
| `复现包/README.md` | 第 3 章、5.3 |
| `CUDA_VM一键部署.sh` | 4.9 |
| `CUDA真机验证_一键粘贴.sh` | 5.2、12.2 |

三个脚本本身仍保留在仓库中，它们是**工具**，不是说明；本文件负责讲清它们各自在什么场景下用、会做什么。

`复现包/gemm_repro.zip` 内部仍保留一份精简的 `README.md`，因为压缩包要能脱离本文单独分发、自己解释自己。它与本文第 3 章内容一致，**以本文为准**。
