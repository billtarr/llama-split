# lib/ — 运行时依赖，12 个文件 / 88 MB

这些是 `llama-split` 跑起来需要的全部动态库，从一个能跑的设备上拉下来的。
**换机器时直接 `adb push lib/*.so` 即可，不需要重新编译 llama.cpp。**

来源：`/data/local/tmp/llama.cpp/lib/`（由带 `patches/llama-layer-split.patch` 的 llama.cpp 源码树 @ `7d7f028` 构建）。

## 清单

| 文件 | 大小 | 是什么 |
|---|---|---|
| `libllama.so` | 41.9 MB | llama.cpp 核心（含全部模型实现） |
| `libggml.so` | 0.4 MB | ggml 后端注册表 |
| `libggml-base.so` | 8.2 MB | ggml 张量/图/调度 |
| `libggml-cpu.so` | 4.3 MB | CPU 后端（tokenize、未上 NPU 的算子、lm_head 在 CPU 时都靠它） |
| `libggml-hexagon.so` | 9.5 MB | **NPU 后端**：host 侧驱动，通过 fastrpc 把 applet 推进 CDSP |
| `libggml-opencl.so` | 15.9 MB | Adreno 后端。**不用它也必须带**，见下 |
| `libggml-rpc.so` | 6.9 MB | RPC 后端。同上 |
| `libggml-htp-v73.so` | 0.9 MB | DSP applet，SM8550（本机在用的这个） |
| `libggml-htp-v75.so` | 0.9 MB | DSP applet，SM8650 / 8 Gen 3 |
| `libggml-htp-v79.so` | 1.0 MB | DSP applet |
| `libggml-htp-v81.so` | 1.0 MB | DSP applet |
| `libggml-htp-v85.so` | 1.0 MB | DSP applet |

## 为什么 opencl / rpc 必须带（哪怕一个都不用）

`libllama.so` 和 `libggml.so` 的 **DT_NEEDED 里硬写了** `libggml-opencl.so`、`libggml-rpc.so`：

```
libllama.so -> libggml.so, libggml-cpu.so, libggml-rpc.so,
               libggml-opencl.so, libggml-hexagon.so, libggml-base.so, ...
```

它们在 DT_NEEDED 里，不是运行时才 dlopen 的可选后端。缺任何一个，动态链接阶段就失败，
`llama-split` 起不来——跟跑不跑 OpenCL 无关。要甩掉它们得用 `-DGGML_OPENCL=OFF -DGGML_RPC=OFF`
重编 `libggml.so`，省 23 MB，不值得。

## 已验证（2026-11-10）

| 检查 | 结果 |
|---|---|
| 依赖闭包（静态解析 DT_NEEDED） | `closure complete` |
| 只从这个目录解析依赖跑 `llama-split` | 正常打印 usage（缺任一 .so 会在链接阶段就失败） |
| 与本机正在跑的 9B 三段链的库对比 md5 | **8/8 SAME** |

第三条是它的真实价值：这 12 个文件和设备上正在跑的那套**字节相同**，所以「能跑」不是推断。

## 不能带、必须由 ROM 提供的

| 库 | 位置 | 说明 |
|---|---|---|
| `libc.so` `libdl.so` `libm.so` `liblog.so` | `/system/lib64` | bionic |
| `libc++_shared.so` | `/system_ext/lib64` | 本机实测从这里加载的 |
| `libOpenCL.so` | `/vendor/lib64` | 厂商 Adreno 的 OpenCL loader，**不能跨设备拷** |
| `libcdsprpc.so` | `/vendor/lib64` | fastrpc 的 CDSP 用户态驱动，NPU 通路靠它 |
| `libQnnHtp*.so` | `/vendor/lib64` | 存在，但**这条链不加载**（见 `L-062`） |

## 怎么用

```sh
ADB=${ADB:-adb}
"$ADB" shell "mkdir -p /data/local/tmp/llama.cpp/lib"
"$ADB" push lib/*.so /data/local/tmp/llama.cpp/lib/
```

applet 必须和 host 库在**同一个目录**（`ADSP_LIBRARY_PATH` 指向它），否则 fastrpc 加载 applet 时报 `0x80000406`。

## 怎么校验闭包没缺

换过库之后，选哪个 applet 由设备自动决定，但「缺库」要能在上设备前发现：

```sh
python - <<'EOF'
import struct, glob, os
def needed(p):
    d=open(p,'rb').read(); off,=struct.unpack_from("<Q",d,0x28)
    es,n,sn=struct.unpack_from("<HHH",d,0x3A)
    if off==0 or sn>=n: return None
    sh=[struct.unpack_from("<IIQQQQIIQQ",d,off+i*es) for i in range(n)]
    base=sh[sn][4]
    nm=lambda x: d[base+x:d.index(b"\0",base+x)].decode()
    dyn=[s for s in sh if nm(s[0])==".dynamic"]
    if not dyn: return None
    ds=[s for s in sh if nm(s[0])==".dynstr"][0][4]; out=[]
    for i in range(dyn[0][5]//16):
        t,v=struct.unpack_from("<qQ",d,dyn[0][4]+i*16)
        if t==0: break
        if t==1: out.append(d[ds+v:d.index(b"\0",ds+v)].decode())
    return out
have={os.path.basename(p) for p in glob.glob("lib/*.so")}
SYS={"libc.so","libdl.so","libm.so","liblog.so","libc++_shared.so","libOpenCL.so","libcdsprpc.so"}
bad=[(os.path.basename(p),x) for p in glob.glob("lib/*.so") for x in (needed(p) or [])
     if x not in have and x not in SYS]
print("closure complete" if not bad else bad)
EOF
```

期望输出：`closure complete`。5 个 applet 没有 `.dynamic` 段，是正常的（它们由 fastrpc 加载，不是 ld 加载）。

## 更新（llama.cpp 重新构建后）

```sh
ADB=${ADB:-adb}; export MSYS_NO_PATHCONV=1
for f in libllama libggml libggml-base libggml-cpu libggml-hexagon \
         libggml-opencl libggml-rpc libggml-htp-v73 libggml-htp-v75 \
         libggml-htp-v79 libggml-htp-v81 libggml-htp-v85; do
  "$ADB" pull "/data/local/tmp/llama.cpp/lib/$f.so" lib/
done
```

**host 库和 applet 必须来自同一次构建**。混用会崩，而且报错往往指向别处（`fastrpc_mmap failed` 或数值全错）。
