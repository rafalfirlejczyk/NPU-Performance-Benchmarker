# NPU-Performance-Benchmarker
# Computer Vision on Mobile Device — Part 2: Eight Years of Evolution

*From Image Classification on SD660 to Real-Time Object Detection on QCS6690/QCS6490*

**Summary: From 11x speedups in 2018 to real-time 4.5ms inference in 2026, this research documents the transformation of mobile computer vision through the lens of industrial Snapdragon hardware. While YOLO11 on the Hexagon HTP delivers a breathtaking 50x performance leap over legacy CPU paths, the journey remains a high-effort craft defined by fragmented SDKs, SELinux roadblocks, and the persistent 'NNAPI mirage' on enterprise firmware.**

---

**Nov 2018.** I ran Inception V3 on a Snapdragon SD660 and measured how fast a DSP can classify 1000 images compared to CPU and GPU. The conclusion: DSP is 11× faster than CPU. NNAPI was not yet supported. The extra effort was worth it — barely.

**May 2026.** I am running YOLO11n and YOLO11s on Honeywell CT70 and CT47 industrial scanners — Snapdragon QCS6690 and QCS6490, Hexagon 780 and 770 HTP — and detecting objects in real time. DSP inference: **4.5 ms**. That is one frame every 4.5 milliseconds for the model alone.

This article documents what changed, what did not, and what it costs in engineering effort to get there.

---

## The 2018 Baseline

The 2018 article (*[Computer Vision Application on Mobile Device](https://medium.com/@firlejczyk/computer-vision-application-on-mobile-device-392ad3a6b26a)*) used the `snpe-net-run` command-line tool included in the SNPE SDK. No custom app. No camera. Just 1000 JPEG files fed to an Inception V3 classification model. One metric: total time for 1000 images.

| Runtime | Total time (1000 images) | Per image |
|---|---|---|
| CPU | 975 s | 975 ms |
| GPU | 380 s | 380 ms |
| DSP | 86 s | **86 ms** |

DSP was 11× faster than CPU and 4.4× faster than GPU. Energy consumption followed the same ranking: DSP used the least, CPU used ~15× more. NNAPI (Android 8.1 Neural Networks API) was not supported on the SD660 at all.

Hardware: SD660, 14 nm, Hexagon 680 DSP, Adreno 512 GPU, 299×299 images, image classification only.

---

## What Changed in Eight Years

### Hardware

| | SD660 (2018) | QCS6690 (2026) | QCS6490 (2026) |
|---|---|---|---|
| Process | 14 nm | 5 nm | 6 nm |
| CPU | 4+4 × A53 | X1 + A78×3 + A55×4 | X1 + A78×3 + A55×4 |
| GPU | Adreno 512 | Adreno 643L | Adreno 643 |
| DSP/HTP | Hexagon 680 | Hexagon 780 HTP | Hexagon 770 HTP |
| HMX units | None | 4 × HMX (matrix) | 4 × HMX |
| INT8 TOPS | ~0.5 | ~12.3 | ~10.8 |

The Hexagon 780 and 770 are a different class of hardware compared to the 680. The HMX (Hexagon Matrix eXtensions) units perform dense INT8 matrix multiplications — the exact operation YOLO inference is made of. The 680 was a scalar/vector DSP doing the same work in software.

### Software

In 2018 the stack was simple: PyTorch/TensorFlow → DLC conversion → `snpe-net-run`. One tool, one format, one decision.

In 2026 the stack has branched significantly:

- **TFLite path**: CPU (FP32/FP16/INT8/integer_quant) + GPU delegate + NNAPI delegate
- **SNPE/QNN path**: DLC quantization → HTP compilation → custom Java/Kotlin app
- **QNN context binary path**: quantized DLC → `qnn-context-binary-generator` → `.bin` → JNI C++ app

Each branch has its own export parameters, quantization quirks, and failure modes. This is not simpler than 2018. It is significantly more complex.

### The Task

2018: image classification (Inception V3, 1000 classes, 299×299, static JPEG files).

2026: real-time object detection (YOLO11n/s original pretrained model on Coco80 and custom model YOLO11n/s model of damaged boxes pretrained on 1700 images of box with 4 classes, 640×640, live camera at 30fps).

These are not directly comparable tasks — detection is harder, the model is larger, the resolution is higher, and the measurement is model-only inference time per frame rather than total batch time. The comparison is still valid as a measure of progress: can the hardware run modern detection models in real time?

---

## The Devices

Two Honeywell industrial Android mobile computers, "scanners" — devices you would find in enterprise, factory, store or warehouse, not in a consumer phone store:

**CT70** — QCS6690 (Snapdragon 778G industrial variant), Hexagon 780 HTP, Adreno 643L, Android 15. Built for rugged scanning environments, not for ML benchmarks, but I wanted to check it how far could it go.

**CT47** — QCS6490 (Snapdragon 695 industrial variant), Hexagon 770 HTP, Adreno 643, Android 15. Rugged, slightly older silicon, slightly lower TDP, less expensive, will it perform in CV project?

Both devices run Honeywell industrial firmware, which introduces constraints not present on consumer devices — particularly around DSP access and NNAPI.

For reference, two additional consumer devices were measured:

**Samsung S21** — SM8450 (Snapdragon 8 Gen 1), Hexagon 780 HTP (same generation as CT70 QCS6690). Consumer flagship, higher sustained clock budget.

**OPPO6** — SN870 Octa-core, older Hexagon generation.

---

## The Models

Two YOLO11 variants were tested:

**YOLO11n** (nano) — 2.6M parameters, 6.5 GFLOPs. The smallest YOLO11 variant.

**YOLO11s** (small) — 9.4M parameters, 21.5 GFLOPs. 3.3× more FLOPs than YOLO11n.

Two datasets:

**COCO80** — standard 80-class COCO dataset used to train detection model, publicly available weights from Ultralytics.

**box-4class** — a custom box damage dataset (1700 images only) with 4 classes (2 real + 2 dummy to satisfy the GPU output channel rule described below). Smaller output tensor: 8 channels (4 coords + 4 classes) vs 84 channels (4 coords + 80 classes).

All models were exported from Ultralytics `.pt` weights. Custom model were "retrained" using box-4class dataset for 100 Epochs.

---

## The Results

All inference times are **model-only** (no pre-processing, no post-processing, no NMS). Total pipeline time (including resize, pixel extraction, NMS) is approximately 30–45 ms higher and is shown separately in the presentation but not the focus of this comparison.

### Complete measurement results (total pipeline[ms] ; model-only inference time[ms])

All times shown as **total pipeline ; model-only** in milliseconds. Total includes resize, pixel packing, NMS and all Kotlin/Java overhead. Model-only is pure DSP/GPU/CPU execution time. Measured after Android Studio ByteBuffer reuse and ImageAnalysis sizing optimizations.

#### YOLO11n — all devices, all accelerators

| Device | CPU FP32 | CPU iquant | GPU FP16 | GPU INT8 | NPU iquant | QNN DSP |
|---|---|---|---|---|---|---|
| CT70 QCS6690 | 220; **200** | 120; **70** | 95; 45 | 95; **45** | 140; **91** | 22; **6** |
| CT47 QCS6490 | 265; **245** | 120; **92** | 78; 44 | 80; **45** | 141; **115** | 24; **4.8** |
| S21 SM8450 | 165; **~200** | 75; **55** | 70; 30 | 80; **21** | 120; 100 | **19**; **2.9** |
| OPPO SM8250 | 275; **250** | 118; **85** | 75; **36** | 80; **39** | 160; 130 | 60; **40** |

#### YOLO11s — all devices, QNN DSP (coco80 ; box-4class)

| Device | CPU FP32 | CPU iquant | GPU FP16 | QNN DSP coco | QNN DSP box |
|---|---|---|---|---|---|
| CT70 QCS6690 | 605; **585** | 190; **170** | 155; **105** | 26; **12** | 26; **12** |
| CT47 QCS6490 | 725; **702** | 225; **200** | 140; **103** | 25; **9** | 22; **9** |
| S21 SM8450 | 545; **522** | 165; **145** | 160; **129** | 20; **3.5** | 32; **3.5** |
| OPPO SM8250 | 650; **630** | 210; **185** | 115; **80** | 82; **59** | 98; **57** |

**Pipeline overhead** (total − model) for QNN DSP path: consistently 16–20ms across all HTP devices regardless of model size. This is the irreducible cost of FastRPC channel negotiation, pixel packing in Kotlin, and NMS post-processing. A 2.9ms model does not produce 2.9ms on-screen latency — it produces 19ms.

**Record:** S21 YOLO11n box-4class QNN DSP = **2.9ms model, 19ms total**.

### Key observations from the full dataset

**DSP speedup vs CPU has grown since 2018.** Comparing model-only times  (apples-to-apples):

| Year | Device | Model | CPU iquant | DSP | Speedup |
|---|---|---|---|---|---|
| 2018 | SD660 Hex680 | Inception V3 | 975ms | 86ms | **11×** |
| 2026 | CT70 QCS6690 | YOLO11n | 70ms | 6ms | **12×** |
| 2026 | CT70 QCS6690 | YOLO11n | 170ms | 12ms | **14×** |
| 2026 | CT47 QCS6490 | YOLO11n | 92ms | 4.8ms | **19×** |
| 2026 | CT47 QCS6490 | YOLO11s | 200ms | 9ms | **22×** |
| 2026 | S21 SM8450 | YOLO11n | 60ms | 2.9ms | **21×** |
| 2026 | S21 SM8450 | YOLO11s | 145ms | 3.5ms | **41×** |
| 2026 | OPPO SM8250 | YOLO11n | 85ms | 40ms | **2×** |
| 2026 | OPPO SM8250 | YOLO11s | 185ms | 59ms | **3×** |

The OPPO result — only 2×-3× faster on DSP than CPU — is the pre-HTP baseline. Hexagon 698 (HVX) has no HMX matrix units. The speedup increse for YOLO11s models and specially 41× speedup on S21 reflects YOLO11s's compute profile: it is nearly compute-bound on the Hexagon 780 HTP, where every FLOP maps directly to an HMX instruction. The bottleneck starts to be not the LPDDR4X RAM and data delivery but 256 HMX compute lanes.

**NNAPI confirmed slower than CPU on every device.** NPU (NNAPI) times exceed CPU iquant times by 15–45ms on all four devices. This is NNAPI delegation overhead added on top of a CPU fallback — paying both costs simultaneously. The underlying routing never reaches the HTP.

**GPU INT8 (uncalibrated) is the fastest GPU path.** YOLO11n model-only: CT70 45ms, CT47 45ms, S21 21ms, OPPO 39ms — consistently 10–20% faster than GPU FP16. The accuracy trade-off (no calibration data) was not validated on domain images.

**GPU INT8 (calibrated) is falling back to CPU.** Fully quantized INT8 model, after calibrated quantization will be rejected by the GPU. 


The DSP advantage grew from 11× to 26–41×. This is partly hardware (HMX vs scalar DSP), partly model (YOLO11n is more matrix-multiply-heavy than Inception V3's depthwise separable convolutions which are less HMX-friendly), and partly software (SNPE HTP optimization pipeline improved significantly over 8 years).

---

## The Engineering Effort: What Nobody Writes About

The numbers above look clean. The path to get them was not.

### Error 73 — DSP access blocked on industrial firmware

The first DSP attempt produced Error 73: `BAD_HANDLE`. Logcat showed:

```
avc: denied { read } for name="sku" dev="sysfs" ... tclass=file permissive=0
```

SNPE needs to read the Snapdragon SKU from sysfs to select the correct DSP routing. On CT70/CT47, SELinux in enforcing mode denies this. The symptom is silent CPU fallback — inference runs at CPU speed with no error message at the app level. Basically, you can call it **silent performance killer**. The fix: call `setUnsignedPD(true)` and set `ADSP_LIBRARY_PATH` with semicolon separators before instantiating the neural network builder. The exact sequence matters. This took approximately three days to diagnose.

### NNAPI — 500ms, always CPU

NNAPI times on CT70 and CT47 match CPU times exactly. The NNAPI delegate initialises, accepts the model, and then quietly executes on CPU. This is confirmed by timing: 275 ms for YOLO11n NNAPI matches 275 ms for float32 CPU on CT47. Industrial firmware does not expose the HTP to NNAPI. This was true in 2018 on SD660 and remains true in 2026.

### SDK version locking

The SNPE/QNN stack requires three components at exactly the same version: the `.aar` Java library, the `.so` native libraries, and the `.dlc` model generated by `snpe-dlc-quantize`. One component at a different version produces opaque errors. Diagnosing this mismatch — between the `.aar` version in `build.gradle` and the `.so` libraries in `jniLibs` — took approximately one week because the errors do not mention version numbers. The fix: ensure `snpe-release.aar`, all `.so` files, and the SDK tools that produced the DLC all come from the same QAIRT release folder.

### GPU INT8 — the calibration trade-off

GPU inference works well on both devices — 42–55 ms for YOLO11n, significantly faster than CPU INT8. But GPU INT8 has an unintuitive constraint.

When exporting with `int8=True, data=dataBox.yaml` in Ultralytics (calibrated quantization), the I/O tensors are quantized to INT8. The TFLite GPU delegate requires FP32 I/O tensors and rejects the model, falling back to CPU. The symptom is a runtime speed matching CPU, not GPU.

The solution: export with `int8=True` without the `data=` parameter. This preserves FP32 I/O while quantizing internal layers, and the GPU delegate accepts it. Accuracy on domain images was not fully validated with this approach.

For the custom box model, `half=True` (FP16 export) gives GPU ~42 ms with FP32 I/O and no calibration compromise.

### GPU class count rule

The GPU delegate on these devices requires that the output channel count satisfies `(num_classes + 4) % 4 == 0`. A 2-class model produces 6 output channels (`6 % 4 ≠ 0`) and is rejected. The fix: add 2 dummy classes to reach 4 total, giving 8 channels (`8 % 4 == 0`). This rule is not documented anywhere in the Ultralytics or Qualcomm documentation. It was discovered by comparing working and failing export configurations.

### DLC variants: three identical outputs

Three DLC variants were generated and measured:
1. `snpe-dlc-quantize` output (base quantized DLC)
2. After `snpe-dlc-graph-prepare --htp_socs qcs6690` (V73 optimised)
3. After `snpe-dlc-graph-prepare --htp_socs qcs6490` (V68 optimised)

Inference times were identical across all three variants on both devices. The `snpe-dlc-graph-prepare` step affects the first-call compilation time (~180 ms native device, ~4000 ms when running a cross-SoC DLC), not steady-state inference. In QAIRT 2.45, `--enable_init_cache` was removed — the behaviour is automatic when `--htp_socs` is specified. The DLC already contains an embedded pre-compiled Hexagon binary.

### 500 ms of FastRPC errors — completely normal

On CT47, every first DSP call produces approximately 500–600 ms of FastRPC negotiation errors in logcat. Lines like `[libadsprpc] ADSP REMOTE CHANNEL ... retry` appear repeatedly before the first successful inference. This is not a bug. It is the Hexagon remote procedure call subsystem establishing the CPU-DSP communication channel on a cold start. Subsequent inferences run at full speed. Do not attempt to fix it — it cannot be fixed by reinstalling the app.

### The QNN context binary (.bin) path — a complete investigation

SNPE's DLC path starts the model on first launch with an on-device compilation step. The QNN context binary (`.bin`) path pre-compiles the model on the development machine, promising near-zero first-launch latency.

**The correct pipeline in QAIRT 2.45:**

The original approach — `ONNX → qnn-onnx-converter → model.so → qnn-context-binary-generator` — is blocked for YOLO11. The `qnn-onnx-converter` fails on YOLO11's C2f backbone with a `getBroadcastedTensorShape` error. The `--model` flag also expects a compiled ELF `.so`, not a `.dlc`.

The correct flag in QAIRT 2.36+ is `--dlc_path`:

```bash
# Step 1: prepare DLC for specific SoC
snpe-dlc-graph-prepare \
    --input_dlc  yolo11s_quantized.dlc \
    --output_dlc yolo11s_v73.dlc \
    --htp_socs   qcs6690          # not v73, not 0x06C6 — product name only

# Step 2: generate context binary
qnn-context-binary-generator \
    --backend  $SNPE_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
    --dlc_path yolo11s_v73.dlc \
    --binary_file yolo11s_v73     # no .bin extension — tool appends it
```

Three flag mistakes that are not documented anywhere and took significant time to find: `--model` vs `--dlc_path`, the SoC name format (`qcs6690` not `v73` or `0x06C6`), and the double-extension trap (`--binary_file foo.bin` produces `foo.bin.bin`).

**On the Android side:**

`QnnSystemContext_create` and related symbols used to extract tensor metadata were renamed in QAIRT 2.45 — `dlsym()` returns null at runtime. The fix: bypass `QnnSystemContext` entirely and hardcode the tensor dimensions confirmed by `qnn-context-binary-utility --json_file`. For YOLO11s COCO: input `[1,3,640,640]` uint8 NCHW (not NHWC like the DLC path), output `[1,84,8400]` uint8.

`QNN_HTP_CONTEXT_CONFIG_OPTION_VTCM_SIZE` was also removed in QAIRT 2.45. The only available VTCM-related option is `QNN_HTP_CONTEXT_CONFIG_OPTION_SKIP_VALIDATION_ON_BINARY_SECTION`, which bypasses signature validation but not DSP firmware memory allocation.

**Results by device:**

| Device | Result | Error | Root cause |
|---|---|---|---|
| CT70 QCS6690 Hex780 | ✗ Fails | 0x138d vtcm unsupported | Restriction for unsigned PD VTCM below 4MB |
| CT47 QCS6490 Hex770 | ✗ Fails | 0x138d vtcm unsupported | Same firmware restriction |
| S21 SM8450 Hex780 | ✓ Works | Quality??? | Consumer firmware, no VTCM restriction |
| OPPO6 SN870 Hex698 | ✗ N/A | — | No HTP silicon — HVX only |

YOLO11s requires exactly 4MB VTCM on Hexagon HTP. `--vtcm_override 1` and `--vtcm_override 2` in `snpe-dlc-graph-prepare` are silently ignored — the model's minimum cannot be reduced below what the graph optimizer determines it needs. Both YOLO11n and YOLO11s produce `vtcmSize: 4` regardless of the override value.

The DLC path works on Honeywell devices because SNPE's on-device compiler negotiates VTCM with the actual firmware at compilation time. The `.bin` path has no equivalent negotiation — it bakes in the theoretical maximum and the device rejects it.

**On Samsung S21, the `.bin` path works, but class confidence values are too low to match SNPE quality.** Context creation succeeds, graph is retrieved, inference runs. Init time on S21: ~17ms for binary loading, ~165ms total including FastRPC DSP channel setup. The FastRPC setup is unavoidable on both paths — it is not a compilation step. The `.bin` advantage over the DLC path only exists when using a raw uncompiled DLC (skipping `snpe-dlc-graph-prepare`); in that case the DLC path takes ~4000ms while the `.bin` path still takes ~180ms.

**Conclusion:** For Honeywell industrial devices, use the DLC path. For consumer Snapdragon devices, the .bin path may work and will eliminates the need for on-device compilation. Investigation on class detection has to be done to get acceptable detection quality.

---

## Model Efficiency: YOLO11n vs YOLO11s

YOLO11s has 3.3× more FLOPs than YOLO11n (21.5 vs 6.5 GFLOPs). On DSP, the inference time ratio is much smaller:

| Device | 11n | 11s | Ratio |
|---|---|---|---|
| CT47 DSP | 4.5 ms | 8 ms | **1.9×** |
| CT70 DSP | 5 ms | 10 ms | **2.0×** |

The ~1.9×–2× measured speedup vs 3.3× expected from FLOPs indicates the DSP is bandwidth-limited on YOLO11n: the HMX compute units are waiting for data from LPDDR4X memory. YOLO11s reaches closer to the compute bound. Both models run in real time at 30fps.

---

## Conclusion

The answer to the 2018 question — *"is the extra effort worth it?"* — is now unambiguous.

In 2018, DSP was 11× faster than CPU for a classification task on Hexagon 680. The effort was: install SNPE SDK, convert model to DLC, run `snpe-net-run`. That is roughly two days of work.

In 2026, DSP is 33–69× faster than CPU for real-time object detection on Hexagon 770/780. YOLO11n runs in **4.5 ms** on a QCS6490 industrial scanner. The effort was: eight weeks of debugging, SDK version matching, SELinux workarounds, GPU calibration root-cause analysis, and JNI layer implementation. That is not two days.

The hardware improved dramatically. The software complexity grew at roughly the same pace in the opposite direction.

What has not changed in eight years: NNAPI still does not reach the Hexagon DSP on Qualcomm industrial firmware. It falls back to CPU as it did in 2018. The dedicated SDK — now called QAIRT instead of SNPE — remains the only practical path.

What improved: the result. 4.5 ms per frame, real-time detection, a model far more capable than Inception V3, on a device that fits in a warehouse worker's hand. The extra effort is still required. So is the reward.


---

## The Video

*[Four Android devices (CT70, CT47, S21, Oppo6), four YOLO11s models (TFLite FP32, FP16, INT8, QNN DLC), four accelerators (CPU, GPU, NPU, HTP DSP). All in one video. Watch till the end.](https://www.youtube.com/watch?v=MQwPHHI9ICk)*

<img width="1900" height="964" alt="image" src="https://github.com/user-attachments/assets/2f2756d8-9cb9-448f-8cec-f344d84ee530" />

The screenshot from that video — all four devices detecting the same scene simultaneously — is the clearest summary of eight years of progress. CT47 Hexagon 770 at 31ms, CT70 Hexagon 780 at 20ms, OPPO Hexagon 698 at 85ms, Samsung S21 Hexagon 780 at 21ms. The OPPO result is the pre-HTP baseline: same SDK, four times slower, no HMX units.

---

## Future Outlook (2026–2030)

Eight years between articles is too long. Here is what needs to happen in the next four. Five wishes or five predictions which would help to develop computer vision solutions on the edge.

### Prediction 1: NPU-Native OS Buffers (my wish)

**•	The Premise:** Android's SurfaceFlinger still routes video frames through intermediate CPU-readable buffers before handing allocations off to hardware wrappers (such as SNPE delegates). This causes severe overhead (tensor conversions, pixel extraction loops, and heap-allocated ByteBuffers) before a single NPU hardware cycle ticks.

**•	The Vision:** The state of the art demands an integrated camera HAL that writes straight to typed tensor buffers residing inside Secure Enclave memory. This eliminates roughly 15–30ms of structural page-transfer latency.

**•	The Engineering Barrier:** A universal host-compiled context binary (.bin) fails on custom enterprise warehouse devices because of low-level VTCM (Vector Tightly Coupled Memory) page layout negotiations. The operating system must broker page tables between compilers and firmware-dependent kernels—which is why runtime frameworks like SNPE operate SKU-by-SKU today.


### Prediction 2: On-Device Vision LoRA Fine-Tuning

**•	The Premise:** YOLO11 detection heads are incredibly compact (consisting of less than a mil-lion total parameters in neural geometry). This represents an execution profile easily handled by gradient computation threads inside modern Hexagon accelerator blocks. This allows local training on a budget without cloud data transmission, similar to your decentralized RTX 2050 PC Lora training runs.

**•	The Vision:** An edge-deployed box-damage model that auto-updates itself overnight using localized on-device HTP backend backpropagation. This bridges the extreme domain gaps (lighting variation, packaging textures, camera lens distortions of individual industrial firmware SKUs) without manual labeling, keeping training weights and data completely local and safe.


### Prediction 3: The End of the Format War 

**•	The Premise:** Overcoming the cumbersome seven-stage conversion line (PyTorch, ONNX, TFLite, DLC files) with multiple target compilers that output diverging numerical met-rics is unsustainable.

**•	The Vision:** Pure, absolute standardization under a uniform execution block (such as ONNX Runtime's QNN Execution Provider). Standard SNPE/DLC manual conversions will eventual-ly be remembered like vendor-specific legacy OpenGL headers: necessary and revolutionary in their epoch, but eventually abstracted away by a unified runtime API


### Prediction 4: Context-Aware Self-Calibration on the Edge

**•	The Premise:** INT8 quantization currently relies on offline calibration image tensors inside a host developer machine, producing a static compromise that ignores the raw real-world con-text of individual deployments.

**•	The Vision:** Quantized networks that self-calibrate after initial deployment by leveraging the first 1,000 real-world frames parsed by the scanner to dynamically calculate scale coeffi-cients. Running localized post-training calibration arrays of 1,000 frames at stride-8 on the modern Hexagon HTP takes less than 10 seconds, eliminating manual, multi-warehouse da-taset calibration tasks.


### Prediction 5: Mandatory NNAPI NPU Routing Checkpoints

**•	The Premise:** Over eight years, NNAPI on industrial enterprise Qualcomm system stacks has been a mirage, defaulting to clean model compilation only to silently redirect execution pipeline graphs to the host CPU cores.

**•	The Vision:** Future Android CDD (Compatibility Definition Document) revisions should man-date that any SoC marketed with "artificial intelligence" silicon has a non-optional, CTS-verified hardware delegation path through NNAPI. Accelerated execution must stop being a marketing gimmick or a firmware mystery.


---

## Conclusion

In 2018 the question was: *is the extra effort worth it?* The answer was: barely, and only for specific workloads.

In 2026 the answer is: yes, unambiguously, if you accept the cost.

The hardware improved by roughly 20×. A real-time object detector with 80 classes and 6.5 GFLOPs runs in 4.5ms on a Hexagon 770 that fits in a "warehouse scanner". The same task would have taken 450ms on the hardware available in 2018.

The software complexity grew at the same pace. Seven-step conversion pipeline. Version-locked SDK triplets. SELinux workarounds. GPU output channel alignment rules. Calibration trade-offs that silently break GPU delegation. NNAPI that quietly routes to CPU. A four-week investigation into a context binary path that ultimately cannot match the quality of on-device compilation.

What has not changed in eight years: if you want the DSP, you use the dedicated SDK. NNAPI is still not the answer. The extra effort is still required.

What changed: the reward is now large enough to justify the effort for real production deployments. 4.5ms inference, 20ms total pipeline, real-time detection on an industrial scanner running Android 15. That is the state of the art in 2026, documented with measurements on four devices.

Part 3 will not take eight years.

---

## Appendix: Hardware and Software Reference

**Devices tested:**
- Honeywell CT70: QCS6690, Hexagon 780 HTP, Adreno 643L, Android 15, soc_id=658
- Honeywell CT47: QCS6490, Hexagon 770 HTP, Adreno 643, Android 15, soc_id=497
- Samsung Galaxy S21 (SM-S908x): SM8450, Hexagon 780 HTP, Android 14, soc_id=457
- OPPO (CPH2247): SN8250 (Snapdragon 870, kona), Hexagon 698 HVX, soc_id=356

**SDK:** Qualcomm QAIRT 2.45.40.260406 (formerly SNPE / QNN AI Engine Direct)

**Models:** Ultralytics YOLO11n, YOLO11s — TFLite and DLC variants

**Complete DLC pipeline (QAIRT 2.45):**
```bash
# COCO80 from TFLite:
snpe-tflite-to-dlc --input_network yolo11s_float32.tflite \
    --input_dim images "1,640,640,3" --output_path yolo11s.dlc

# Custom class counts (use ONNX — TFLite path fails for non-80-class):
snpe-onnx-to-dlc --input_network best.onnx --output_path best.dlc

# Quantize:
snpe-dlc-quantize --input_dlc yolo11s.dlc --input_list input_list.txt \
    --output_dlc yolo11s_quantized.dlc --enable_htp

# HTP compilation (--htp_socs uses product names, not Hexagon arch names):
snpe-dlc-graph-prepare --input_dlc yolo11s_quantized.dlc \
    --output_dlc yolo11s_final.dlc --htp_socs qcs6690

# Context binary (--dlc_path, NOT --model; no .bin extension in --binary_file):
qnn-context-binary-generator \
    --backend $SNPE_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
    --dlc_path yolo11s_final.dlc \
    --binary_file yolo11s_v73   # tool appends .bin automatically
```

**Key rules for TFLite GPU delegation:**
- Use `half=True` (FP16) — always FP32 I/O, no calibration compromise
- Never use `data=` with `int8=True` for GPU — INT8 I/O causes silent GPU rejection
- Set `rect=False` in all TFLite exports
- Output channels must satisfy `(num_classes + 4) % 4 == 0`

---

*Part 1 (2018): [Computer Vision Application on Mobile Device](https://medium.com/@firlejczyk/computer-vision-application-on-mobile-device-392ad3a6b26a)*

*Source code and WSL scripts: [GitHub repository link](https://github.com/rafalfirlejczyk/NPU-Performance-Benchmarker)*

*Tags: Qualcomm DSP, Hexagon HTP, YOLO11, Android, SNPE, QNN, Object Detection, Edge AI, Industrial AI, Honeywell CT70, CT47*
