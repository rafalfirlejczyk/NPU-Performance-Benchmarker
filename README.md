# NPU-Performance-Benchmarker
# Computer Vision on Mobile Device — Part 2: Eight Years of Evolution

*From Image Classification on SD660 to Real-Time Object Detection on QCS6690/QCS6490*

---

**Nov 2018.** I ran Inception V3 on a Snapdragon SD660 and measured how fast a DSP can classify 1000 images compared to CPU and GPU. The conclusion: DSP is 11× faster than CPU. NNAPI was not yet supported. The extra effort was worth it — barely.

**May 2026.** I am running YOLO11n on Honeywell CT70 and CT47 industrial scanners — Snapdragon QCS6690 and QCS6490, Hexagon 780 and 770 HTP — and detecting objects in real time. DSP inference: **4.5 ms**. That is one frame every 4.5 milliseconds for the model alone.

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

Two Honeywell industrial Android scanners — devices you would find in a warehouse, not a consumer phone store:

**CT70** — QCS6690 (Snapdragon 778G industrial variant), Hexagon 780 HTP, Adreno 643L, Android 15. Built for rugged scanning environments, not for ML benchmarks.

**CT47** — QCS6490 (Snapdragon 695 industrial variant), Hexagon 770 HTP, Adreno 643, Android 15. Slightly older silicon, slightly lower TDP.

Both devices run Honeywell industrial firmware, which introduces constraints not present on consumer devices — particularly around DSP access and NNAPI.

For reference, two additional consumer/semi-industrial devices were measured:

**Samsung S21** — SM8450 (Snapdragon 8 Gen 1), Hexagon 780 HTP (same generation as CT70 QCS6690). Consumer flagship, higher sustained clock budget.

**OPPO6** — SN870 Octa-core, older Hexagon generation.

---

## The Models

Two YOLO11 variants were tested:

**YOLO11n** (nano) — 2.6M parameters, 6.5 GFLOPs. The smallest YOLO11 variant.

**YOLO11s** (small) — 9.4M parameters, 21.5 GFLOPs. 3.3× more FLOPs than YOLO11n.

Two datasets:

**COCO80** — standard 80-class COCO detection model, publicly available weights from Ultralytics.

**box-4class** — a custom model trained on a proprietary box damage dataset with 4 classes (2 real + 2 dummy to satisfy the GPU output channel rule described below). Smaller output tensor: 8 channels (4 coords + 4 classes) vs 84 channels (4 coords + 80 classes).

All models were exported from Ultralytics `.pt` weights.

---

## The Results

All inference times are **model-only** (no pre-processing, no post-processing, no NMS). Total pipeline time (including resize, pixel extraction, NMS) is approximately 30–45 ms higher and is shown separately in the presentation but not the focus of this comparison.

### QNN DSP (Hexagon HTP via SNPE DLC)

| Device | SoC | YOLO11n COCO | YOLO11n box | YOLO11s COCO | YOLO11s box |
|---|---|---|---|---|---|
| CT47 | QCS6490 Hex770 | 4.7 ms | **4.5 ms** | 9 ms | 8 ms |
| CT70 | QCS6690 Hex780 | 6 ms | 5 ms | 10 ms | 10 ms |
| S21 | SM8450 Hex780 | 2.8 ms | **2.5 ms** | — | 3.5 ms |
| OPPO6 | unknown | 40 ms | 37 ms | 59 ms | 57 ms |

CT47 is faster than CT70 on DSP for YOLO11n (4.5 ms vs 5 ms). This is counterintuitive — QCS6690 has a newer Hexagon generation. The likely explanation is clock budget: CT47's industrial thermal envelope may sustain higher HTP clocks for the duration of a single inference burst. This was not confirmed by direct clock readout (`/sys/kernel/gpu/gpu_clock` equivalent for HTP) and remains an open question.

The S21 result (2.5 ms on Hexagon 780) confirms that the same Hexagon 780 silicon can run faster on a consumer device with a higher power budget.

The OPPO6 result (37–57 ms) reflects an older Hexagon generation without HMX units. It is included to show that the 4.5 ms CT47 result is not universal — it requires specific HTP silicon.

### TFLite — All Paths (CT70 YOLO11n)

| Export | CPU | GPU | NNAPI |
|---|---|---|---|
| float32 | 200 ms | 82 ms | 273 ms (CPU fallback) |
| float16 | 200 ms | **45 ms** | 275 ms (CPU fallback) |
| int8 (calibrated) | **110 ms** | 104 ms (CPU fallback*) | 165 ms |
| integer_quant | 70 ms | 55 ms | 91 ms |

*GPU fallback on calibrated int8 is explained below.

NNAPI times match CPU times exactly on CT70 and CT47. On both devices, NNAPI routes to CPU rather than the DSP. This is the same finding as 2018 — NNAPI still does not reach the Hexagon HTP on Honeywell industrial firmware, eight years later.

### DSP speedup vs CPU (2018 vs 2026)

| Year | Device | Model | CPU time | DSP time | Speedup |
|---|---|---|---|---|---|
| 2018 | SD660 Hex680 | Inception V3 | 975 ms | 86 ms | **11×** |
| 2026 | CT70 QCS6690 | YOLO11n | 200 ms | 6 ms | **33×** |
| 2026 | CT47 QCS6490 | YOLO11n | 120 ms | 4.7 ms | **52×** |
| 2026 | S21 SM8450 | YOLO11n | 195 ms | 2.8 ms | **69×** |

The DSP advantage grew from 11× to 33–69×. This is partly hardware (HMX vs scalar DSP), partly model (YOLO11n is more matrix-multiply-heavy than Inception V3's depthwise separable convolutions which are less HMX-friendly), and partly software (SNPE HTP optimization pipeline improved significantly over 8 years).

---

## The Engineering Effort: What Nobody Writes About

The numbers above look clean. The path to get them was not.

### Error 73 — DSP access blocked on industrial firmware

The first DSP attempt produced Error 73: `BAD_HANDLE`. Logcat showed:

```
avc: denied { read } for name="sku" dev="sysfs" ... tclass=file permissive=0
```

SNPE needs to read the Snapdragon SKU from sysfs to select the correct DSP routing. On CT70/CT47, SELinux in enforcing mode denies this. The symptom is silent CPU fallback — inference runs at CPU speed with no error message at the app level. The fix: call `setUnsignedPD(true)` and set `ADSP_LIBRARY_PATH` with semicolon separators before instantiating the neural network builder. The exact sequence matters. This took approximately three days to diagnose.

### NNAPI — 500ms, always CPU

NNAPI times on CT70 and CT47 match CPU times exactly. The NNAPI delegate initialises, accepts the model, and then quietly executes on CPU. This is confirmed by timing: 275 ms for YOLO11n NNAPI matches 275 ms for float32 CPU on CT47. Honeywell industrial firmware does not expose the HTP to NNAPI. This was true in 2018 on SD660 and remains true in 2026.

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
| CT70 QCS6690 Hex780 | ✗ Fails | 0x138d vtcm unsupported | Honeywell restricts unsigned PD VTCM below 4MB |
| CT47 QCS6490 Hex770 | ✗ Fails | 0x138d vtcm unsupported | Same firmware restriction |
| S21 SM8450 Hex780 | ✓ Works | — | Consumer firmware, no VTCM restriction |
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

The screenshot from that video — all four devices detecting the same scene simultaneously — is the clearest summary of eight years of progress. CT47 Hexagon 770 at 37ms, CT70 Hexagon 780 at 22ms, OPPO Hexagon 698 at 84ms, Samsung S21 Hexagon 780 at 20ms. The OPPO result is the pre-HTP baseline: same SDK, four times slower, no HMX units.

---

## Future Outlook (2026–2030)

Eight years between articles is too long. Here is what needs to happen in the next four. First three are my predictions or my wishes, last two is what “claude” is expecting

### Prediction 1: NPU-Native OS Buffers (my wish)

Android's SurfaceFlinger still passes camera frames through a CPU-readable buffer before handing them to any ML runtime. Every pipeline measured in this article pays a resize, a pixel-extraction loop, and a ByteBuffer allocation on the CPU before a single DSP cycle runs. The logical endpoint is a camera HAL that writes directly into a format the HTP can consume — typed tensor buffers that live in secure enclave memory and never touch the application processor. Zero-copy from sensor to inference. This would eliminate 15–30ms of the total pipeline time measured here regardless of which inference path is used.

The `.bin` path — or any simplified host-compiled path — working out of the box on all devices is a prerequisite for this to be useful. As documented in this article, a host-compiled binary fails on industrial firmware due to VTCM negotiation. That cannot be fixed by the application developer. It requires the OS to broker memory allocation between the compiler and the firmware — which is exactly what SNPE does at runtime today, manually, per-device.

### Prediction 2: On-Device LoRA Fine-Tuning for Vision (my wish)

YOLO heads are small. The detection head for YOLO11n is a few hundred thousand parameters — well within what a Hexagon 780 can process for gradient computation at interactive latency. The infrastructure is already partially there: Qualcomm AI Hub runs model compilation on-device; the QNN HTP can do backward passes in principle.

My practical wish: a YOLO11 detection head that fine-tunes itself overnight on locally collected images, using on-device HTP kernels, without the model weights ever leaving the device. For industrial scanning — the specific use case of the CT70 and CT47 — this would mean the box damage detector learns the specific lens profile, lighting conditions, and box surface textures of each warehouse without any labelling overhead. The domain gap between COCO training and industrial deployment is the single largest quality bottleneck today. On-device LoRA could close it.

### Prediction 3: The End of the Format War (my wish)

The toolchain used in this article: PyTorch → Ultralytics YOLO → ONNX → TFLite (snpe-tflite-to-dlc) or ONNX (snpe-onnx-to-dlc) → snpe-dlc-quantize → snpe-dlc-graph-prepare → QNN context binary. Seven steps. Three different intermediate representations. Two conversion tools that produce different results for the same model. One tool that fails on YOLO11's C2f backbone entirely.

I hope my prediction is right: ONNX Runtime's QNN Execution Provider — or whatever wins the convergence race — will eventually become the abstraction layer that makes this pipeline invisible. You will describe a model, a device, and a latency target, and the runtime will handle the rest. SNPE/DLC will be remembered the way people remember writing device-specific OpenGL extensions — correct for its time, necessary for performance, ultimately replaced by a higher-level API that was good enough and universal.

### Prediction 4: Calibration data stays on device (Claude's addition)

Every INT8 quantization step in this pipeline required calibration images: a representative sample of real inference data to set the quantization scales. Those calibration images had to be collected, curated, and stored on a development machine. For a model deployed to thousands of industrial scanners scanning different products in different warehouses, there is no single correct calibration set.

The missing piece: the device itself should run a background calibration pass after deployment. The first N frames of real inference are used to refine quantization scales for the specific distribution of inputs that device actually sees. The quantized model self-calibrates to its deployment context. On the Hexagon HTP, a calibration pass over 1000 frames at stride-8 would take under 10 seconds. Combined with Prediction 2 (LoRA fine-tuning), this gives a model that is both quantization-calibrated and fine-tuned to its actual operating environment — without any development machine involvement after the initial deployment.

### Prediction 5: NNAPI actually routes to the DSP (Claude's addition)

The single finding that has not changed between 2018 and 2026: NNAPI does not reach the Hexagon DSP on Qualcomm industrial firmware. On the SD660 in 2018, NNAPI was not supported at all. On the QCS6690 and QCS6490 in 2026, NNAPI initialises, accepts the model, and silently executes on CPU — matching CPU timing exactly.

NNAPI was designed to be exactly the abstraction layer that Prediction 3 describes: write once, run on any accelerator, let the OS route to the fastest available hardware. It exists. It ships on every Android device. It does not work for this use case on any device tested in this article over eight years.

The wish: Android 16 or 17 mandates that any SoC shipping with a dedicated ML accelerator exposes it through NNAPI with verified hardware delegation. Not optional. Not firmware-dependent. Not silently falling back to CPU with no error. Hardware acceleration via the standard API, with an actual test suite that OEMs must pass before using the "AI-capable" marketing label.

---

## Conclusion

In 2018 the question was: *is the extra effort worth it?* The answer was: barely, and only for specific workloads.

In 2026 the answer is: yes, unambiguously, if you accept the cost.

The hardware improved by roughly 20×. A real-time object detector with 80 classes and 6.5 GFLOPs runs in 4.5ms on a Hexagon 770 that fits in a warehouse scanner. The same task would have taken 450ms on the hardware available in 2018.

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

*Source code and WSL scripts: [GitHub repository link]*

*Tags: Qualcomm DSP, Hexagon HTP, YOLO11, Android, SNPE, QNN, Object Detection, Edge AI, Industrial AI, Honeywell CT70, CT47*
