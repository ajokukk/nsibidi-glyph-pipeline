"""
Compression experiment (E12): can the embedding network be made small and fast without losing accuracy?

Variants of the SAME fine-tuned ResNet-18 (a 512-d embedding, then cosine similarity to prototypes):
    torch_fp32   original PyTorch model
    onnx_fp32    exported to ONNX, run by ONNX Runtime
    onnx_fp16    ONNX weights converted to half precision      (needs: pip install onnxconverter-common)
    onnx_int8    ONNX static post-training quantisation (QDQ, per-channel), calibrated on dictionary crops

For every variant it reports file size, latency per crop (batch of 1, median of --runs, on 1 thread and on all
threads of THIS computer), few-shot accuracy with the same episodic protocol as experiments E1-E7 (so the numbers
are directly comparable to Table 4.6), and the mean cosine similarity of its embeddings to the fp32 embeddings.

    pip install onnx onnxruntime
    python compress_classifier.py --resnet-ckpt e7_clean_s0.pth --gray-norm ^
        --support-dir "%R%\\train" --query-dir "%R%\\test" --out-dir compress_out

Needs classify_detections.py in the same folder.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify_detections as cd                      # noqa: E402


class OrtEmbedder:
    """Same interface as classify_detections.Embedder (PIL images -> L2-normalised features)."""

    def __init__(self, session, prep):
        self.sess, self.prep = session, prep
        self.inp = session.get_inputs()[0].name

    def __call__(self, pil_images):
        x = self.prep._tensor(pil_images).cpu().numpy().astype(np.float32)
        out = np.concatenate([self.sess.run(None, {self.inp: x[i:i + 1]})[0] for i in range(len(x))], 0)
        out = out.astype(np.float32)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)


class TorchEmbedder:
    def __init__(self, base):
        self.base = base

    def __call__(self, pil_images):
        return self.base(pil_images)


def export_onnx(net, path):
    """Export with several fallbacks: a MemoryError inside the exporter (seen on low-RAM or 32-bit
    Python) is often avoided by turning off constant folding, using a fixed batch size or an older opset."""
    import gc
    import torch
    net = net.cpu().eval()
    x = torch.zeros(1, 3, cd.IMG_SIZE, cd.IMG_SIZE)
    attempts = [
        dict(opset_version=13, do_constant_folding=True, dyn=True),
        dict(opset_version=13, do_constant_folding=False, dyn=True),
        dict(opset_version=12, do_constant_folding=False, dyn=False),
        dict(opset_version=11, do_constant_folding=False, dyn=False),
    ]
    last = None
    for a in attempts:
        kw = dict(input_names=["x"], output_names=["emb"], opset_version=a["opset_version"],
                  do_constant_folding=a["do_constant_folding"])
        if a["dyn"]:
            kw["dynamic_axes"] = {"x": {0: "n"}, "emb": {0: "n"}}
        gc.collect()
        try:
            try:                           # newer torch: classic exporter (no onnxscript needed)
                torch.onnx.export(net, x, path, dynamo=False, **kw)
            except TypeError:              # older torch has no `dynamo` argument
                torch.onnx.export(net, x, path, **kw)
            print(f"  ONNX export ok ({a})")
            return
        except (MemoryError, RuntimeError) as ex:
            last = ex
            print(f"  export failed with {a}: {type(ex).__name__}: {ex}")
    raise SystemExit(f"ONNX export failed in every setting ({last}). Close other programs (Streamlit, browsers), "
                     "check that Python is 64-bit (python -c \"import struct;print(struct.calcsize('P')*8)\"), "
                     "and send me the output of that command and of: python -c \"import torch;print(torch.__version__)\"")


def to_fp16(src, dst):
    import onnx
    from onnxconverter_common import float16
    m = onnx.load(src)
    onnx.save(float16.convert_float_to_float16(m, keep_io_types=True), dst)


def to_int8(src, dst, prep, calib_files, n):
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType, quantize_static)
    from PIL import Image

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib_files[:n])

        def get_next(self):
            p = next(self.it, None)
            if p is None:
                return None
            x = prep._tensor([cd.load_rgb(p)]).cpu().numpy().astype(np.float32)
            return {"x": x}

    quantize_static(src, dst, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)


def session(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def latency_ms(fn, sample, runs, warm=10):
    for _ in range(warm):
        fn([sample])
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn([sample])
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resnet-ckpt", required=True)
    ap.add_argument("--gray-norm", action="store_true")
    ap.add_argument("--support-dir", nargs="+", required=True)
    ap.add_argument("--query-dir", nargs="+", required=True)
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 2, 5])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--pool-per-class", type=int, default=60)
    ap.add_argument("--calib", type=int, default=200, help="calibration crops for int8")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=cd.SEED)
    ap.add_argument("--out-dir", default="compress_out")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    import torch

    base = cd.Embedder("resnet", resnet_ckpt=a.resnet_ckpt, gray_norm=a.gray_norm)
    base.device = torch.device("cpu")
    base.resnet = base.resnet.cpu().eval()
    base.mean, base.std = base.mean.cpu(), base.std.cpu()

    torch_path = os.path.join(a.out_dir, "torch_fp32.pth")
    torch.save(base.resnet.state_dict(), torch_path)
    paths = {"torch_fp32": torch_path}
    fp32 = os.path.join(a.out_dir, "onnx_fp32.onnx")
    export_onnx(base.resnet, fp32)
    paths["onnx_fp32"] = fp32

    calib = [p for fs in cd.list_class_files(a.support_dir[0]).values() for p in fs]
    np.random.RandomState(0).shuffle(calib)
    try:
        p16 = os.path.join(a.out_dir, "onnx_fp16.onnx")
        to_fp16(fp32, p16)
        paths["onnx_fp16"] = p16
    except Exception as ex:
        print(f"[skip fp16] {type(ex).__name__}: {ex}   (pip install onnx onnxconverter-common)")
    try:
        p8 = os.path.join(a.out_dir, "onnx_int8.onnx")
        to_int8(fp32, p8, base, calib, a.calib)
        paths["onnx_int8"] = p8
    except Exception as ex:
        print(f"[skip int8] {type(ex).__name__}: {ex}")

    sample = cd.load_rgb(calib[0])
    probe = [cd.load_rgb(p) for p in calib[:64]]
    ref = base(probe)

    rows = []
    for name, path in paths.items():
        print(f"\n=== {name} ===")
        if name == "torch_fp32":
            torch.set_num_threads(1)
            emb = TorchEmbedder(base)
            lat1 = latency_ms(emb, sample, a.runs)
            torch.set_num_threads(max(1, os.cpu_count() or 1))
            latn = latency_ms(emb, sample, a.runs)
        else:
            emb = OrtEmbedder(session(path, 0), base)
            latn = latency_ms(emb, sample, a.runs)
            lat1 = latency_ms(OrtEmbedder(session(path, 1), base), sample, a.runs)
        cos = float(np.mean(np.sum(emb(probe) * ref, axis=1)))
        res = cd.run_eval(emb, a.support_dir, a.query_dir, a.shots, a.episodes, a.pool_per_class, a.seed,
                          out_csv=None, tag=name)
        row = {"variant": name, "size_MB": f"{os.path.getsize(path) / 1e6:.2f}",
               "latency_ms_1thread": f"{lat1:.1f}", "latency_ms_all_threads": f"{latn:.1f}",
               "cos_vs_fp32": f"{cos:.4f}"}
        for r in res:
            row[f"acc_{r['k_shot']}shot"] = f"{float(r['mean_accuracy']) * 100:.1f} +/- {float(r['std']) * 100:.1f}"
        rows.append(row)
    out = os.path.join(a.out_dir, "compression_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("\n" + "-" * 90)
    for r in rows:
        print(r)
    print(f"\nSaved {out}   (CPU: {os.cpu_count()} logical cores)")


if __name__ == "__main__":
    main()
