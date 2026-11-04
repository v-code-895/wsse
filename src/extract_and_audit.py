import zipfile
import os
import json
import h5py
import numpy as np

ZIP_FILES = [
    "legacy/TinySR-18.zip",
    "legacy/PCA-GAN.zip",
]

def unzip_all():
    for zf in ZIP_FILES:
        if os.path.exists(zf):
            out_dir = zf.replace(".zip", "")
            if not os.path.exists(out_dir):
                print(f"[*] Extracting {zf} -> {out_dir}/")
                with zipfile.ZipFile(zf, 'r') as z:
                    z.extractall(".")

def audit_h5_weights(h5_path, report_lines):
    report_lines.append(f"\n{'='*70}")
    report_lines.append(f"WEIGHT-SPACE STATIC AUDIT: {h5_path}")
    report_lines.append(f"{'='*70}")
    
    with h5py.File(h5_path, 'r') as f:
        # Print Keras version / model config if stored in attributes
        if 'keras_version' in f.attrs:
            report_lines.append(f"Keras Version: {f.attrs['keras_version']}")
        if 'model_config' in f.attrs:
            cfg = f.attrs['model_config']
            if isinstance(cfg, bytes):
                cfg = cfg.decode('utf-8')
            report_lines.append("Model Config JSON found in H5 attributes.")

        # Walk all datasets (weight tensors) in the H5 file
        tensors = []
        def visitor(name, node):
            if isinstance(node, h5py.Dataset):
                tensors.append((name, node[:]))
        f.visititems(visitor)

        total_params = sum(arr.size for _, arr in tensors)
        report_lines.append(f"Total Tensors: {len(tensors)} | Total Parameters: {total_params:,}\n")

        for name, w in tensors:
            w_f32 = w.astype(np.float32)
            shape = w.shape
            norm = float(np.linalg.norm(w_f32))
            mean = float(np.mean(w_f32))
            std = float(np.std(w_f32))
            sparsity = float(np.mean(np.abs(w_f32) < 1e-5) * 100.0)
            
            line = f"Tensor: {name} | Shape: {shape} | L2: {norm:.4f} | Mean: {mean:+.5f} | Std: {std:.5f} | NearZero%: {sparsity:.1f}%"
            
            # If 4D Conv kernel: (H, W, C_in, C_out) in Keras
            if w_f32.ndim == 4:
                H, W, C_in, C_out = shape
                # Per-output-channel Frobenius norms
                ch_norms = np.linalg.norm(w_f32.reshape(-1, C_out), axis=0)
                dead_ch = int(np.sum(ch_norms < 0.05 * np.median(ch_norms)))
                
                # Pairwise cosine similarity between output filters to detect redundancy
                flat_out = w_f32.reshape(-1, C_out)
                norms = np.linalg.norm(flat_out, axis=0, keepdims=True) + 1e-8
                normed = flat_out / norms
                cos_sim = np.abs(normed.T @ normed)
                np.fill_diagonal(cos_sim, 0.0)
                max_collinear = float(np.max(cos_sim)) if C_out > 1 else 0.0
                redundant_pairs = int(np.sum(cos_sim > 0.85) // 2)

                # Spatial frequency analysis (Low-pass vs High-pass / Edge-detector ratio)
                # Sum of weights vs sum of absolute weights per 2D kernel
                dc_energy = np.mean(np.abs(np.sum(w_f32, axis=(0, 1))))
                total_energy = np.mean(np.sum(np.abs(w_f32), axis=(0, 1))) + 1e-8
                lp_ratio = float(dc_energy / total_energy) # High = smoother/averager, Low = zero-mean edge detector

                line += (f"\n   -> [Conv4D] OutCh: {C_out} (Dead: {dead_ch}) | "
                         f"ChNorms[min={ch_norms.min():.3f}, med={np.median(ch_norms):.3f}, max={ch_norms.max():.3f}] | "
                         f"MaxCosSim: {max_collinear:.3f} (Pairs>0.85: {redundant_pairs}) | "
                         f"LowPassRatio: {lp_ratio:.3f}")
                
                # If first or last conv layer (C_in==3 or C_out==3 or C_in==1), dump actual 3x3 spatial slice
                if C_in in (1, 3) or C_out in (1, 3):
                    sample_k = np.round(w_f32[:, :, 0, 0], 4).tolist()
                    line += f"\n   -> [Sample Kernel [:,:,0,0]]: {sample_k}"

            # If 2D Dense matrix: (In, Out)
            elif w_f32.ndim == 2 and min(shape) > 1:
                s = np.linalg.svd(w_f32, compute_uv=False)
                top5_s = [round(float(x), 3) for x in s[:5]]
                eff_rank = float((np.sum(s)**2) / (np.sum(s**2) + 1e-8))
                line += f"\n   -> [Dense2D] SVD Top-5: {top5_s} | EffectiveRank: {eff_rank:.1f}/{min(shape)}"

            report_lines.append(line)

def dump_code_summaries(report_lines):
    py_files = [
        "legacy/TinySR-18/DLSS GAN Training.py",
        "legacy/TinySR-18/Load Model and Analyze Results.py",
        "legacy/TinySR-18/model.json",
        "legacy/PCA-GAN/PCA GAN Training.py",
        "legacy/PCA-GAN/PCA GAN Inference.py",
        "legacy/PCA-GAN/Model Visualization.py",
    ]
    for pf in py_files:
        for root_prefix in ["", "legacy/TinySR-18/", "legacy/PCA-GAN/"]:
            candidate = os.path.join(root_prefix, pf) if root_prefix else pf
            if os.path.exists(candidate):
                report_lines.append(f"\n{'='*70}\nFILE: {candidate}\n{'='*70}")
                with open(candidate, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    report_lines.append(content[:4000]) # First 4000 chars
                break

if __name__ == "__main__":
    unzip_all()
    report = []
    dump_code_summaries(report)
    
    for root, _, files in (w for d in ("legacy/TinySR-18", "legacy/PCA-GAN") for w in os.walk(d)):
        for file in files:
            if file.endswith(".h5"):
                audit_h5_weights(os.path.join(root, file), report)
                
    out_text = "\n".join(report)
    with open("results/audit/weight_audit_dump.txt", "w", encoding="utf-8") as f:
        f.write(out_text)
    print(f"\n[+] Done! Saved {len(out_text):,} chars to weight_audit_dump.txt")
    print("Read results/audit/weight_audit_dump.txt alongside the paper's Section 4.")