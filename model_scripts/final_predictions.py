import os
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import polars as pl
import numpy as np
import math
import gc
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. HYPERPARAMETERS & ABSOLUTE PATHS
# ==========================================
# The script will only generate predictions for rows strictly AFTER this index
LAST_TRAINING_ROW_INDEX = 2142

# Absolute paths based on the script's actual location
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE  = SCRIPT_DIR.parent / "processed_data" / "nifty_master_file.parquet"
MODELS_DIR = SCRIPT_DIR / "final_model_2"
RESULTS_DIR = SCRIPT_DIR.parent / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hardware Optimization for M2 Mac Air (Apple Silicon)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🖥️  Hardware: Using Apple Silicon M2 GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🖥️  Hardware: Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("🖥️  Hardware: Using CPU")

# ==========================================
# 2. PYTORCH ARCHITECTURE (Must match training exactly)
# ==========================================
class RobustInstanceNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta  = nn.Parameter(torch.zeros(num_features))
        self.eps   = eps
    def forward(self, x):
        if len(x.shape) == 3:
            med = torch.median(x, dim=1, keepdim=True)[0]
            mad = torch.median(torch.abs(x - med), dim=1, keepdim=True)[0]
        else:
            med = torch.median(x, dim=0, keepdim=True)[0]
            mad = torch.median(torch.abs(x - med), dim=0, keepdim=True)[0]
        return (x - med) / (mad + self.eps) * self.gamma + self.beta


def generate_decay_mask(seq_len, decay_rate=0.15):
    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i: mask[i, j] = -decay_rate * (i - j)
            else:      mask[i, j] = float('-inf')
    return mask


class GLU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
    def forward(self, x): return torch.sigmoid(self.fc1(x)) * self.fc2(x)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.fc1        = nn.Linear(input_size, hidden_size)
        self.elu        = nn.ELU()
        self.fc2        = nn.Linear(hidden_size, output_size)
        self.dropout    = nn.Dropout(dropout)
        self.gate       = GLU(output_size)
        self.layer_norm = nn.LayerNorm(output_size)
        self.skip_proj  = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
    def forward(self, x):
        residual = self.skip_proj(x)
        x = self.fc1(x); x = self.elu(x); x = self.fc2(x)
        x = self.dropout(x); x = self.gate(x)
        return self.layer_norm(residual + x)


class AnchoredCrossAttentionTFT(nn.Module):
    def __init__(self, num_var_feats, num_dir_feats, vol_state_idx,
                 dte_state_idx,                          # ADDED vs old inference script
                 seq_len=10, d_model=32, nhead=4, dropout=0.2,
                 num_targets=1, num_grn_layers=3, decay_rate=0.15):
        super().__init__()
        self.num_targets = num_targets
        self.seq_len     = seq_len
        self.register_buffer('vol_state_idx_buf',
                             torch.tensor(vol_state_idx, dtype=torch.long))
        self.register_buffer('dte_state_idx_buf',        # ADDED vs old inference script
                             torch.tensor(dte_state_idx, dtype=torch.long))

        self.dir_embed      = nn.Parameter(torch.randn(num_dir_feats, d_model))
        self.dir_attn       = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                                    dropout=dropout, batch_first=True)
        self.dir_layer_norm = nn.LayerNorm(d_model)
        self.mu_layer       = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, num_targets))

        self.var_norm       = RobustInstanceNorm(num_var_feats)
        self.var_proj       = nn.Linear(num_var_feats, d_model)
        self.var_grn_in     = nn.Sequential(*[
            GatedResidualNetwork(d_model, d_model * 2, d_model, dropout)
            for _ in range(num_grn_layers)])
        self.var_attn       = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                                    dropout=dropout, batch_first=True)
        self.var_layer_norm = nn.LayerNorm(d_model)
        self.var_grn_out    = GatedResidualNetwork(d_model, d_model, d_model, dropout)

        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                                dropout=dropout, batch_first=True)
        self.fusion_grn = GatedResidualNetwork(d_model * 2, d_model * 2, d_model, dropout)

        self.spread_layer = nn.Linear(d_model, 4)
        self.aux_layer    = nn.Linear(d_model, num_targets)

        # CHANGED vs old inference script:
        # Vol gate input: vol_state_now(3) + vol_state_hist(3) + dte_state(4) = 10
        _vol_gate_in = len(vol_state_idx) * 2 + len(dte_state_idx)
        self.vol_gate = nn.Sequential(
            nn.Linear(_vol_gate_in, 16), nn.GELU(),
            nn.Linear(16, 4), nn.Sigmoid())

        pe       = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.register_buffer('decay_mask', generate_decay_mask(seq_len, decay_rate))

    def forward(self, x, idx_var, idx_dir):
        x_var = x[:, :, idx_var]
        x_dir = x[:, :, idx_dir]
        mask  = self.decay_mask.to(x.device)

        x_dir_last    = x_dir[:, -1, :]
        dir_embedded  = x_dir_last.unsqueeze(-1) * self.dir_embed.unsqueeze(0)
        d_attn_out, _ = self.dir_attn(dir_embedded, dir_embedded, dir_embedded)
        d_out         = self.dir_layer_norm(dir_embedded + d_attn_out)
        dir_context   = d_out.mean(dim=1)
        expected_return = self.mu_layer(dir_context)

        x_var_norm    = self.var_norm(x_var)
        v_in          = self.var_proj(x_var_norm)
        v_in          = self.var_grn_in(v_in) + self.pe
        v_attn_out, _ = self.var_attn(v_in, v_in, v_in, attn_mask=mask)
        v_seq         = self.var_layer_norm(v_in + v_attn_out)
        v_last        = self.var_grn_out(v_seq[:, -1, :])

        query        = dir_context.unsqueeze(1).detach()
        cross_out, _ = self.cross_attn(query, v_seq, v_seq)
        cross_out    = cross_out.squeeze(1)
        fused        = self.fusion_grn(torch.cat([v_last, cross_out], dim=-1))

        vol_state_now  = x_var[:, -1, self.vol_state_idx_buf]
        hist_len       = min(5, x_var.shape[1] - 1)
        vol_state_hist = (x_var[:, -(hist_len + 1):-1, self.vol_state_idx_buf].mean(dim=1)
                          if hist_len > 0 else vol_state_now)
        dte_state      = x_var[:, -1, self.dte_state_idx_buf]  # ADDED vs old inference script

        # CHANGED vs old inference script: dte_state appended to vol gate input
        vol_gate  = 0.5 + self.vol_gate(
            torch.cat([vol_state_now, vol_state_hist, dte_state], dim=-1))

        raw_spreads = (F.softplus(self.spread_layer(fused)) + 1e-5) * vol_gate

        dir_bias      = torch.tanh(expected_return.detach())
        skew          = 0.3 * dir_bias

        spread_d_core = (raw_spreads[:, 0:1] * (1.0 - skew)).clamp(min=1e-5)
        spread_u_core = (raw_spreads[:, 1:2] * (1.0 + skew)).clamp(min=1e-5)
        spread_d_tail = (raw_spreads[:, 2:3] * (1.0 - skew)).clamp(min=1e-5)
        spread_u_tail = (raw_spreads[:, 3:4] * (1.0 + skew)).clamp(min=1e-5)

        q40  = expected_return - spread_d_core
        q60  = expected_return + spread_u_core
        q075 = q40 - spread_d_tail
        q925 = q60 + spread_u_tail

        aux_preds = self.aux_layer(v_last)
        return q075, q40, expected_return, q60, q925, aux_preds


# ==========================================
# 3. INFERENCE SCRIPT
# ==========================================
def main():
    print(f"{'='*50}")
    print("🚀 RUNNING LIVE INFERENCE ENGINE")
    print(f"{'='*50}")

    # 1. Locate Latest Metadata
    meta_files = sorted(glob.glob(str(MODELS_DIR / "*_meta.json")))
    if not meta_files:
        raise FileNotFoundError(f"No metadata file found in {MODELS_DIR}. Are you sure the file is there?")

    latest_meta_path = meta_files[-1]
    print(f"Loading Configuration from: {Path(latest_meta_path).name}")

    with open(latest_meta_path, "r") as f:
        meta = json.load(f)

    # Extract strictly mapped architecture and dynamic configs
    feature_cols = meta["feature_cols"]
    seq_len  = meta["CONFIG"]["seq_len_1d"]
    outer_ci = meta["CONFIG"].get("outer_ci_pct", 85)
    inner_ci = meta["CONFIG"].get("inner_ci_pct", 20)
    VAR_IDX  = torch.tensor(meta["VAR_IDX"], dtype=torch.long, device=device)

    # 2. Re-hydrate Scalers using frozen state
    first_weight_file = meta["saved_model_files"][0]
    first_weight_path = MODELS_DIR / first_weight_file

    if not first_weight_path.exists():
        raise FileNotFoundError(
            f"Cannot find the model weight file {first_weight_file} to extract scalers.")

    print(f"Extracting scalers from {first_weight_file}...")
    ckpt_for_scalers = torch.load(first_weight_path, map_location="cpu", weights_only=False)

    sc_std = StandardScaler()
    sc_std.mean_  = np.array(ckpt_for_scalers["sc_std_mean"])
    sc_std.scale_ = np.array(ckpt_for_scalers["sc_std_scale"])

    sc_dir = StandardScaler(with_mean=False)
    sc_dir.scale_ = np.array(ckpt_for_scalers["sc_dir_scale"])

    del ckpt_for_scalers
    gc.collect()

    # 3. Load Data
    print(f"Loading pre-processed market data from: {DATA_FILE.name}")
    df_pd = pl.read_parquet(DATA_FILE).to_pandas()

    total_rows = len(df_pd)
    if total_rows <= LAST_TRAINING_ROW_INDEX + 1:
        print(f"No new rows to predict. Dataset ends at index {total_rows - 1}. Awaiting market data.")
        return

    vol_raw  = df_pd['Realized_Vol_Ann'].ewm(span=meta["vol_ewma_span"], min_periods=1).mean().values.astype(np.float32)

    X_raw    = df_pd[feature_cols].values.astype(np.float32)
    X_scaled = np.zeros_like(X_raw, dtype=np.float32)
    X_scaled[:, meta["VAR_IDX"]]     = sc_std.transform(X_raw[:, meta["VAR_IDX"]])
    X_scaled[:, meta["ALL_DIR_IDX"]] = sc_dir.transform(X_raw[:, meta["ALL_DIR_IDX"]])

    # 4. Prepare Memory-Safe Prediction Pipeline
    valid_indices  = [i for i in range(LAST_TRAINING_ROW_INDEX + 1, total_rows)
                      if i - seq_len + 1 >= 0]
    ensemble_preds = {i: [[] for _ in range(5)] for i in valid_indices}

    print(f"Generating Predictions for {len(valid_indices)} new rows...")

    # LOOP OVER MODELS
    for weight_file in meta["saved_model_files"]:
        weight_path = MODELS_DIR / weight_file
        if not weight_path.exists():
            print(f"⚠️ Warning: Missing weight file {weight_file}")
            continue

        print(f"  -> Inferring via {weight_file}...")
        ckpt       = torch.load(weight_path, map_location=device, weights_only=False)
        model_arch = ckpt["arch"]
        # If checkpoint was saved before dte_state_idx was added to the arch,
        # inject it from the metadata so old and new checkpoints both load cleanly.
        if "dte_state_idx" not in model_arch:
            model_arch["dte_state_idx"] = meta["DTE_STATE_IDX"]
        m = AnchoredCrossAttentionTFT(**model_arch).to(device)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()

        m_dir_idx = ckpt["model_dir_idx"].to(device)

        with torch.inference_mode():
            for i in valid_indices:
                x_window = X_scaled[i - seq_len + 1 : i + 1]
                x_tensor = torch.tensor(x_window).unsqueeze(0).to(device)
                o = m(x_tensor, VAR_IDX, m_dir_idx)
                for k in range(5):
                    ensemble_preds[i][k].append(o[k].cpu().item())

        # MEMORY CLEARING FOR MAC M2
        del m, ckpt, m_dir_idx
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # 5. Average the Ensemble & Convert to Underlying Price Bounds
    results = []
    for i in valid_indices:
        vol_target = vol_raw[i]
        pred_date  = df_pd['Date'].iloc[i]

        # The base underlying price for the prediction day (Day T)
        base_price = float(df_pd['Spot_Close'].iloc[i])

        # Average log returns across the models, multiply by EWMA Vol
        Qm_raw = [np.mean(ensemble_preds[i][k]) * vol_target for k in range(5)]

        # Convert Log Returns to Nominal Price: Price_T1 = Price_T0 * e^(LogReturn)
        Qm_price = [base_price * math.exp(q) for q in Qm_raw]

        # Order: 6 Price Columns + 6 Raw Output Columns
        results.append({
            "Prediction_Date": pred_date,

            # Underlying Price Predictions
            "Base_Price": base_price,
            f"Lower_{outer_ci}_CI_Price": Qm_price[0],
            f"Lower_{inner_ci}_CI_Price": Qm_price[1],
            "Mean_Predicted_Price":        Qm_price[2],
            f"Upper_{inner_ci}_CI_Price": Qm_price[3],
            f"Upper_{outer_ci}_CI_Price": Qm_price[4],

            # Raw Log Return Predictions
            f"Lower_{outer_ci}_CI_Raw": Qm_raw[0],
            f"Lower_{inner_ci}_CI_Raw": Qm_raw[1],
            "Mean_Predicted_Raw":        Qm_raw[2],
            f"Upper_{inner_ci}_CI_Raw": Qm_raw[3],
            f"Upper_{outer_ci}_CI_Raw": Qm_raw[4],
            "Implied_Vol_Scaler": vol_target,
        })

    # 6. Save Results
    if results:
        df_res = pd.DataFrame(results)

        start_date_str = pd.to_datetime(df_res['Prediction_Date'].iloc[0]).strftime("%Y%m%d")
        end_date_str   = pd.to_datetime(df_res['Prediction_Date'].iloc[-1]).strftime("%Y%m%d")
        output_name    = f"model_predictions_{start_date_str}_to_{end_date_str}.csv"
        output_path    = RESULTS_DIR / output_name

        df_res.to_csv(output_path, index=False)
        print(f"\n✅ SUCCESS! Predicted {len(df_res)} new days.")
        print(f"Results saved to: {output_path}")
        print(df_res.head())


if __name__ == "__main__":
    main()