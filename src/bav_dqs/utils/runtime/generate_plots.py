import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal as sp_sig
from scipy.signal.windows import hann, blackman
from pathlib import Path
import argparse

from bav_dqs.utils.io.data_manager import DataManager
from bav_dqs.utils.plugins.dirac_simulation import load_dirac_simulation_yaml, parse_analysis_cfg, parse_detector_cfg, parse_validity_cfg
from bav_dqs.utils.helpers.resolution import safe_window

COLOR_MASS = '#1f77b4'
COLOR_CAUSAL = '#d62728'
COLOR_BASELINE = '#7f7f7f'
COLOR_HIGHLIGHT = '#ff7f0e'
COLOR_CAUSALITY = '#17becf'

class ScientificReport:
    def __init__(self, data_file: Path):
        self.data_file = data_file
        self.output_dir = data_file.parent / f"report_{self.data_file.name.removesuffix('.h5')}".lower()
        self.output_dir.mkdir(exist_ok=True)
        
        plt.rcParams.update({
            'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.2,
            'lines.linewidth': 1.2, 'figure.dpi': 300
        })

    @staticmethod
    def _edge_window(run, default: int = 2) -> int:
        """Edge window W used by the detector for this run (stored as an attribute).
        Legacy files written before provenance capture fall back to 2."""
        return int(run.attrs.get("edge_window", default))

    def plot_lieb_robinson_cone(self, run, n_q):
        """
        Visualize the Lieb-Robinson Light Cone.
        It shows that information (ZZ) travels ahead of mass (Z).
        """

        corr_data = np.abs(run["correlation"][:])
        occ_data = np.abs(run["occ_full"][:])

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        im1 = ax1.imshow(occ_data, aspect='auto', origin='lower', cmap='YlGnBu')
        ax1.set_title(f"Mass Transport (Z) N={n_q}", color=COLOR_MASS)
        ax1.set_xlabel("Site Index")
        ax1.set_ylabel("Physical Time")
        plt.colorbar(im1, ax=ax1, label=r'$\langle Z_j \rangle$')

        im2 = ax2.imshow(corr_data, aspect='auto', origin='lower', cmap='YlOrRd')
        ax2.set_title(f"Information Spread (ZZ) N={n_q}", color=COLOR_CAUSAL)
        ax2.set_xlabel("Bond Index (ref=center)")
        plt.colorbar(im2, ax=ax2, label=r'$| \langle Z_{ref} Z_j \rangle |$')

        plt.tight_layout()
        plt.savefig(self.output_dir / f"causal_cone_comparison_n{n_q}.png")
        plt.close()

    def plot_calibration_floor_5sigma(self, run, n_q):
        """
        Demonstrates the rigor of 5-sigma calibration.
        """
        thr_global = run.attrs.get("threshold", 0.0)
    
        thr_z = max(run.attrs.get("threshold_occupancy", thr_global), 1e-7)
        thr_zz = max(run.attrs.get("threshold_correlation", thr_global), 1e-7)

        labels = ['Occupancy (Z)', 'Correlation (ZZ)']
        values = [thr_z, thr_zz]

        _, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, values, color=['#3498db', '#e74c3c'], alpha=0.7)
        
        ax.axhline(y=1e-3, color='black', ls='--', alpha=0.3, label='Standard Heuristic')

        ax.set_yscale('log')
        ax.set_ylim(1e-6, max(max(values) * 10, 1e-2)) 
        
        ax.set_title(rf"Adaptive 5-$\sigma$ Detection Floor (N={n_q})")
        ax.set_ylabel("Threshold Amplitude (log)")
        ax.legend(fontsize='small')

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2e}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    def plot_richardson_error(self, run, n_q, dt):
        """Describes Richardson convergence at the boundary."""
        if "occ_rich" not in run:
            return

        W = self._edge_window(run)
        # None for legacy files that did not record p (never assume a default order).
        p = int(run.attrs["richardson_order_p"]) if "richardson_order_p" in run.attrs else None
        t = np.arange(run["occ_full"].shape[0]) * dt
        edge_full = np.abs(np.mean(run["occ_full"][:, :W], axis=1))   # base grid, step Δt
        # refined grid (step Δt/2) sampled at the even steps, i.e. on the base time grid
        edge_half = np.abs(np.mean(run["occ_half"][:][0::2][:, :W], axis=1)) if "occ_half" in run else None
        edge_rich = np.abs(np.mean(run["occ_rich"][:, :W], axis=1))   # Richardson combination of Δt and Δt/2

        _, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(t, edge_full, label=r'base grid ($\Delta t$)', color=COLOR_BASELINE, alpha=0.6)
        if edge_half is not None:
            ax.semilogy(t[:len(edge_half)], edge_half, label=r'refined grid ($\Delta t/2$, twice the steps)', color=COLOR_MASS, ls='--', alpha=0.8)
        ax.semilogy(t[:len(edge_rich)], edge_rich, label=r'Richardson($\Delta t$, $\Delta t/2$)', color=COLOR_CAUSAL, lw=1.5)
        
        # NOTE: these curves are edge occupations (mean over the W left-edge sites), NOT numerical errors.
        # The error against an exact reference is produced by ``verify_trotter_order``.
        # Richardson uses the two resolutions Δt and Δt/2; p is the global error order of the integrator.
        p_txt = f", p={p}" if (p is not None and p > 0) else ", p not recorded"
        ax.set_title(rf"Edge signal at $\Delta t$, $\Delta t/2$ and Richardson (N={n_q}, W={W}{p_txt})", fontsize=10)
        ax.set_ylabel("Edge occupation (log)")
        ax.set_xlabel("Physical Time")
        ax.legend(fontsize='small')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"richardson_error_n{n_q}.png")
        plt.close()

    def plot_validity_gating_summary(self, run, n_q, dt, p_min, auto_thr, cfg):
        """
        The Admissibility Gating.
        It shows the exact moment when the simulation ceases to be valid.
        """
        validity_cfg = parse_validity_cfg(cfg)
        t_warmup = p_min * dt
        ref_ds = "metric_full_right"
        if ref_ds not in run:
            ref_ds = list(run.keys())[0]
        t = np.arange(run[ref_ds].shape[0]) * dt
        # Boundary marker at both resolutions; the gate fires on whichever edge hits first.
        dt_h = float(run.attrs.get("dt_half", dt / 2.0))
        dl = (np.maximum(run["metric_full_left"][:], run["metric_full_right"][:])
              if "metric_full_left" in run else run["metric_full_right"][:])
        if "metric_half_left" in run and "metric_half_right" in run:
            dl_h = np.maximum(run["metric_half_left"][:], run["metric_half_right"][:])
            t_h = np.arange(dl_h.shape[0]) * dt_h
        else:
            dl_h, t_h = None, None

        _, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, dl, label=r'edge marker $\Delta$ at $\Delta t$', color=COLOR_MASS)
        if dl_h is not None:
            ax.plot(t_h, dl_h, label=r'edge marker $\Delta$ at $\Delta t/2$', color=COLOR_BASELINE, ls=':', alpha=0.8)

        # Hits on the two resolutions are compared in PHYSICAL TIME (see utils/helpers/resolution.py).
        hit_z = int(run.attrs.get("first_hit_full", -1))   # hit index on the base grid (step Δt)
        hit_h = int(run.attrs.get("first_hit_half", -1))   # hit index on the refined grid (step Δt/2)
        sw = safe_window(hit_z, hit_h, dt)

        if sw.t_full is not None:
            ax.axvline(x=sw.t_full, color=COLOR_CAUSAL, ls='--', label=r'hit at $\Delta t$')
        if sw.t_half is not None:
            ax.axvline(x=sw.t_half, color=COLOR_HIGHLIGHT, ls=':', label=r'hit at $\Delta t/2$')

        if auto_thr:
            ax.axvspan(0, t_warmup, color=COLOR_HIGHLIGHT, alpha=0.15, label='Calibration (Warmup)')
            ax.axvline(x=t_warmup, color=COLOR_HIGHLIGHT, ls='-', lw=1, alpha=0.5)
        ax.set_xlabel("Time")
        ax.set_ylabel(r"Signal Deviation ($\Delta$)")
        
        hits = sw.t_safe is not None
        p_safe_msg = "No HITS"
        is_admissible = True
        reason = ""
        if hits:
            reason = {"dt": r"$\Delta t$", "dt/2": r"$\Delta t/2$", "both": r"$\Delta t$ and $\Delta t/2$"}[sw.limiting]
        t_min = validity_cfg['p_min'] * dt
        valid_until = sw.t_safe if hits else t[-1]
        useful_window = (valid_until - t_warmup) if auto_thr else valid_until
        if hits:
            p_safe_msg = f"$p_{{safe}}$={sw.p_safe_base}"
            is_admissible = useful_window >= t_min
            ax.axvspan(0, sw.t_safe, color=COLOR_CAUSALITY, alpha=0.1, label='Causal Safe')
            ax.axvspan(sw.t_safe, t[-1], color=COLOR_CAUSAL, alpha=0.1, label=f'Invalid domain (first hit at {reason})')

        status_str = "PASS" if is_admissible else "FAIL (window too small)"
        warmup_str = ""
        if auto_thr:
            warmup_str = f"Warmup: {t_warmup:.2f}s, "
        ax.set_title(f"Gating: {status_str} | Validy for {useful_window:.2f}s ({warmup_str}N={n_q}, {p_safe_msg} )")
        ax.legend(loc='upper left', fontsize='x-small')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"validity_gating_n{n_q}.png")
        plt.close()

    def generate_admissibility_table(self, summary_data, p_min: int = 32):
        """
        Generates the inferential admissibility results table (Dirac/Occupancy).
        Focused on validating whether the network size (N) supports the p_min criterion.
        """
        df = pd.DataFrame(summary_data)
        
        df = df.sort_values(["n_qubits"])
        has_hit = df["first_hit_full"] >= 0                    # -1 is the 'no hit' sentinel; index 0 is a valid hit
        l_safe = (df["first_hit_full"] - 1).clip(lower=0)      # safe steps before the hit (conservative convention)
        df["Admissible"] = np.where(~has_hit, "NO HIT", np.where(l_safe >= p_min, "PASS", "FAIL"))
        df["L_safe"] = np.where(has_hit, l_safe.astype(object), "-")
        df["first_hit_full"] = np.where(has_hit, df["first_hit_full"].astype(object), "-")

        numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
        df[numeric_cols] = df[numeric_cols].round(2)

        print("\n[LOG] Admissibility Data:")
        print(df.to_string(index=False))

        df_latex = df.drop(columns=['T_safe', 'max_steps_full', 'm', 'w', 'threshold', 'dt_half', 'dt_full', 'd_hit_eff'], errors='ignore').rename(columns={
            "n_qubits": "Lattice ($N$)",
            "threshold": "Threshold ($\\theta$)",
            "first_hit_full": "First Hit ($p_{\\text{hit}}$)",
            "L_safe": "$p_{\\text{safe}}$",
            "Admissible": r"($p_{\text{safe}} \geq 32$)"
        })
        m_val = df['m'].iloc[0]
        w_val = df['w'].iloc[0]
        p_val = df["max_steps_full"].iloc[0]
        dt_val = df['dt_full'].iloc[0]
        theta_val = df['threshold'].iloc[0]

        caption = (
            r"Boundary detection and inferential admissibility. Use case is a 1D Dirac lattice simulation "
            rf"($m = {m_val:.2f}, w = {w_val:.2f}, \Delta t = {dt_val:.2f}, \theta = {theta_val:.2f}, p_{{\max}} = {p_val}$)."
            r"$p_{\text{safe}}$ represents the temporal support (steps) before causality loss. "
        )

        df_latex.to_csv(self.output_dir / "admissibility_table.csv", index=False)
        
        with open(self.output_dir / "admissibility_table.tex", "w") as f:
            f.write(df_latex.to_latex(
                index=False, 
                caption=caption, 
                label="tab:results_summary", 
                escape=False,
                column_format="ccccccc",
                float_format="%.3f"
            ))

    def plot_zitterbewegung_analysis(self, run, n_q, dt, auto_th, p_min):
        """
        Visualizes the Zitterbewegung oscillation extracted from the average occupancy.
        """
        occ = run["occ_full"][:]
        steps, sites = occ.shape
        t = np.arange(steps) * dt
        
        site_indices = np.arange(sites)
        x_mean = np.sum(occ * site_indices, axis=1) / np.sum(occ, axis=1)
        
        _, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t, x_mean, color=COLOR_MASS, label=r'$\langle \hat{x} \rangle$ (Simulated)')
        fh = run.attrs.get("first_hit_full", -1)
        if auto_th:
            t_warmup = p_min * dt
            ax.axvspan(0, t_warmup, color=COLOR_HIGHLIGHT, alpha=0.15, label='Calibration (Warmup)')
            ax.axvline(x=t_warmup, color=COLOR_HIGHLIGHT, ls='-', lw=1, alpha=0.5)
        if fh > 0:
            t_hit = fh * dt
            ax.axvspan(0, t_hit, color=COLOR_CAUSALITY, alpha=0.1, label='Causal Safe')
            ax.axvspan(t_hit, t[-1], color=COLOR_CAUSAL, alpha=0.05, label='Invalid (Boundary)')
            ax.axvline(x=t_hit, color=COLOR_CAUSAL, linestyle='--', alpha=0.7)

        ax.set_title(f"zitterbewegung dynamics n={n_q}", fontsize=11)
        ax.set_xlabel("time")
        ax.set_ylabel("mean position <x>")
        ax.legend(fontsize='x-small', loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"zitter_dynamics_n{n_q}.png", dpi=300)
        plt.close()

    def causal_velocity(self, summary_data):
        """Apparent boundary-arrival velocity v_hit = d_eff / t_hit, with t_hit = p_hit * dt.

        d_eff is the distance from the source to the innermost window site of the side that was
        hit, saved by the runner as ``d_hit_eff``. For even N the source sits at N//2, so the right
        edge is one site closer than the left one; using N/2 for every run (as v1 did) overestimates
        the velocity. Legacy files without ``d_hit_eff`` therefore get NaN instead of a guess.

        This is NOT a Lieb-Robinson velocity: t_hit depends on the threshold and on the
        persistence rule, and at low thresholds the marker fires on the exponential tail ahead of
        the front, which can give apparent velocities above the model's maximum group velocity.
        Runs with no hit (-1 sentinel) or a hit at index 0 have no defined velocity (NaN).
        """
        df = pd.DataFrame(summary_data)
        if "d_hit_eff" in df.columns:
            d_eff = pd.to_numeric(df["d_hit_eff"], errors="coerce")
            d_eff = d_eff.where(d_eff > 0)
        else:
            d_eff = pd.Series(np.nan, index=df.index)
        has_hit = df["first_hit_full"] > 0
        t_hit = (df["first_hit_full"] * df["dt_full"]).where(has_hit)
        out = pd.DataFrame({
            "n_qubits": df["n_qubits"],
            "d_eff": d_eff,
            "t_hit": t_hit,
            "v_hit": d_eff / t_hit,
        })
        return out

    def generate_causal_velocity_table(self, summary_data):
        """Table of apparent boundary-arrival velocities (threshold-dependent; not v_LR)."""
        df_v = self.causal_velocity(summary_data).sort_values("n_qubits")

        latex_table = df_v.to_latex(
            index=False,
            caption=("Apparent boundary-arrival velocity $v_{hit} = d_{eff}/t_{hit}$ "
                     "(threshold-dependent; not a Lieb-Robinson velocity)."),
            label="tab:causal_velocity",
            na_rep="-",
            float_format="%.3f",
        )

        with open(self.output_dir / "causal_velocity.tex", "w") as f:
            f.write(latex_table)


    def generate_stability_table(self, summary_data):
        """
        Generates the causal stability analysis table.
        Focused on n_qubits and the physical safety time (T_safe).
        """
        df = pd.DataFrame(summary_data).sort_values("n_qubits")
        df.to_csv(self.output_dir / "stability_table.csv", index=False)

        m_val = df['m'].iloc[0]
        w_val = df['w'].iloc[0]
        dt_val = df['dt_full'].iloc[0]
        thresh = df['threshold'].iloc[0]

        cols_to_keep = ["n_qubits", "first_hit_full", "T_safe"]
        df_filtered = df[cols_to_keep].copy()

        formatters = {
            "T_safe": "{:.2f}".format,
            "first_hit_full": "{:d}".format,
            "n_qubits": "{:d}".format
        }

        caption_str = (f"Causal Stability Analysis ($m={m_val:.1f}, \\omega={w_val:.1f}, "
                    f"\\Delta t={dt_val:.3f}, \\epsilon={thresh:.1f}$)")

        latex_table = df_filtered.to_latex(
            index=False,
            caption=caption_str,
            label="tab:stability",
            formatters=formatters,
            column_format="rrr",
            escape=False
        )

        with open(self.output_dir / "stability_table.tex", "w") as f:
            f.write(latex_table)

    
    def plot_edge_means(self, run, n_q, dt, fh, auto_th, p_min):
        """Recria o gráfico 'edge means (aligned)' das páginas 3 e 14."""
        t = np.arange(run["occ_full"].shape[0]) * dt
        
        W = self._edge_window(run)
        # Gate input: peak deviation over the edge window, max_i |n_i(t) - n_i(0)|, as stored by the detector.
        peak_left, peak_right = run["metric_full_left"][:], run["metric_full_right"][:]
        # Diagnostic only: mean occupation over the same window.
        left_full = np.mean(run["occ_full"][:, :W], axis=1)
        right_full = np.mean(run["occ_full"][:, -W:], axis=1)
        
        _, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t, peak_left, label=r'left $\Delta$ (peak, used by the gate)', color=COLOR_MASS)
        ax.plot(t, peak_right, label=r'right $\Delta$ (peak, used by the gate)', color=COLOR_CAUSALITY)
        ax.plot(t, left_full, label='left mean (diagnostic)', color=COLOR_MASS, ls=':', alpha=0.6)
        ax.plot(t, right_full, label='right mean (diagnostic)', color=COLOR_CAUSALITY, ls=':', alpha=0.6)
        if "threshold" in run.attrs:
            ax.axhline(float(run.attrs["threshold"]), color='gray', ls='-.', lw=0.8, label=r'threshold $\theta$')
        
        if fh > 0:
            ax.axvline(x=fh*dt, color=COLOR_CAUSAL, linestyle='--', alpha=0.7, label='first_hit_full')
        
        if auto_th:
            ax.axvspan(0, p_min * dt, color=COLOR_HIGHLIGHT, alpha=0.1, label='Warmup')
        ax.set_title(f"edge marker (peak, W={W}) and window mean, n={n_q}", fontsize=11)
        ax.set_xlabel("time")
        ax.set_ylabel("peak deviation / mean occupancy")
        ax.legend(fontsize='x-small')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"edge_means_n{n_q}.png", dpi=300)
        plt.close()

    def plot_fft_comparison(self, run, n_q, dt, fh, p_min, analysis_cfg):
        """Gera os gráficos de FFT 'Full' vs 'Safe' (Páginas 4, 6, 15, 17)."""
        # Spectrum of the gated quantity (peak deviation at the right edge), as the title states.
        signal = run["metric_full_right"][:]
        
        def get_fft(s, delta_t, window_type='hann'):
            s_active = s[p_min:]
            n = len(s_active)
            if n < 2: return np.array([]), np.array([])
            s_centered = sp_sig.detrend(s_active - np.mean(s_active))
            if window_type == 'blackman':
                fft_w = blackman(n)
            else:
                fft_w = hann(n)
            
            s_windowed = s_centered * fft_w
            yf = np.fft.fft(s_windowed)
            xf = np.fft.fftfreq(n, delta_t)[:n//2]
            amplitude = (2.0 / np.sum(fft_w)) * np.abs(yf[0:n//2])
            return xf, amplitude

        xf_f, yf_f = get_fft(signal, dt, analysis_cfg.get("fft_window", "hann"))
        __name__, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xf_f, yf_f, color=COLOR_MASS, label='d_right')
        ax.set_title(rf"FFT(edge_delta_right) [full] n={n_q}")
        ax.set_ylabel("amplitude")
        ax.set_xlabel("frequency (1/time)")
        plt.savefig(self.output_dir / f"fft_full_n{n_q}.png")
        plt.close()

        if (fh >= p_min) or (fh == -1):
            xf_s, yf_s = get_fft(signal[p_min:], dt, analysis_cfg.get("fft_window", "hann"))
            _, ax = plt.subplots(figsize=(6, 4))
            ax.plot(xf_s, yf_s, color=COLOR_MASS)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            ax.set_title(f"FFT(edge_delta_right) [safe] n={n_q}")
            ax.set_ylabel("amplitude")
            ax.set_xlabel("frequency (1/time)")
            plt.savefig(self.output_dir / f"fft_safe_n{n_q}.png")
            plt.close()

    def plot_heatmaps(self, run, n_q):
        """It generates clean heatmaps."""
        datasets = [("occ_full", "raw"), ("occ_rich", "derived")]
        for ds_name, tag in datasets:
            if ds_name in run:
                data = run[ds_name][:]
                _, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(data, aspect='auto', origin='lower', cmap='viridis')
                plt.colorbar(im, label='occupancy')
                ax.set_title(f"{ds_name} ({tag}) n={n_q}")
                ax.set_xlabel("site")
                ax.set_ylabel("step")
                plt.savefig(self.output_dir / f"heatmap_{ds_name}_n{n_q}.png")
                plt.close()

    def plot_causal_invariance(self, run, n_q, dt_f, dt_h, fh_f, auto_th, p_min):
        """
        Visualization of Causal Invariance:
        Separates free (causal) evolution from the region contaminated by reflections.
        """
        dr_f = run["metric_full_right"][:]
        dr_h = run["metric_half_right"][:]
        
        t_f = np.arange(len(dr_f)) * dt_f
        t_h = np.arange(len(dr_h)) * dt_h
        t_safe = fh_f * dt_f
        t_max = max(t_f[-1], t_h[-1])

        _, ax = plt.subplots(figsize=(8, 4.5))

        ax.plot(t_f, dr_f, label=r'$\Delta_R (\Delta t)$', color=COLOR_CAUSALITY, lw=1.5, zorder=3)
        ax.plot(t_h, dr_h, label=r'$\Delta_R (\Delta t/2)$', color=COLOR_MASS, alpha=0.8, ls=':', lw=1.2, zorder=4)
        
        thresh = run.attrs.get("threshold", 1e-5)
        ax.axhline(y=thresh, color='black', ls=':', alpha=0.6, label='Detection Threshold')
        if auto_th:
            t_warmup = p_min * dt_f
            ax.axvspan(0, t_warmup, color=COLOR_HIGHLIGHT, alpha=0.15, label='Calibration (Warmup)')
            ax.axvline(x=t_warmup, color=COLOR_HIGHLIGHT, ls='-', lw=1, alpha=0.5)

        if t_safe > 0:
            ax.axvline(x=t_safe, color='red', ls='--', lw=1, label=r'$t_{safe}$ (Impact)')
            
            ax.axvspan(t_safe, t_max, color=COLOR_CAUSAL, alpha=0.15, label='Reflection Zone (Non-Causal)')
            
        ax.set_title(f"Causal Invariance Analysis (N={n_q})", fontsize=12)
        ax.set_xlabel("Time (t)", fontsize=10)
        ax.set_ylabel(r"Signal Deviation ($\Delta_R$)", fontsize=10)
        
        ax.set_ylim(-thresh * 0.1, max(np.max(dr_f), thresh) * 1.2)
        ax.set_xlim(0, t_max)
        
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        ax.legend(loc='upper left', fontsize='x-small', frameon=True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"causal_invariance_n{n_q}.png", dpi=300)
        plt.close()


    def plot_edge_means_comparison(self, run, n_q, dt, auto_th, p_min):
        """
        Overlays the edge marker (peak deviation, the quantity the gate uses) on the base grid (step Δt)
        and on the refined grid (step Δt/2, sampled at the even steps). Agreement between both curves
        is the resolution stability of the marker.
        """
        t = np.arange(run["occ_full"].shape[0]) * dt
        
        W = self._edge_window(run)
        l_full, r_full = run["metric_full_left"][:], run["metric_full_right"][:]
        l_half, r_half = run["metric_half_left"][:][0::2], run["metric_half_right"][:][0::2]

        _, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t, l_full, label=r'Left ($\Delta t$)', color=COLOR_MASS, ls='-')
        ax.plot(t[:len(l_half)], l_half, label=r'Left ($\Delta t/2$, aligned)', color='tab:cyan', ls='--')
        ax.plot(t, r_full, label=r'Right ($\Delta t$)', color=COLOR_CAUSAL, ls='-')
        ax.plot(t[:len(r_half)], r_half, label=r'Right ($\Delta t/2$, aligned)', color=COLOR_HIGHLIGHT, ls='--')
        if "threshold" in run.attrs:
            ax.axhline(float(run.attrs["threshold"]), color='gray', ls='-.', lw=0.8, label=r'threshold $\theta$')
        if auto_th:
            ax.axvspan(0, p_min * dt, color=COLOR_HIGHLIGHT, alpha=0.1, label='Warmup')
        ax.set_title(f"Edge marker (peak, W={W}): resolution stability (N={n_q})")
        ax.set_xlabel("Time")
        ax.set_ylabel(r"Peak deviation $\Delta$")
        ax.legend(fontsize='xx-small', ncol=2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"edge_means_comparison_n{n_q}.png", dpi=300)
        plt.close()

    def run(self, cfg):
        self.data_manager = DataManager.from_h5_file(self.data_file)
        summary_stats = []
        detector_cfg = parse_detector_cfg(cfg)
        analysis_cfg = parse_analysis_cfg(cfg)
        auto_th = detector_cfg.get("auto_threshold", False)

        with self.data_manager.session() as reader:
            runs = reader.file["dirac_simulation/runs"]

            for r_id in runs.keys():
                r = runs[r_id]
                attrs = r.attrs
                n_q, dt, dt_half, fh = attrs["n_qubits"], attrs["dt_full"], attrs["dt_half"], attrs["first_hit_full"]

                summary_stats.append({
                    "m": round(attrs["m"], 2),
                    "w": round(attrs["w"], 2),
                    "threshold": attrs["threshold"],
                    "max_steps_full": attrs["max_steps_full"],
                    "n_qubits": n_q,
                    "dt_full":dt,
                    "first_hit_full": fh,
                    "dt_half": dt_half,
                    "d_hit_eff": float(attrs.get("d_hit_eff", np.nan)),  # absent in legacy files
                    "T_safe": fh * dt if fh > 0 else 0
                })
                validity_cfg = parse_validity_cfg(cfg)
                p_min = validity_cfg['p_min']
                self.plot_calibration_floor_5sigma(r, n_q)
                self.plot_heatmaps(r, n_q)
                self.plot_fft_comparison(r, n_q, dt, fh, p_min, analysis_cfg)
                self.plot_validity_gating_summary(r,n_q, dt, p_min, auto_th, cfg)
                self.plot_richardson_error(r, n_q, dt)
                self.plot_edge_means(r, n_q, dt, fh, auto_th, p_min)
                self.plot_edge_means_comparison(r, n_q, dt, auto_th, p_min)
                self.plot_causal_invariance(
                    r, n_q, 
                    dt, dt_half, 
                    fh, auto_th, p_min
                )
                self.plot_zitterbewegung_analysis(r,n_q,dt, auto_th, p_min)
                self.plot_lieb_robinson_cone(r, n_q)

        self.generate_stability_table(summary_stats)
        self.generate_admissibility_table(summary_stats, parse_validity_cfg(cfg)['p_min'])
        self.generate_causal_velocity_table(summary_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("src/configs/dirac_simulation.yaml"))
    args = parser.parse_args()
    cfg = load_dirac_simulation_yaml(args.config)
    ScientificReport(args.data_file).run(cfg)
