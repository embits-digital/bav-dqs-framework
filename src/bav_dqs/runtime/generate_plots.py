import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftfreq
from pathlib import Path
import argparse

COLOR_MASS = 'tab:blue'
COLOR_CAUSAL = 'tab:red'
COLOR_BASELINE = 'tab:gray'
COLOR_HIGHLIGHT = 'tab:orange'

class ScientificReport:
    def __init__(self, data-file: Path):
        self.data-file = data-file
        self.output_dir = data-file.parent / "scientific_report_v2026"
        self.output_dir.mkdir(exist_ok=True)
        # Estilo estrito para journals (Phys. Rev. Style)
        plt.rcParams.update({
            'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.2,
            'lines.linewidth': 1.2, 'figure.dpi': 300
        })

    def plot_lieb_robinson_cone(self, run, n_q, dt):
        """
        [NOVO] Visualiza o Cone de Luz de Lieb-Robinson.
        Prova que a informação (ZZ) viaja à frente da massa (Z).
        """
        if "correlations" not in run:
            return

        corr_data = np.abs(run["correlations"][:])
        occ_data = np.abs(run["history"][:])
        steps, sites = occ_data.shape
        t = np.arange(steps) * dt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        # Heatmap de Ocupação (Massa)
        im1 = ax1.imshow(occ_data, aspect='auto', origin='lower', cmap='YlGnBu')
        ax1.set_title(f"Mass Transport (Z) N={n_q}", color=COLOR_MASS)
        ax1.set_xlabel("Site Index")
        ax1.set_ylabel("Physical Time")
        plt.colorbar(im1, ax=ax1, label=r'$\langle Z_j \rangle$')

        # Heatmap de Correlação (Informação/Causalidade)
        im2 = ax2.imshow(corr_data, aspect='auto', origin='lower', cmap='YlOrRd')
        ax2.set_title(f"Information Spread (ZZ) N={n_q}", color=COLOR_CAUSAL)
        ax2.set_xlabel("Bond Index (ref=center)")
        plt.colorbar(im2, ax=ax2, label=r'$| \langle Z_{ref} Z_j \rangle_c |$')

        plt.tight_layout()
        plt.savefig(self.output_dir / f"causal_cone_comparison_n{n_q}.png")
        plt.close()

    def plot_calibration_floor_5sigma(self, run, n_q):
        """
        [NOVO] Demonstra o rigor da calibração 5-sigma.
        Blindagem direta contra a crítica de 'threshold arbitrário'.
        """
        # Extraímos os thresholds salvos nos atributos do run
        thr_z = run.attrs.get("threshold_occupancy", 0.0)
        thr_zz = run.attrs.get("threshold_correlation", 0.0)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        labels = ['Occupancy (Z)', 'Correlation (ZZ)']
        values = [thr_z, thr_zz]
        
        bars = ax.bar(labels, values, color=['tab:blue', 'tab:red'], alpha=0.7)
        ax.axhline(y=1e-3, color='black', ls='--', alpha=0.3, label='Standard Heuristic')
        
        ax.set_yscale('log')
        ax.set_title(rf"Adaptive 5-$\sigma$ Detection Floor (N={n_q})")
        ax.set_ylabel("Threshold Amplitude (log)")
        ax.legend(fontsize='small')
        
        # Adiciona labels de valor nas barras
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2e}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / f"calibration_5sigma_n{n_q}.png")
        plt.close()

    def plot_richardson_error(self, run, n_q, dt):
        """Comprova a convergência de Richardson na borda."""
        if "occ_rich" not in run:
            return

        t = np.arange(run["occ_full"].shape[0]) * dt
        # Erro no 'vácuo' da borda
        edge_full = np.abs(np.mean(run["occ_full"][:, :2], axis=1))
        edge_rich = np.abs(np.mean(run["occ_rich"][:, :2], axis=1))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(t, edge_full, label=r'Error $O(\Delta t)$', color=COLOR_BASELINE, alpha=0.6)
        ax.semilogy(t[:len(edge_rich)], edge_rich, label=r'Error $O(\Delta t^2)$', color=COLOR_CAUSAL, lw=1.5)
        
        ax.set_title(rf"Numerical Error Suppression (N={n_q})")
        ax.set_ylabel("Absolute Deviation (log)")
        ax.set_xlabel("Physical Time")
        ax.legend(fontsize='small')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"richardson_error_n{n_q}.png")
        plt.close()

    def plot_validity_gating_summary(self, run, n_q, dt):
        """
        [NOVO] O 'Gating' de Admissibilidade.
        Mostra o momento exato em que a simulação deixa de ser válida.
        """
        t = np.arange(run["history"].shape[0]) * dt
        dl = run["dL"][:] # Delta Left da Ocupação
        
        # Simulando o sinal de correlação se disponível
        corr_max = np.max(np.abs(run["correlations"][:]), axis=1) if "correlations" in run else None

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, dl, label='Physical Edge Signal (Z)', color=COLOR_MASS)
        if corr_max is not None:
            ax.plot(t, corr_max, label='Causal Edge Signal (ZZ)', color=COLOR_CAUSAL, alpha=0.6)

        # Linhas de Hit
        hit_z = run.attrs.get("first_hit_full", -1)
        hit_zz = run.attrs.get("first_hit_half", -1)

        if hit_z > 0:
            ax.axvline(x=hit_z * dt, color=COLOR_MASS, ls='--', label='Physical Limit')
        if hit_zz > 0:
            ax.axvline(x=hit_zz * dt, color=COLOR_CAUSAL, ls=':', label='Causal Limit (LR)')

        ax.set_title(f"Inferential Admissibility Gating (N={n_q})")
        ax.set_xlabel("Time")
        ax.set_ylabel(r"Signal Deviation ($\Delta$)")
        ax.legend(loc='upper left', fontsize='x-small')
        
        # Shading da zona inválida
        first_hit = min(h for h in [hit_z, hit_zz] if h > 0)
        ax.axvspan(first_hit * dt, t[-1], color='gray', alpha=0.1, label='Invalid Domain')

        plt.tight_layout()
        plt.savefig(self.output_dir / f"validity_gating_n{n_q}.png")
        plt.close()

    def generate_admissibility_table(self, summary_data):
        """
        Gera a Tabela I: Resultados de admissibilidade inferencial (Dirac/Ocupação).
        Focada em validar se o tamanho da rede (N) suporta o critério L_min.
        """
        df = pd.DataFrame(summary_data)
        
        # 1. Ordenação e Cálculos
        df = df.sort_values(["n_qubits"])
        df["L_safe"] = df["first_hit_full"] - 1
        df["first_hit_full"] = df["first_hit_full"].apply(lambda x: "-" if x <= 0 else x)
        df["Admissible"] = df["L_safe"].apply(lambda x: "PASS" if x >= 32 else ("NO HIT" if x <= 0 else "FAIL"))
        df["L_safe"] = df["L_safe"].apply(lambda x: "-" if x <= 0 else x)

        # 2. Arredondamento Global para o CLI e CSV
        # Seleciona apenas colunas numéricas para não dar erro em colunas de texto
        numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
        df[numeric_cols] = df[numeric_cols].round(2)

        # Print para o CLI (Log) formatado
        print("\n[LOG] Admissibility Data:")
        print(df.to_string(index=False))

        # 3. Preparação para o LaTeX
        # Renomear colunas para o padrão do LaTeX
        df_latex = df.drop(columns=['T_safe', 'max_steps_full', 'm', 'w', 'threshold', 'dt_half', 'dt_full']).rename(columns={
            "n_qubits": "Lattice ($N$)",
            "threshold": "Threshold ($\\theta$)",
            "first_hit_full": "First Hit ($p_{\\text{hit}}$)",
            "L_safe": "$L_{\\text{safe}}$",
            "Admissible": r"($L_{\text{safe}} \geq 32$)"
        })

        # Selecionar apenas as colunas que devem aparecer na tabela final (opcional)
        # df_latex = df_latex]

        # 4. Correção da Legenda (Extraindo valores escalares com .iloc[0])
        m_val = df['m'].iloc[0]
        w_val = df['w'].iloc[0]
        p_val = df["max_steps_full"].iloc[0]
        dt_val = df['dt_full'].iloc[0]
        theta_val = df['threshold'].iloc[0]

        caption = (
            r"Boundary detection and inferential admissibility. Use case is a 1D Dirac lattice simulation "
            rf"($m = {m_val:.2f}, w = {w_val:.2f}, \Delta t = {dt_val:.2f}, \theta = {theta_val:.2f}, p_{{\max}} = {p_val}$)."
            r"$L_{\text{safe}}$ represents the temporal support (steps) before causality loss. "
            r"The failure of the $N = 4$ configuration to meet the $L_{\min} = 32$ criterion "
            r"demonstrates the framework's capacity to preclude inference in undersized domains."
        )

        # 5. Exportação
        df_latex.to_csv(self.output_dir / "admissibility_table.csv", index=False)
        
        with open(self.output_dir / "admissibility_table.tex", "w") as f:
            # float_format="%.2f" garante que o LaTeX não ignore o arredondamento
            f.write(df_latex.to_latex(
                index=False, 
                caption=caption, 
                label="tab:results_summary", 
                escape=False,
                column_format="ccccccc", # Ajuste conforme o número de colunas
                float_format="%.3f"
            ))
            
        print(f"[INFO] Tabela I (Admissibilidade) gerada em: {self.output_dir}")

    def plot_zitterbewegung_analysis(self, run, n_q, dt):
        """
        Visualiza a oscilação de Zitterbewegung extraída da ocupação média.
        fh: first_hit_full (proveniente dos atributos do run)
        """
        occ = run["occ_full"][:]
        steps, sites = occ.shape
        t = np.arange(steps) * dt
        
        # Calcula a posição média <x> em cada passo de tempo
        # x_mean = sum(site_index * occupancy) / sum(occupancy)
        site_indices = np.arange(sites)
        x_mean = np.sum(occ * site_indices, axis=1) / np.sum(occ, axis=1)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t, x_mean, color='tab:blue', label=r'$\langle \hat{x} \rangle$ (Simulated)')
        fh = run.attrs.get("first_hit_full", -1)

        if fh > 0:
            t_hit = fh * dt
            # Destaca a zona segura e a zona corrompida por bordas
            ax.axvspan(0, t_hit, color='tab:green', alpha=0.1, label='Causal Safe')
            ax.axvspan(t_hit, t[-1], color='tab:red', alpha=0.05, label='Invalid (Boundary)')
            ax.axvline(x=t_hit, color='tab:red', linestyle='--', alpha=0.7)

        ax.set_title(f"zitterbewegung dynamics n={n_q}", fontsize=11)
        ax.set_xlabel("time")
        ax.set_ylabel("mean position <x>")
        ax.legend(fontsize='x-small', loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"zitter_dynamics_n{n_q}.png", dpi=300)
        plt.close()

    def generate_causal_velocity_table(self, summary_data):
        """
        [NOVO] Tabela comparativa de velocidade de informação vs massa.
        Extrai a 'Velocidade de Lieb-Robinson' medida pelo framework.
        """
        df = pd.DataFrame(summary_data)
        
        # Cálculo da velocidade operacional v = (distância até a borda) / t_hit
        # Distância assumida como N/2 para pacotes centralizados
        df["v_causal"] = (df["n_qubits"] / 2) / (df["first_hit_full"] * df["dt_full"])
        
        # Filtra e formata para LaTeX
        df_v = df[["n_qubits", "v_causal"]].copy()
        df_v = df_v[df_v["v_causal"] < np.inf] # Remove NO HIT
        
        latex_table = df_v.to_latex(
            index=False,
            caption="Measured Operational Causal Velocity ($v_{LR}$) across different lattice sizes.",
            label="tab:causal_velocity",
            formatters={"v_causal": "{:.3f}".format}
        )
        
        with open(self.output_dir / "causal_velocity.tex", "w") as f:
            f.write(latex_table)


    def generate_stability_table(self, summary_data):
        """
        Gera a Tabela II: Análise de estabilidade causal.
        Focada em n_qubits e no tempo físico de segurança (T_safe).
        """
        df = pd.DataFrame(summary_data).sort_values("n_qubits")
        df.to_csv(self.output_dir / "stability_table.csv", index=False)

        # 1. Extrair valores constantes para o caption antes de remover as colunas
        # Assumindo que os valores são os mesmos em todas as linhas
        m_val = df['m'].iloc[0]
        w_val = df['w'].iloc[0]
        dt_val = df['dt_full'].iloc[0]
        thresh = df['threshold'].iloc[0]

        # 2. Filtrar colunas: Mantemos apenas o que é relevante para a comparação
        # Ajuste a lista abaixo se quiser manter 'max_steps_full' ou outras.
        cols_to_keep = ["n_qubits", "first_hit_full", "T_safe"]
        df_filtered = df[cols_to_keep].copy()

        # 3. Formatação: 2 casas para o tempo, 0 para contagens/qubits
        formatters = {
            "T_safe": "{:.2f}".format,
            "first_hit_full": "{:d}".format,
            "n_qubits": "{:d}".format
        }

        # Gerar o caption dinâmico com os parâmetros fixos
        caption_str = (f"Causal Stability Analysis ($m={m_val:.1f}, \\omega={w_val:.1f}, "
                    f"\\Delta t={dt_val:.3f}, \\epsilon={thresh:.1f}$)")

        latex_table = df_filtered.to_latex(
            index=False,
            caption=caption_str,
            label="tab:stability",
            formatters=formatters,
            column_format="rrr", # Alinhamento à direita
            escape=False        # Permite símbolos LaTeX no caption
        )

        with open(self.output_dir / "stability_table.tex", "w") as f:
            f.write(latex_table)
        
        print("[INFO] Tabela II (Estabilidade) gerada com sucesso.")

    
    def plot_edge_means(self, run, n_q, dt, fh):
        """Recria o gráfico 'edge means (aligned)' das páginas 3 e 14."""
        t = np.arange(run["occ_full"].shape[0]) * dt
        
        # O PDF usa a média dos sites de borda para suavizar a física
        # e comparar as resoluções full e half_aligned
        left_full = np.mean(run["occ_full"][:, :2], axis=1)
        right_full = np.mean(run["occ_full"][:, -2:], axis=1)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t, left_full, label='left_full', color='tab:blue')
        ax.plot(t, right_full, label='right_full', color='tab:green')
        
        if fh > 0:
            ax.axvline(x=fh*dt, color='tab:red', linestyle='--', alpha=0.7, label='first_hit_full')
        
        ax.set_title(f"edge means (aligned) n={n_q}", fontsize=11)
        ax.set_xlabel("time")
        ax.set_ylabel("mean occupancy")
        ax.legend(fontsize='x-small')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"edge_means_n{n_q}.png", dpi=300)
        plt.close()

    def plot_fft_comparison(self, run, n_q, dt, fh):
        """Gera os gráficos de FFT 'Full' vs 'Safe' (Páginas 4, 6, 15, 17)."""
        signal = np.mean(run["occ_full"][:, -2:], axis=1)
        
        def get_fft(s, delta_t):
            n = len(s)
            yf = fft(s - np.mean(s))
            xf = fftfreq(n, delta_t)[:n//2]
            return xf, 2.0/n * np.abs(yf[0:n//2])

        # 1. FFT FULL (Sinal poluído - Escala 10^-3)
        xf_f, yf_f = get_fft(signal, dt)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xf_f, yf_f, color='tab:blue', label='d_right')
        ax.set_title(rf"FFT(edge_delta_right) [full] n={n_q}")
        ax.set_ylabel("amplitude")
        ax.set_xlabel("frequency (1/time)")
        plt.savefig(self.output_dir / f"fft_full_n{n_q}.png")
        plt.close()

        # 2. FFT SAFE (Sinal limpo - Escala 10^-7)
        if fh > 20:
            xf_s, yf_s = get_fft(signal[:fh], dt)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(xf_s, yf_s, color='tab:blue')
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            ax.set_title(f"FFT(edge_delta_right) [safe] n={n_q}")
            ax.set_ylabel("amplitude")
            ax.set_xlabel("frequency (1/time)")
            plt.savefig(self.output_dir / f"fft_safe_n{n_q}.png")
            plt.close()

    def plot_heatmaps(self, run, n_q):
        """Gera os mapas de calor limpos (Páginas 10-12)."""
        datasets = [("occ_full", "raw"), ("occ_rich", "derived")]
        for ds_name, tag in datasets:
            if ds_name in run:
                data = run[ds_name][:]
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(data, aspect='auto', origin='lower', cmap='viridis')
                plt.colorbar(im, label='occupancy')
                ax.set_title(f"{ds_name} ({tag}) n={n_q}")
                ax.set_xlabel("site")
                ax.set_ylabel("step")
                plt.savefig(self.output_dir / f"heatmap_{ds_name}_n{n_q}.png")
                plt.close()

    def plot_causal_invariance(self, run, n_q, dt_f, dt_h, fh_f, fh_h):
        """Refatorado: Prova de Invariância Causal com zoom no Threshold."""
        dr_f = run["metric_full_right"][:]
        dr_h = run["metric_half_right"][:]
        t_f = np.arange(len(dr_f)) * dt_f
        t_h = np.arange(len(dr_h)) * dt_h

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t_f, dr_f, label=r'$\Delta_R (\Delta t)$', color='tab:blue')
        ax.plot(t_h, dr_h, label=r'$\Delta_R (\Delta t/2)$', color='tab:orange', alpha=0.7)
        
        thresh = run.attrs.get("threshold", 1e-5)
        ax.axhline(y=thresh, color='black', ls=':', label='Threshold')
        
        if fh_f > 0:
            ax.axvline(x=fh_f * dt_f, color='tab:blue', ls='--', label=r'$t_{safe}$')

        ax.set_title(f"Causal Invariance (N={n_q})")
        ax.set_ylim(-thresh*0.5, thresh*5) # Foco na detecção
        ax.legend(loc='upper left', fontsize='xx-small')
        plt.savefig(self.output_dir / f"causal_invariance_n{n_q}.png")
        plt.close()

    def plot_edge_means_comparison(self, run, n_q, dt):
        """
        Recria 'edge means (aligned)' sobrepondo full e half_aligned (Pág 3 e 14).
        Prova que o refinamento de Richardson preserva a média física.
        """
        t = np.arange(run["occ_full"].shape[0]) * dt
        
        # Média dos sites de borda (Esquerda e Direita)
        l_full = np.mean(run["occ_full"][:, :2], axis=1)
        r_full = np.mean(run["occ_full"][:, -2:], axis=1)
        
        # Half aligned (downsampled para bater com a grade full)
        # Assumindo que seu script de simulação já salvou o occ_half_aligned
        occ_half = run["occ_half"][:]
        occ_half_aligned = occ_half[0::2] # Downsample manual se não estiver no HDF5
        l_half = np.mean(occ_half_aligned[:, :2], axis=1)
        r_half = np.mean(occ_half_aligned[:, -2:], axis=1)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(t, l_full, label='Left (Full)', color='tab:blue', ls='-')
        ax.plot(t[:len(l_half)], l_half, label='Left (Half Aligned)', color='tab:cyan', ls='--')
        ax.plot(t, r_full, label='Right (Full)', color='tab:red', ls='-')
        ax.plot(t[:len(r_half)], r_half, label='Right (Half Aligned)', color='tab:orange', ls='--')

        ax.set_title(f"Edge Means Comparison: Resolution Stability (N={n_q})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Mean Occupancy")
        ax.legend(fontsize='xx-small', ncol=2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"edge_means_comparison_n{n_q}.png", dpi=300)
        plt.close()

    def run(self):
        summary_stats = []
        with h5py.File(self.data-file, "r") as f:
            runs = f["dirac_simulation/runs"]
            def print_structure(name, obj):
                # Recuo visual baseado na profundidade da estrutura
                indent = '  ' * name.count('/')
                print(f"{indent}{name} ({type(obj).__name__})")
                
                # Imprime os Atributos (Metadados)
                for key, val in obj.attrs.items():
                    print(f"{indent}    [Attr] {key}: {val}")
            
            runs.visititems(print_structure)

            for r_id in runs.keys():
                r = runs[r_id]
                attrs = r.attrs

                summary_stats.append({
                    "m": round(attrs["m"], 2),
                    "w": round(attrs["w"], 2),
                    "threshold": attrs["threshold"],
                    "max_steps_full": attrs["max_steps_full"],
                    "n_qubits": attrs["n_qubits"],  # Certifique-se de que a chave é exatamente "n_qubits"
                    "dt_full": attrs["dt_full"],
                    "first_hit_full": attrs["first_hit_full"],
                    "dt_half": attrs["dt_half"],
                    "T_safe": attrs["first_hit_full"] * attrs["dt_full"] if attrs["first_hit_full"] > 0 else 0
                })

                n_q, dt, fh = attrs["n_qubits"], attrs["dt_full"], attrs["first_hit_full"]
                self.plot_causal_invariance(r, attrs["n_qubits"], attrs["dt_full"], attrs["dt_half"], attrs["first_hit_full"], attrs["first_hit_half"])
                self.plot_richardson_error(r, attrs["n_qubits"], attrs["dt_full"])
                self.plot_edge_means(r, n_q, dt, fh)
                self.plot_fft_comparison(r, n_q, dt, fh)
                self.plot_heatmaps(r, n_q)
                self.plot_causal_invariance(
                    r, attrs["n_qubits"], 
                    attrs["dt_full"], attrs["dt_half"], 
                    attrs["first_hit_full"], attrs["first_hit_half"]
                )
                self.plot_edge_means_comparison(r, attrs["n_qubits"], attrs["dt_full"])
                self.plot_zitterbewegung_analysis(r,n_q,dt)
                print(f"Sucesso: Imagens de validação científica geradas para N={attrs['n_qubits']}")

        self.generate_stability_table(summary_stats)
        self.generate_admissibility_table(summary_stats)
        self.generate_causal_velocity_table(summary_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=Path, required=True)
    args = parser.parse_args()
    ScientificReport(args.data-file).run()
