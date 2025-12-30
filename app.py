import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import datetime

st.set_page_config(page_title="Gráficos I-MR - Carta de controle", layout="wide")
st.title("Gráficos I-MR - Carta de controle")

# --- FUNÇÕES AUXILIARES ---
def check_nelson_rule_2(series, mean):
    sinal = np.sign(series - mean)
    return sinal.rolling(window=9, min_periods=9).apply(
        lambda x: all(~np.isnan(x)) and (all(val > 0 for val in x) or all(val < 0 for val in x)),
        raw=True
    ).fillna(False)

def check_nelson_rule_3(series):
    diffs = series.diff()
    return diffs.rolling(window=6, min_periods=6).apply(
        lambda x: all(~np.isnan(x)) and (all(val > 0 for val in x) or all(val < 0 for val in x)),
        raw=True
    ).fillna(False)

def check_nelson_rule_4(series):
    diffs = series.diff()
    def _alterna_robust(x):
        if np.any(np.isnan(x)):
            return False
        signs = (x > 0)
        return all(signs[i] != signs[i + 1] for i in range(len(signs) - 1))
    return diffs.rolling(window=14, min_periods=14).apply(_alterna_robust, raw=True).fillna(False)

def mark_nelson_violations(row, prefix=''):
    codigos = []
    if row.get(f'{prefix}viol_nelson_1', False): codigos.append('1')
    if row.get(f'{prefix}viol_nelson_2', False): codigos.append('2')
    if row.get(f'{prefix}viol_nelson_3', False): codigos.append('3')
    if row.get(f'{prefix}viol_nelson_4', False): codigos.append('4')
    return ','.join(codigos)

def add_stats_text(ax, df, ucl_val, lcl_val, mean_val, dp_val=None, prefix='', decimal_places=2):
    last_date = df['Data'].iloc[-1]
    ax.text(last_date, ucl_val, f'{prefix}UCL = {ucl_val:.{decimal_places}f}', color='red', va='bottom', ha='right')
    ax.text(last_date, lcl_val, f'{prefix}LCL = {lcl_val:.{decimal_places}f}', color='red', va='top', ha='right')
    ax.text(last_date, mean_val, f'{prefix}Média = {mean_val:.{decimal_places}f}', color='black', va='bottom', ha='right')
    if dp_val is not None:
        ax.plot([], [], color='darkgreen', linestyle='-', label=f'DP = {dp_val:.5f}')

def add_nelson_plots(ax, df_violacoes, y_col, plot_label='Violação Nelson', text_offset=0.03):
    if not df_violacoes.empty:
        sns.scatterplot(data=df_violacoes, x='Data', y=y_col, ax=ax, color='red', label=plot_label, zorder=10, s=100, marker='X')
        for i, row in df_violacoes.iterrows():
            viol_text = row.get('violacoes_nelson', '') or row.get('cond_violacoes_nelson', '')
            if viol_text:
                ax.text(row['Data'], row[y_col] + text_offset, viol_text, color='red', fontsize=8, ha='center')

def add_reference_lines(ax, refs, initial_label=None, color='gray', linestyle='--'):
    for i, ref in enumerate(refs):
        if ref is not None:
            label = f'{ref} {un_med}' if (initial_label and i == 0) else None
            ax.axhline(ref, color=color, linestyle=linestyle, label=label)

# --- INTERFACE ---
st.subheader("Parâmetros de Controle")
col1, col2 = st.columns(2)

with col1:
    parametro = st.text_input("Nome do Parâmetro", value="Condutividade")
    un_med = st.text_input("Unidade de medida", value="µS/cm")
    alerta_txt = st.text_input("Limite de Alerta", value="")
    acao_txt = st.text_input("Limite de Ação", value="")
    especificacao_txt = st.text_input("Limite de Especificação", value="")   
    escala_min_txt = st.text_input("Escala Mínima", value="")
    escala_max_txt = st.text_input("Escala Máxima", value="")
    intervalo_escala_txt = st.text_input("Intervalo da Escala", value="")

def to_float(val):
    try: return float(val.replace(",", "."))
    except: return None

alerta, acao, espec = to_float(alerta_txt), to_float(acao_txt), to_float(especificacao_txt)
e_min, e_max, e_int = to_float(escala_min_txt), to_float(escala_max_txt), to_float(intervalo_escala_txt)

st.markdown("---")
uploaded_file = st.file_uploader("Upload do arquivo CSV (Separador ';')", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=';', encoding='latin1')
        
        # Validação de Colunas
        if not all(c in df.columns for c in ['Data', 'Ponto', 'Resultado']):
            st.error(f"⚠️ Erro: Colunas esperadas não encontradas. O arquivo contém: {list(df.columns)}")
            st.stop()

        # Conversão
        df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')
        if df['Resultado'].dtype == object:
            df['Resultado'] = df['Resultado'].str.replace(',', '.', regex=False).astype(float)
        df['Resultado'] = pd.to_numeric(df['Resultado'], errors='coerce')
        df.dropna(subset=['Data', 'Resultado'], inplace=True)
        df = df.sort_values(by=['Ponto', 'Data'])

        ponto = st.selectbox('Selecione o ponto:', df['Ponto'].unique())
        df_p = df[df['Ponto'] == ponto].copy()

        if len(df_p) < 2:
            st.warning("Pontos insuficientes para cálculo.")
            st.stop()

        # Estatísticas
        df_p['MR'] = df_p['Resultado'].diff().abs()
        mr_m = df_p['MR'].mean()
        std_m = mr_m / 1.128
        media_r = df_p['Resultado'].mean()
        ucl_r, lcl_r = media_r + 3*std_m, media_r - 3*std_m
        ucl_mr = 3.267 * mr_m

        # Violações
        df_p['c_v1'] = (df_p['Resultado'] > ucl_r) | (df_p['Resultado'] < lcl_r)
        df_p['c_v2'] = check_nelson_rule_2(df_p['Resultado'], media_r)
        df_p['cond_violacoes_nelson'] = df_p.apply(lambda r: mark_nelson_violations(r, 'c_'), axis=1)

        # Gráficos
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        sns.lineplot(data=df_p, x='Data', y='Resultado', marker='o', ax=axes[0])
        axes[0].axhline(media_r, color='blue', ls='--')
        add_reference_lines(axes[0], [alerta, acao, espec], initial_label=True)
        
        sns.lineplot(data=df_p, x='Data', y='MR', marker='o', color='orange', ax=axes[1])
        axes[1].axhline(mr_m, color='green', ls='--')

        # Escala Manual
        if all(v is not None for v in [e_min, e_max, e_int]) and e_max > e_min:
            ticks = np.arange(e_min, e_max + 0.1, e_int)
            axes[0].set_yticks(ticks)
            axes[1].set_yticks(ticks)

        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        plt.xticks(rotation=45)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erro ao processar: {e}")
else:
    st.info("Por favor, faça o upload do arquivo.")
