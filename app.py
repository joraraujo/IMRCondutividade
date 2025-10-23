import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import datetime
import io

st.set_page_config(page_title="Gráficos I-MR - Carta de controle", layout="wide")
st.title("Gráficos I-MR - Carta de controle")

# --- FUNÇÕES AUXILIARES (copiadas do script original) ---
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
        sns.scatterplot(
            data=df_violacoes,
            x='Data',
            y=y_col,
            ax=ax,
            color='red',
            label=plot_label,
            zorder=10,
            s=100,
            marker='X'
        )
        for i, row in df_violacoes.iterrows():
            viol_text = row.get('violacoes_nelson', '') or row.get('cond_violacoes_nelson', '')
            if viol_text:
                ax.text(
                    row['Data'],
                    row[y_col] + text_offset,
                    viol_text,
                    color='red',
                    fontsize=8,
                    ha='center'
                )

def add_reference_lines(ax, refs, initial_label=None, color='gray', linestyle='--'):
    for i, ref in enumerate(refs):
        # pular refs não informadas (None)
        if ref is None:
            continue
        label = f'{ref} {um}' if (initial_label and i == 0) else None
        ax.axhline(ref, color=color, linestyle=linestyle, label=label)

# --- INTERFACE STREAMLIT ---

st.subheader("Parâmetros de Controle")

col1, col2 = st.columns(2)

with col1:
    parametro = st.text_input("Nome do Parâmetro", value="")
    um = st.text_input("Unidade de medida", value="")
    alerta_txt = st.text_input("Limite de Alerta (usar ponto para separar casas decimais)", value="")
    acao_txt = st.text_input("Limite de Ação (usar ponto para separar casas decimais)", value="")
    especificacao_txt = st.text_input("Limite de Especificação (usar ponto para separar casas decimais)", value="")   
    escala_min = st.text_input("Escala Mínima do Gráfico (Eixo y)", value="")
    escala_max = st.text_input("Escala Máxima do Gráfico (Eixo y)", value="")
    intervalo_escala = st.text_input("Intervalo da Escala do Gráfico (Eixo y)", value="")

# Conversão dos valores digitados para float 
def to_float_or_none(value):
    try:
        return float(value.replace(",", "."))
    except:
        return None

# Conversão dos textos numéricos para floats (ou None)
alerta = to_float_or_none(alerta_txt)
acao = to_float_or_none(acao_txt)
especificacao = to_float_or_none(especificacao_txt)
escala_min_val = to_float_or_none(escala_min)
escala_max_val = to_float_or_none(escala_max)
intervalo_escala_val = to_float_or_none(intervalo_escala)

st.markdown("Faça upload do arquivo de dados (.csv) para começar.")
uploaded_file = st.file_uploader("Arquivo CSV de Condutividade", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';', encoding='latin1')
    # Conversão de datas e tipos
    if not pd.api.types.is_datetime64_any_dtype(df['Data']):
        df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')
    if df['Resultado'].dtype == object:
        df['Resultado'] = df['Resultado'].str.replace(',', '.', regex=False).astype(float)
    df['Resultado'] = pd.to_numeric(df['Resultado'], errors='coerce')
    df = df.sort_values(by=['Ponto', 'Data'])

    # Seleção de ponto
    pontos = df['Ponto'].unique()
    ponto = st.selectbox('Selecione o ponto para visualizar:', pontos)
    df_ponto = df[df['Ponto'] == ponto].copy().sort_values('Data')

    # Limpeza de NaNs
    initial_rows = len(df_ponto)
    df_ponto.dropna(subset=['Data', 'Resultado'], inplace=True)
    if len(df_ponto) < 2:
        st.error(f"Dados insuficientes para o ponto '{ponto}' após limpeza. São necessários pelo menos 2 pontos.")
        st.stop()

    # Cálculo de MR
    df_ponto['MR'] = df_ponto['Resultado'].diff().abs()

    d2_constant_for_n2 = 1.128
    mr_media = df_ponto['MR'].dropna().mean()
    if pd.isna(mr_media):
        st.error(f"MR média não pôde ser calculada para o ponto '{ponto}'. Verifique os dados de Condutividade.")
        st.stop()

    std_condutividade_minitab = mr_media / d2_constant_for_n2
    media_condutividade = df_ponto['Resultado'].mean()
    ucl_condutividade = media_condutividade + 3 * std_condutividade_minitab
    lcl_condutividade = media_condutividade - 3 * std_condutividade_minitab

    D4 = 3.267
    D3 = 0
    ucl_mr = D4 * mr_media
    lcl_mr = D3 * mr_media

    # Regras de Nelson para (I-Chart)
    df_ponto['cond_viol_nelson_1'] = (df_ponto['Resultado'] > ucl_condutividade) | (df_ponto['Resultado'] < lcl_condutividade)
    df_ponto['cond_viol_nelson_2'] = check_nelson_rule_2(df_ponto['Resultado'], media_condutividade)
    df_ponto['cond_viol_nelson_3'] = check_nelson_rule_3(df_ponto['Resultado'])
    df_ponto['cond_viol_nelson_4'] = check_nelson_rule_4(df_ponto['Resultado'])
    df_ponto['cond_violacoes_nelson'] = df_ponto.apply(lambda row: mark_nelson_violations(row, prefix='cond_'), axis=1)

    # Regras de Nelson para MR
    df_ponto['viol_nelson_1'] = (df_ponto['MR'] > ucl_mr) | (df_ponto['MR'] < lcl_mr)
    df_ponto['viol_nelson_2'] = check_nelson_rule_2(df_ponto['MR'], mr_media)
    df_ponto['viol_nelson_3'] = check_nelson_rule_3(df_ponto['MR'])
    df_ponto['viol_nelson_4'] = check_nelson_rule_4(df_ponto['MR'])
    df_ponto['violacoes_nelson'] = df_ponto.apply(lambda row: mark_nelson_violations(row), axis=1)

    # --- GRÁFICOS ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Gráfico 1: Condutividade
    sns.lineplot(data=df_ponto, x='Data', y='Resultado', marker='o', ax=axes[0])
    axes[0].axhline(media_condutividade, color='blue', linestyle='--', label='Média')
    axes[0].axhline(ucl_condutividade, color='red', linestyle='--', label='UCL')
    axes[0].axhline(lcl_condutividade, color='red', linestyle='--', label='LCL')
    add_reference_lines(axes[0], [alerta, acao, especificacao], initial_label=True)
    axes[0].set_title(f'{parametro} - {ponto}')
    axes[0].set_ylabel(f'{parametro} ({um})')
    add_stats_text(axes[0], df_ponto, ucl_condutividade, lcl_condutividade, media_condutividade, std_condutividade_minitab, decimal_places=4)
    axes[0].legend()
    axes[0].grid(False)
    # definir yticks apenas se valores válidos foram informados
    if escala_min_val is not None and escala_max_val is not None and intervalo_escala_val not in (None, 0):
        try:
            axes[0].set_yticks(np.arange(escala_min_val, escala_max_val + intervalo_escala_val/2.0, intervalo_escala_val))
        except Exception:
            pass

    violacoes_condutividade = df_ponto[df_ponto['cond_violacoes_nelson'] != '']
    add_nelson_plots(axes[0], violacoes_condutividade, 'Resultado', plot_label='Violação Nelson (I-Chart)')

    # Gráfico 2: MR
    sns.lineplot(data=df_ponto, x='Data', y='MR', marker='o', color='orange', ax=axes[1])
    axes[1].axhline(mr_media, color='green', linestyle='--', label='MR Média')
    axes[1].axhline(ucl_mr, color='red', linestyle='--', label='UCL MR')
    axes[1].axhline(lcl_mr, color='red', linestyle='--', label='LCL MR')
    add_reference_lines(axes[1], [alerta, acao, especificacao], initial_label=False)
    axes[1].set_title(f'Amplitude Móvel (MR) - {ponto}')
    axes[1].set_ylabel('MR (|diferença|)')
    add_stats_text(axes[1], df_ponto, ucl_mr, lcl_mr, mr_media, prefix='MR ', decimal_places=3)
    axes[1].legend()
    axes[1].grid(False)
    # aplicar yticks validados também para o gráfico de MR
    if escala_min_val is not None and escala_max_val is not None and intervalo_escala_val not in (None, 0):
        try:
            axes[1].set_yticks(np.arange(escala_min_val, escala_max_val + intervalo_escala_val/2.0, intervalo_escala_val))
        except Exception:
            pass

    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    axes[1].tick_params(axis='x', rotation=45)
    violacoes_mr = df_ponto[df_ponto['violacoes_nelson'] != '']
    add_nelson_plots(axes[1], violacoes_mr, 'MR', plot_label='Violação Nelson (MR-Chart)')

    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("Faça upload do arquivo CSV para visualizar os gráficos.")
    st.stop()















