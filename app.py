import streamlit as st
import pandas as pd
import joblib
import os
from streamlit_option_menu import option_menu

# ===============================
# Carregar modelo
# ===============================
CAMINHO_ATUAL = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(CAMINHO_ATUAL, 'models\modelo_obesidade.pkl')
modelo = joblib.load(MODELO_PATH)

with st.sidebar:
    selected5 = option_menu('Projeto Saúde', ["Predição", "Dashboard"],
                            icons=['house', 'cloud-upload'],
                            key='menu_2')
    selected5
    if selected5 == 'Home':
        add_radio = st.radio(
            "Escolha a forma de visualização",
            ("Dataframe Atual", "Gráfico de Linhas", "Gráfico de Barras")
            )

with st.container():
    
    #==========Página de predição==========
    if selected5 == 'Predição':
        st.write('# Predição à Obesidade')
        st.write('---')
        st.write('## 1. Dados pessoais do paciente')
        
        col1, col2 = st.columns(2)
        with col1:
            genero = st.selectbox('Gênero', ['Feminino', 'Masculino'])
            idade = st.number_input('Idade', min_value=10, max_value=100, value=25, step=1)
        with col2:
            altura = st.number_input('Altura (em metros)', min_value=1.0, max_value=2.5, value=1.7, step=0.01)
            peso = st.number_input('Peso (em kg)', min_value=30.0, max_value=200.0, value=70.0, step=0.1)
        
        # Histórico do paciente
        st.write('---')
        st.write('## 2. Dados Gerais')
        col3, col4 = st.columns(2)
        with col3:
            historico_familiar_sobrepeso = st.radio('Histórico familiar de obesidade?', ['Sim', 'Não'])
            consumo_alimentos_caloricos	 = st.radio('Consome alimentos calóricos?', ['Sim', 'Não'])
            refeicoes_por_dia = st.selectbox('Número de refeições por dia', [1, 2, 3, '4 ou mais'])
            consumo_agua_por_dia = st.selectbox('Consumo de água por dia', ['Menos de 1 litro', 'Entre 1 e 2 litros', 'Mais de 2 litros'])
            consumo_alcoolico = st.selectbox('Consumo alcoólico?', ['Nunca', 'As vezes', 'Frequentemente', 'Sempre'])
            atividade_fisica_semana = st.selectbox('Pratica atividade física?', ['Nenhuma', '1 a 2 vezes por semana', '3 a 4 vezes por semana', '5 ou mais vezes por semana'])
        with col4:
            fumante = st.radio('Fumante?', ['Sim', 'Não'])
            monitora_calorias = st.radio('Monitora consumo de calorias?', ['Sim', 'Não'])
            lanches_entre_refeicoes = st.selectbox('Lanche entre refeições?', ['Nunca', 'As vezes', 'Frequentemente', 'Sempre'])
            consumo_vegetais = st.selectbox('Consumo de vegetais?', ['Raramente', 'As vezes', 'Sempre'])
            tempo_eletronicos_por_dia = st.selectbox('Tempo com eletrônicos/telas por dia', ['Menos de 2 horas', '3 a 5 horas', 'Maior que 5 horas'])
            meio_transporte= st.selectbox('Meio de transporte habitual', ['Caminhada', 'Transporte público', 'Bicicleta', 'Motocicleta', 'Automóvel'])
        
        # Gerador da análise
        st.write('---')
        st.write('## 2. Gerar Análise')
        
        # Mapear respostas para valores esperados pelo modelo
        map_sim_nao = {'Sim': 1, 'Não': 0}
        map_genero = {'Feminino': 0, 'Masculino': 1}
        map_refeicoes = {
            1: 'uma_refeicao_dia',
            2: 'duas_refeicao_dia',
            3: 'tres_refeicao_dia',
            '4 ou mais': 'quatro_ou_mais_refeicao_dia'
        }
        map_consumo_agua = {
            'Menos de 1 litro': 'menor_um_litro',
            'Entre 1 e 2 litros': 'entre_um_dois_litro',
            'Mais de 2 litros': 'maior_dois_litro'
        }
        map_consumo_vegetais = {
            'Raramente': 'raramente',
            'As vezes': 'as_vezes',
            'Sempre': 'sempre'
        }
        map_lanches_alcool = {
            'Nunca': 'no',
            'As vezes': 'Sometimes',
            'Frequentemente': 'Frequently',
            'Sempre': 'Always'
        }
        map_atividade_fisica = {
            'Nenhuma': 'nenhuma',
            '1 a 2 vezes por semana': 'uma_a_duas_por_semana',
            '3 a 4 vezes por semana': 'tres_a_quatro_por_semana',
            '5 ou mais vezes por semana': 'cinco_ou_mais_por_semana'
        }
        map_tempo_eletornicos = {
            'Menos de 2 horas': 'menor_igual_duas_hora_dia',
            '3 a 5 horas': 'tres_a_cinco_hora_dia',
            'Maior que 5 horas': 'maior_cinco_hora_dia'
        }
        map_transporte = {
            'Caminhada': 'Walking',
            'Bicicleta': 'Bike',
            'Transporte público': 'Public_Transportation',
            'Motocicleta': 'Motorbike',
            'Automóvel': 'Automobile'
        }
        
        # Gerar botão azul que fica vermelho quando clicado
        if st.button('Analisar'):

            dados_paciente = pd.DataFrame({
                'idade': [idade],
                'altura': [altura],
                'peso': [peso],
                'imc': [peso / (altura ** 2)],
                'genero': [map_genero[genero]],
                'historico_familiar_sobrepeso': [map_sim_nao[historico_familiar_sobrepeso]],
                'consumo_alimentos_caloricos': [map_sim_nao[consumo_alimentos_caloricos]],
                'fumante': [map_sim_nao[fumante]],
                'monitora_calorias': [map_sim_nao[monitora_calorias]],
                'consumo_vegetais': [map_consumo_vegetais[consumo_vegetais]],
                'refeicoes_por_dia': [map_refeicoes[refeicoes_por_dia]],
                'lanches_entre_refeicoes': [map_lanches_alcool[lanches_entre_refeicoes]],
                'consumo_agua_por_dia': [map_consumo_agua[consumo_agua_por_dia]],
                'atividade_fisica_semana': [map_atividade_fisica[atividade_fisica_semana]],
                'tempo_eletronicos_por_dia': [map_tempo_eletornicos[tempo_eletronicos_por_dia]],
                'consumo_alcoolico': [map_lanches_alcool[consumo_alcoolico]],
                'meio_transporte': [map_transporte[meio_transporte]]
            })
            
            FEATURES_ORDER = [
                'idade',
                'altura',
                'peso',
                'imc',
                'genero',
                'historico_familiar_sobrepeso',
                'consumo_alimentos_caloricos',
                'fumante',
                'monitora_calorias',
                'consumo_vegetais',
                'refeicoes_por_dia',
                'lanches_entre_refeicoes',
                'consumo_agua_por_dia',
                'atividade_fisica_semana',
                'tempo_eletronicos_por_dia',
                'consumo_alcoolico',
                'meio_transporte'
            ]
            
            previsao = modelo.predict(dados_paciente)
            probabilidade = modelo.predict_proba(dados_paciente)
            
            # Mostrar resultado
            st.write('### Resultado da Análise')
            if previsao[0] == 1:
                st.error('O paciente está propenso a Obesidade.')
            else:
                st.success(' paciente não está propenso a Obesidade.')
            
            st.write(f'Probabilidade de ser Obeso: {100*probabilidade[0][1]}%')
        