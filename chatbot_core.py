import json
import os
import datetime
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from difflib import SequenceMatcher as ComparadorSequencia

class ChatbotClinica:
    def __init__(self):
        self.ARQUIVO_BASE_CONHECIMENTO = 'base_conhecimento.json'
        self.ARQUIVO_MODELO_ML = 'modelo_chatbot.joblib'
        self.DIRETORIO_HISTORICO_USUARIOS = 'historico_usuarios'
        
        # Carrega tudo ao inicializar
        self.frases, self.categorias = self.carregar_base_conhecimento()
        self.frases, self.categorias = self.verificar_consistencia_dados(self.frases, self.categorias)
        self.modelo = self.carregar_ou_treinar_modelo_ml(self.frases, self.categorias)
        self.respostas = self._carregar_respostas_pre_definidas()
        
        print("🚀 Chatbot inicializado com sucesso!")

    def _carregar_respostas_pre_definidas(self):
        return {
            "SAUDAÇÃO": {"initial": "Olá! Seja bem-vindo à nossa clínica. Como posso ajudar você hoje?"},
            "AJUDA": {
                "initial": "Claro! Posso ajudar com informações, agendamentos, ou dúvidas gerais. O que você precisa?",
                "continuation": "Em que mais posso ajudar agora?"
            },
            "INFORMAÇÃO": {
                "initial": "📋 Horário: Seg-Sex 7h-19h | Tel: (11) 3333-4444 | End: Rua Saúde, 123",
                "continuation": "Precisa de mais alguma informação?"
            },
            "CANCELAMENTO": {
                "initial": "Para cancelamento, vou transferir para um atendente...",
                "continuation": "Transferindo para atendente..."
            },
            "EXAMES": {
                "initial": "Serviços: Agendamento e consulta de resultados de exames.",
                "continuation": "Sobre exames, em que mais posso ajudar?"
            },
            "VALORES": {
                "initial": "Para orçamentos, consulte diretamente na clínica.",
                "continuation": "Consulte na clínica para valores precisos."
            },
            "DESCONHECIDO": {
                "initial": "Desculpe, não entendi. Pode reformular?"
            }
        }

    def carregar_base_conhecimento(self):
        if os.path.exists(self.ARQUIVO_BASE_CONHECIMENTO):
            with open(self.ARQUIVO_BASE_CONHECIMENTO, 'r', encoding='utf-8') as arquivo:
                dados = json.load(arquivo)
                return dados['frases'], dados['categorias']
        else:
            # SUA BASE ATUAL AQUI (copie e cole do seu main.py)
            frases_iniciais = ["bom dia", "boa tarde", "oi", "olá", ...]  # Cole suas frases
            categorias_iniciais = ["SAUDAÇÃO", "SAUDAÇÃO", ...]  # Cole suas categorias
            return frases_iniciais, categorias_iniciais

    def verificar_consistencia_dados(self, frases, categorias):
        if len(frases) != len(categorias):
            min_length = min(len(frases), len(categorias))
            frases = frases[:min_length]
            categorias = categorias[:min_length]
        return frases, categorias

    def criar_e_treinar_pipeline_ml(self, frases, categorias):
        pipeline_ml = Pipeline([
            ('vetorizacao', TfidfVectorizer(lowercase=True)),
            ('classificador', MultinomialNB())
        ])
        pipeline_ml.fit(frases, categorias)
        return pipeline_ml

    def carregar_ou_treinar_modelo_ml(self, frases, categorias):
        if os.path.exists(self.ARQUIVO_MODELO_ML):
            try:
                return joblib.load(self.ARQUIVO_MODELO_ML)
            except Exception as erro:
                print(f"Erro ao carregar modelo, retreinando: {erro}")
                modelo = self.criar_e_treinar_pipeline_ml(frases, categorias)
                joblib.dump(modelo, self.ARQUIVO_MODELO_ML)
                return modelo
        else:
            modelo = self.criar_e_treinar_pipeline_ml(frases, categorias)
            joblib.dump(modelo, self.ARQUIVO_MODELO_ML)
            return modelo

    def obter_texto_resposta(self, categoria, historico_usuario):
        if categoria not in self.respostas:
            return self.respostas["DESCONHECIDO"]["initial"]

        if not historico_usuario:
            return self.respostas[categoria]["initial"]

        ultima_categoria = historico_usuario[-1]['categoria'] if historico_usuario and 'categoria' in historico_usuario[-1] else None

        if "continuation" in self.respostas[categoria] and ultima_categoria in ["SAUDAÇÃO", "AJUDA"]:
            return self.respostas[categoria]["continuation"]
        
        return self.respostas[categoria]["initial"]

    def obter_resposta_fallback(self, mensagem, historico_usuario, limiar_similaridade=0.5):
        mensagem_lower = mensagem.lower()
        
        palavras_chave_prioritarias = {
            "cancelar": "CANCELAMENTO", "exame": "EXAMES", "horário": "INFORMAÇÃO",
            "ajuda": "AJUDA", "oi": "SAUDAÇÃO", "orçamento": "VALORES", "preço": "VALORES"
        }
        
        for palavra, categoria in palavras_chave_prioritarias.items():
            if palavra in mensagem_lower:
                return self.obter_texto_resposta(categoria, historico_usuario), categoria, True

        maior_similaridade = 0
        categoria_mais_similar = None

        for frase_base, categoria_base in zip(self.frases, self.categorias):
            similaridade = ComparadorSequencia(None, mensagem_lower, frase_base.lower()).ratio()
            if similaridade > maior_similaridade:
                maior_similaridade = similaridade
                categoria_mais_similar = categoria_base

        if maior_similaridade >= limiar_similaridade and categoria_mais_similar in self.respostas:
            return self.obter_texto_resposta(categoria_mais_similar, historico_usuario), categoria_mais_similar, True
        
        return None, None, False

    def carregar_historico_usuario(self, id_usuario):
        caminho_arquivo = os.path.join(self.DIRETORIO_HISTORICO_USUARIOS, f"{id_usuario}_historico.json")
        if os.path.exists(caminho_arquivo):
            with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo:
                return json.load(arquivo)
        return []

    def salvar_historico_usuario(self, id_usuario, mensagem, resposta_bot, categoria):
        if not os.path.exists(self.DIRETORIO_HISTORICO_USUARIOS):
            os.makedirs(self.DIRETORIO_HISTORICO_USUARIOS)

        historico = self.carregar_historico_usuario(id_usuario)
        historico.append({
            "mensagem_usuario": mensagem,
            "resposta_chatbot": resposta_bot,
            "categoria": categoria,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        caminho_arquivo = os.path.join(self.DIRETORIO_HISTORICO_USUARIOS, f"{id_usuario}_historico.json")
        with open(caminho_arquivo, 'w', encoding='utf-8') as arquivo:
            json.dump(historico, arquivo, indent=4, ensure_ascii=False)

    def processar_mensagem(self, mensagem_usuario, id_usuario="anonimo"):
        """
        MÉTODO PRINCIPAL PARA A API
        """
        historico_atual = self.carregar_historico_usuario(id_usuario)
        
        mensagem_limpa = mensagem_usuario.strip().lower()

        # Classificação com ML
        probabilidades = self.modelo.predict_proba([mensagem_limpa])[0]
        indice_maior_prob = probabilidades.argmax()
        maior_probabilidade = probabilidades[indice_maior_prob]
        categoria_prevista_ml = self.modelo.classes_[indice_maior_prob]

        if maior_probabilidade >= 0.4 and categoria_prevista_ml in self.respostas:
            texto_resposta = self.obter_texto_resposta(categoria_prevista_ml, historico_atual)
            categoria_detectada = categoria_prevista_ml
            usou_base_conhecimento = True
        else:
            # Fallback
            texto_fallback, categoria_fallback, fallback_usado = self.obter_resposta_fallback(mensagem_limpa, historico_atual)
            if fallback_usado:
                texto_resposta = texto_fallback
                categoria_detectada = categoria_fallback
                usou_base_conhecimento = True
            else:
                texto_resposta = self.respostas["DESCONHECIDO"]["initial"]
                categoria_detectada = "DESCONHECIDO"
                usou_base_conhecimento = False

        # Salva no histórico
        self.salvar_historico_usuario(id_usuario, mensagem_usuario, texto_resposta, categoria_detectada)

        return {
            "success": True,
            "data": {
                "resposta": texto_resposta,
                "categoria": categoria_detectada,
                "usou_base_conhecimento": usou_base_conhecimento,
                "timestamp": datetime.datetime.now().isoformat()
            }
        }