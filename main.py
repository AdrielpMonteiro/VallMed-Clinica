import json
import os
import datetime
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from difflib import SequenceMatcher as ComparadorSequencia

# --- Configurações Principais do Chatbot ---
ARQUIVO_BASE_CONHECIMENTO = 'base_conhecimento.json'
ARQUIVO_MODELO_ML = 'modelo_chatbot.joblib'
DIRETORIO_HISTORICO_USUARIOS = 'historico_usuarios'

# --- Funções para Gerenciar a Base de Conhecimento ---
def carregar_base_conhecimento():
    """
    Carrega as frases de treinamento e suas categorias de um arquivo JSON.
    Se o arquivo não existir, inicia com uma base padrão para a clínica.
    """
    if os.path.exists(ARQUIVO_BASE_CONHECIMENTO):
        with open(ARQUIVO_BASE_CONHECIMENTO, 'r', encoding='utf-8') as arquivo:
            dados = json.load(arquivo)
            return dados['frases'], dados['categorias']
    else:
        # Base de Conhecimento Inicial da Clínica - CORRIGIDA (mesmo número de frases e categorias)
        frases_iniciais = [
            # Saudações (11 frases)
            "bom dia", "boa tarde", "boa noite", "oi", "olá", "tudo bem?", "e aí", "saudações", "olá, tudo bem?",
            "como você está", "hey",
            
            # Ajuda (10 frases)
            "preciso de ajuda", "preciso de auxílio", "gostaria de ajuda", "me ajude", "socorro", "pode me ajudar?", 
            "tenho uma dúvida", "não sei como fazer", "preciso de orientação", "como funciona",
            
            # Informações sobre a clínica (13 frases)
            "qual horário de funcionamento", "que horas vocês abrem", "telefone da clínica", "endereço da clínica", 
            "contato", "localização", "qual o horário", "quais os telefones", "onde fica a clínica", 
            "horário de atendimento", "como chegar", "qual o endereço", "número de telefone",
            
            # Cancelamento (8 frases)
            "quero cancelar meu plano", "cancelamento", "não quero mais continuar com plano", "cancelar", 
            "desativar conta", "gostaria de cancelar", "encerrar plano", "cancelar assinatura",
            
            # Exames (13 frases)
            "exame", "quero saber valores dos meus exames", "resultado exames", "exames", "agendar exame", 
            "laboratório", "verificar exame", "como pego resultado", "consulta de exames", "quais exames vocês fazem",
            "marcar exame", "resultados de exames", "laudo médico",
            
            # Orçamentos e Valores (16 frases)
            "orçamento", "valores", "qual valor", "me faça orçamento", "quais planos", "me diga valores", 
            "valor", "preços", "qual preço", "quanto custa", "preço da consulta", "valor do exame",
            "orçamentos", "tabela de preços", "custos", "quanto é"
        ]
        
        categorias_iniciais = [
            # Saudações 
            "SAUDAÇÃO", "SAUDAÇÃO", "SAUDAÇÃO", "SAUDAÇÃO", "SAUDAÇÃO", "SAUDAÇÃO", "SAUDAÇÃO", "SAUDAÇÃO", "SAUDAÇÃO",
            "SAUDAÇÃO", "SAUDAÇÃO",
            
            # Ajuda 
            "AJUDA", "AJUDA", "AJUDA", "AJUDA", "AJUDA", "AJUDA", "AJUDA", "AJUDA", "AJUDA", "AJUDA",
            
            # Informações 
            "INFORMAÇÃO", "INFORMAÇÃO", "INFORMAÇÃO", "INFORMAÇÃO", "INFORMAÇÃO", "INFORMAÇÃO", "INFORMAÇÃO", 
            "INFORMAÇÃO", "INFORMAÇÃO", "INFORMAÇÃO", "INFORMAÇÃO", "INFORMAÇÃO", "INFORMAÇÃO",
            
            # Cancelamento 
            "CANCELAMENTO", "CANCELAMENTO", "CANCELAMENTO", "CANCELAMENTO", "CANCELAMENTO", "CANCELAMENTO", 
            "CANCELAMENTO", "CANCELAMENTO",
            
            # Exames 
            "EXAMES", "EXAMES", "EXAMES", "EXAMES", "EXAMES", "EXAMES", "EXAMES", "EXAMES", "EXAMES", 
            "EXAMES", "EXAMES", "EXAMES", "EXAMES",
            
            # Valores/Orçamentos 
            "VALORES", "VALORES", "VALORES", "VALORES", "VALORES", "VALORES", "VALORES", "VALORES", "VALORES",
            "VALORES", "VALORES", "VALORES", "VALORES", "VALORES", "VALORES", "VALORES"
        ]
        
        # VERIFICAÇÃO DE CONSISTÊNCIA
        if len(frases_iniciais) != len(categorias_iniciais):
            print(f"⚠️  AVISO: Inconsistência detectada! Frases: {len(frases_iniciais)}, Categorias: {len(categorias_iniciais)}")
            # Ajusta automaticamente para ter o mesmo número
            min_length = min(len(frases_iniciais), len(categorias_iniciais))
            frases_iniciais = frases_iniciais[:min_length]
            categorias_iniciais = categorias_iniciais[:min_length]
            print(f"✅ Ajustado para: Frases: {len(frases_iniciais)}, Categorias: {len(categorias_iniciais)}")
        
        return frases_iniciais, categorias_iniciais

def salvar_base_conhecimento(frases: list, categorias: list):
    """Salva as frases e categorias atualizadas no arquivo JSON da base de conhecimento."""
    # Verifica consistência antes de salvar
    if len(frases) != len(categorias):
        print(f"❌ ERRO: Não é possível salvar - Frases ({len(frases)}) e Categorias ({len(categorias)}) têm quantidades diferentes!")
        return
    
    dados = {'frases': frases, 'categorias': categorias}
    with open(ARQUIVO_BASE_CONHECIMENTO, 'w', encoding='utf-8') as arquivo:
        json.dump(dados, arquivo, indent=4, ensure_ascii=False)

# Função para verificar e corrigir consistência dos dados
def verificar_consistencia_dados(frases: list, categorias: list) -> tuple:
    """Verifica e corrige inconsistências entre frases e categorias."""
    if len(frases) != len(categorias):
        print(f"⚠️  Corrigindo inconsistência: Frases ({len(frases)}) vs Categorias ({len(categorias)})")
        # Mantém apenas os pares que têm ambos
        min_length = min(len(frases), len(categorias))
        frases = frases[:min_length]
        categorias = categorias[:min_length]
        print(f"✅ Dados ajustados: {len(frases)} pares consistentes")
    return frases, categorias

# Carrega a base de conhecimento no início do programa
FRASES_CONHECIDAS, CATEGORIAS_CONHECIDAS = carregar_base_conhecimento()

# Verifica consistência imediatamente após carregar
FRASES_CONHECIDAS, CATEGORIAS_CONHECIDAS = verificar_consistencia_dados(FRASES_CONHECIDAS, CATEGORIAS_CONHECIDAS)

# Respostas pré-definidas para cada categoria identificada pelo chatbot
RESPOSTAS_PRE_DEFINIDAS = {
    "SAUDAÇÃO": {
        "initial": "Olá! Seja bem-vindo à nossa clínica. Como posso ajudar você hoje?"
    },
    "AJUDA": {
        "initial": "Claro! Posso ajudar com  agendamentos, ou dúvidas gerais. O que você precisa?",
        "continuation": "Em que mais posso ajudar agora?"
    },
    "INFORMAÇÃO": {
        "initial": (
            "📋 **Informações da Clínica:**\n"
            "• **Horário de Atendimento:** Segunda a Sexta, das 7h às 19h\n"
            "• **Telefone:** (11) 3333-4444\n"
            "• **Endereço:** Rua Saúde Perfeita, 123 - Centro\n"
            "• **WhatsApp:** (11) 98888-7777"
        ),
        "continuation": "Precisa de mais alguma informação sobre a clínica?"
    },
    "CANCELAMENTO": {
        "initial": (
            "Poxa, que pena! Para prosseguir com o cancelamento do seu plano, "
            "preciso transferir você para um de nossos atendentes. "
            "Aguarde um instante, por favor."
        ),
        "continuation": (
            "Entendi sobre o cancelamento. Vou te transferir para um atendente agora. "
            "Tenha um ótimo dia!"
        )
    },
    "EXAMES": {
        "initial": "Nossos serviços de exames incluem:\n• Agendamento de Exames\n• Consulta de Resultados Online\nQual serviço você gostaria de usar?",
        "continuation": "Sobre exames, em que mais posso ajudar?"
    },
    "VALORES": {
        "initial": "Para orçamentos e valores, é necessário consultar diretamente na clínica para obter informações precisas sobre preços e planos disponíveis.",
        "continuation": "Sobre valores e orçamentos, preciso que você consulte diretamente na clínica para informações precisas."
    },
    "DESCONHECIDO": {
        "initial": "Desculpe, não consegui entender sua pergunta no momento. Você poderia reformular de outra forma?"
    }
}

# --- Treinamento e Gerenciamento do Modelo de Machine Learning ---
def criar_e_treinar_pipeline_ml(frases: list, categorias: list) -> Pipeline:
    """
    Cria e treina um pipeline de Machine Learning (vetorização + classificação)
    usando o Scikit-learn para identificar a categoria das mensagens.
    """
    # Verificação final antes do treinamento
    if len(frases) != len(categorias):
        raise ValueError(f"Não é possível treinar o modelo: Frases ({len(frases)}) e Categorias ({len(categorias)}) têm quantidades diferentes!")
    
    if len(frases) == 0:
        raise ValueError("Não é possível treinar o modelo: Não há dados de treinamento!")
    
    print(f"🔧 Treinando modelo com {len(frases)} exemplos...")
    
    pipeline_ml = Pipeline([
        ('vetorizacao', TfidfVectorizer(lowercase=True)),
        ('classificador', MultinomialNB())
    ])
    pipeline_ml.fit(frases, categorias)
    print("✅ Modelo treinado com sucesso!")
    return pipeline_ml

def carregar_ou_treinar_modelo_ml(frases: list, categorias: list) -> Pipeline:
    """
    Tenta carregar um modelo de ML já treinado de um arquivo. Se o arquivo não existe
    ou se houver erro ao carregar, treina um novo modelo e o salva.
    """
    # Verifica consistência antes de qualquer operação
    frases, categorias = verificar_consistencia_dados(frases, categorias)
    
    if os.path.exists(ARQUIVO_MODELO_ML):
        try:
            modelo = joblib.load(ARQUIVO_MODELO_ML)
            print("✅ Modelo de chatbot de ML carregado com sucesso do arquivo.")
            return modelo
        except Exception as erro:
            print(f"⚠️ Erro ao carregar o modelo '{ARQUIVO_MODELO_ML}': {erro}. Retreinando...")
            modelo = criar_e_treinar_pipeline_ml(frases, categorias)
            joblib.dump(modelo, ARQUIVO_MODELO_ML)
            print("✅ Modelo de chatbot de ML retreinado e salvo novamente.")
            return modelo
    else:
        modelo = criar_e_treinar_pipeline_ml(frases, categorias)
        joblib.dump(modelo, ARQUIVO_MODELO_ML)
        print("✅ Modelo de chatbot de ML treinado e salvo pela primeira vez.")
        return modelo

# Carrega ou treina o modelo de ML uma única vez quando o sistema inicia
try:
    MODELO_CHATBOT_ML = carregar_ou_treinar_modelo_ml(FRASES_CONHECIDAS, CATEGORIAS_CONHECIDAS)
    print(f"🚀 Chatbot inicializado com sucesso! Modelo pronto para uso.")
except Exception as e:
    print(f"❌ Erro crítico ao inicializar o modelo: {e}")
    print("💡 Verifique se há dados suficientes na base de conhecimento.")
    exit(1)




def obter_texto_resposta(categoria: str, historico_usuario: list) -> str:
    """Define qual versão da resposta deve ser usada para uma dada categoria."""
    if categoria not in RESPOSTAS_PRE_DEFINIDAS:
        return RESPOSTAS_PRE_DEFINIDAS["DESCONHECIDO"]["initial"]

    if not historico_usuario:
        return RESPOSTAS_PRE_DEFINIDAS[categoria]["initial"]

    ultima_categoria_interacao = None
    if historico_usuario and 'categoria' in historico_usuario[-1]:
        ultima_categoria_interacao = historico_usuario[-1]['categoria']

    if "continuation" in RESPOSTAS_PRE_DEFINIDAS[categoria] and \
       ultima_categoria_interacao in ["SAUDAÇÃO", "AJUDA"]:
        return RESPOSTAS_PRE_DEFINIDAS[categoria]["continuation"]
    
    return RESPOSTAS_PRE_DEFINIDAS[categoria]["initial"]

def obter_resposta_fallback(
    mensagem: str, frases_conhecidas: list, categorias_conhecidas: list,
    historico_usuario: list, limiar_similaridade: float = 0.5
) -> tuple:
    """Mecanismo de fallback para quando o modelo de ML não está confiante."""
    mensagem_lower = mensagem.lower()
    
    palavras_chave_prioritarias = {
        "cancelar": "CANCELAMENTO", "cancelamento": "CANCELAMENTO", "plano": "CANCELAMENTO",
        "exame": "EXAMES", "resultado": "EXAMES", "agendar": "EXAMES", "laboratório": "EXAMES",
        "horário": "INFORMAÇÃO", "telefone": "INFORMAÇÃO", "endereço": "INFORMAÇÃO", "contato": "INFORMAÇÃO",
        "ajuda": "AJUDA", "auxílio": "AJUDA", "socorro": "AJUDA", "dúvida": "AJUDA",
        "oi": "SAUDAÇÃO", "olá": "SAUDAÇÃO", "bom dia": "SAUDAÇÃO", "boa tarde": "SAUDAÇÃO", "boa noite": "SAUDAÇÃO",
        "orçamento": "VALORES", "preço": "VALORES", "valor": "VALORES", "quanto custa": "VALORES", "custa": "VALORES"
    }
    
    for palavra, categoria_prioritaria in palavras_chave_prioritarias.items():
        if palavra in mensagem_lower:
            texto_resposta = obter_texto_resposta(categoria_prioritaria, historico_usuario)
            return texto_resposta, categoria_prioritaria, True

    maior_similaridade = 0
    categoria_mais_similar = None

    for frase_base, categoria_base in zip(frases_conhecidas, categorias_conhecidas):
        similaridade = ComparadorSequencia(None, mensagem_lower, frase_base.lower()).ratio()
        if similaridade > maior_similaridade:
            maior_similaridade = similaridade
            categoria_mais_similar = categoria_base

    if maior_similaridade >= limiar_similaridade and categoria_mais_similar in RESPOSTAS_PRE_DEFINIDAS:
        texto_resposta = obter_texto_resposta(categoria_mais_similar, historico_usuario)
        return texto_resposta, categoria_mais_similar, True
    else:
        return None, None, False

def classificar_e_responder(
    mensagem: str,
    historico_usuario: list,
    modelo_ml: Pipeline,
    frases_conhecidas: list,
    categorias_conhecidas: list,
    limiar_confianca_ml: float = 0.4
) -> tuple:
    """Processa a mensagem do usuário e gera a resposta adequada."""
    mensagem_limpa = mensagem.strip().lower()

    probabilidades = modelo_ml.predict_proba([mensagem_limpa])[0]
    indice_maior_prob = probabilidades.argmax()
    maior_probabilidade = probabilidades[indice_maior_prob]
    categoria_prevista_ml = modelo_ml.classes_[indice_maior_prob]

    if maior_probabilidade >= limiar_confianca_ml and categoria_prevista_ml in RESPOSTAS_PRE_DEFINIDAS:
        texto_resposta = obter_texto_resposta(categoria_prevista_ml, historico_usuario)
        return texto_resposta, categoria_prevista_ml, True

    texto_fallback, categoria_fallback, fallback_usado = obter_resposta_fallback(
        mensagem_limpa, frases_conhecidas, categorias_conhecidas, historico_usuario
    )
    if fallback_usado:
        return texto_fallback, categoria_fallback, True

    return RESPOSTAS_PRE_DEFINIDAS["DESCONHECIDO"]["initial"], "DESCONHECIDO", False

def carregar_historico_usuario(id_usuario: str) -> list:
    """Carrega o histórico de conversas de um usuário específico."""
    caminho_arquivo_usuario = os.path.join(DIRETORIO_HISTORICO_USUARIOS, f"{id_usuario}_historico.json")
    if os.path.exists(caminho_arquivo_usuario):
        with open(caminho_arquivo_usuario, 'r', encoding='utf-8') as arquivo:
            return json.load(arquivo)
    return []

def salvar_historico_usuario(id_usuario: str, mensagem: str, resposta_bot: str, categoria: str):
    """Salva a interação no histórico do usuário."""
    if not os.path.exists(DIRETORIO_HISTORICO_USUARIOS):
        os.makedirs(DIRETORIO_HISTORICO_USUARIOS)

    historico = carregar_historico_usuario(id_usuario)
    historico.append({
        "mensagem_usuario": mensagem,
        "resposta_chatbot": resposta_bot,
        "categoria": categoria,
        "timestamp": datetime.datetime.now().isoformat()
    })
    
    caminho_arquivo_usuario = os.path.join(DIRETORIO_HISTORICO_USUARIOS, f"{id_usuario}_historico.json")
    with open(caminho_arquivo_usuario, 'w', encoding='utf-8') as arquivo:
        json.dump(historico, arquivo, indent=4, ensure_ascii=False)

def servico_chatbot_entrada(id_usuario: str, mensagem_usuario: str) -> dict:
    """Função principal para integração com backend."""
    historico_atual_usuario = carregar_historico_usuario(id_usuario)

    texto_resposta, categoria_detectada, usou_base_conhecimento = classificar_e_responder(
        mensagem_usuario, historico_atual_usuario, MODELO_CHATBOT_ML, FRASES_CONHECIDAS, CATEGORIAS_CONHECIDAS
    )

    salvar_historico_usuario(id_usuario, mensagem_usuario, texto_resposta, categoria_detectada)

    return {
        "resposta_chatbot": texto_resposta,
        "categoria": categoria_detectada,
        "usou_base_conhecimento": usou_base_conhecimento
    }

# --- Uso Local ---
if __name__ == "__main__":
    print("--- Chatbot da Clínica (Modo Console) ---")
    print("Digite 'sair' ou 'tchau' para encerrar a conversa a qualquer momento.")

    id_usuario_teste = input("Por favor, digite seu nome ou um ID para começar: ").strip()
    if not id_usuario_teste:
        id_usuario_teste = "usuario_padrao"
        print(f"Nenhum nome/ID digitado, usando ID padrão: {id_usuario_teste}")

    print(f"\nOlá, {id_usuario_teste}! Iniciando sua conversa com o chatbot da clínica...")
    print(f"Chatbot: {RESPOSTAS_PRE_DEFINIDAS['SAUDAÇÃO']['initial']}")

    while True:
        entrada_usuario = input(f"{id_usuario_teste}> ").strip()
        
        if entrada_usuario.lower() in ['sair', 'exit', 'tchau', 'encerrar','obrigado','adeus']:
            print("Chatbot: Chat encerrado. Tenha um ótimo dia! 👋")
            break

        resultado_servico = servico_chatbot_entrada(id_usuario_teste, entrada_usuario)

        print(f"Chatbot: {resultado_servico['resposta_chatbot']}")
        print(f"  (Categoria Detectada: {resultado_servico['categoria']}, Resposta da KB: {resultado_servico['usou_base_conhecimento']})")
        print("-" * 50)