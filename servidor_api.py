from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot_core import ChatbotClinica
import datetime

app = Flask(__name__)
CORS(app)  # Permite frontend acessar

# Instância global do chatbot
chatbot = ChatbotClinica()

@app.route('/api/chat/mensagem', methods=['POST'])
def processar_mensagem():
    """
    Endpoint principal - espera JSON: {"mensagem": "texto", "usuario_id": "opcional"}
    """
    try:
        data = request.get_json()
        
        if not data or 'mensagem' not in data:
            return jsonify({
                "success": False,
                "error": "Campo 'mensagem' é obrigatório"
            }), 400
        
        mensagem = data['mensagem']
        usuario_id = data.get('usuario_id', 'anonimo')
        
        resultado = chatbot.processar_mensagem(mensagem, usuario_id)
        return jsonify(resultado)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Erro interno: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Verifica se o serviço está online"""
    return jsonify({
        "status": "online", 
        "service": "chatbot-clinica",
        "timestamp": datetime.datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🌐 Iniciando servidor do chatbot...")
    app.run(host='0.0.0.0', port=5000, debug=True)