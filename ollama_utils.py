import pandas as pd
from collections import Counter
from langchain.prompts import PromptTemplate
import ollama
from dotenv import load_dotenv
import os
import json
import nltk
from datetime import datetime

nltk.download('punkt')

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

def load_comments(json_file):
    """Load comments from a JSON file with multiple lines of JSON objects."""
    comments = []
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                comment = json.loads(line.strip())
                if 'content' in comment and 'date' in comment:
                    comments.append({
                        'content': comment['content'],
                        'date': comment['date']
                    })
    except FileNotFoundError:
        print(f"Erro: O arquivo {json_file} não foi encontrado.")
    except json.JSONDecodeError as e:
        print(f"Erro ao ler o arquivo {json_file}: {e}")
    except Exception as e:
        print(f"Erro ao ler o arquivo {json_file}: {e}")
    return comments

def create_prompt_template(case_of_use):
    templates = {
        "analise_sentimento": (
            "Analisar o sentimento dos seguintes comentários (positivo, negativo, neutro):\n{comments}. "
            "O resultado deve ser **apenas** um JSON estruturado com os seguintes campos informados abaixo, sem explicações adicionais, texto fora do JSON ou comentários. "
            "'Tabela de Sentimentos' (um dicionário com a contagem de cada sentimento), "
            "'Comentários por Tópicos' (um dicionário com listas de comentários categorizados por sentimento e obrigatoriamente a data do comentário), "
            "e 'Análise para o Conselho Executivo' (um texto de no máximo 10 linhas). "
            "Não inclua nenhum texto adicional fora do JSON."
        )
    }
    return templates.get(case_of_use, "")

def save_results(results, prefix):
    """Save data to a JSON file with a unique name."""
    try:
        base_date = datetime.now().strftime("%d_%m_%Y")
        filename = f"{base_date}_{prefix}.json"
        counter = 1
        while os.path.exists(filename):
            filename = f"{base_date}_{counter}_{prefix}.json"
            counter += 1
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Resultados salvos em {filename}")
    except Exception as e:
        print(f"Erro ao salvar os resultados em {filename}: {e}")

def extract_json(content):
    """Extract JSON part from the content."""
    import re
    try:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json_match.group(0)
        else:
            raise ValueError("JSON não encontrado na resposta.")
    except Exception as e:
        print(f"Erro ao extrair JSON: {e}")
        return ""

def format_response(result):
    """Format the response to match the expected JSON structure."""
    sentiment_counts = Counter()
    comments_by_sentiment = {"Positivo": [], "Negativo": [], "Neutro": []}

    # Função para formatar a data no formato brasileiro (DD-MM-YYYY)
    def format_date(date_str):
        try:
            # Tentar parsear a data no formato original (YYYY-MM-DD ou qualquer outro)
            date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
            return date_obj.strftime("%d-%m-%Y")
        except ValueError:
            # Caso não consiga parsear, retornar a data original
            return date_str
    
    # Percorrer os comentários por tópicos e categorizar os sentimentos
    for date, comments in result["Comentários por Tópicos"].items():
        formatted_date = format_date(date)  # Formatar a data aqui
        for comment in comments:
            sentiment = comment["sentimento"]
            if sentiment in comments_by_sentiment:
                # Adicionar o comentário à lista de cada sentimento
                comments_by_sentiment[sentiment].append({
                    "Data": formatted_date,  # Data formatada
                    "Comentário": comment["comentario"]
                })
                sentiment_counts[sentiment] += 1

    # Montar o resultado formatado
    formatted_result = {
        "Tabela de Sentimentos": {
            "Positivo": sentiment_counts["Positivo"],
            "Negativo": sentiment_counts["Negativo"],
            "Neutro": sentiment_counts["Neutro"]
        },
        "Comentários por Tópicos": comments_by_sentiment,
        "Análise para o Conselho Executivo": result.get("Análise para o Conselho Executivo", "")
    }

    return formatted_result

def analyze_sentiment(comments):
    """Analyze sentiment of comments."""
    try:
        # Exemplo de saída esperada
        exemplo_saida = """
        Exemplo de saída esperada, pode conter mais dados, esse é a estrutura modelo de exemplo:
        {
            "Tabela de Sentimentos": {
                "Positivo": 5,
                "Negativo": 2,
                "Neutro": 4
            },
            "Comentários por Tópicos": {
                "16-11-2024": [
                    {"sentimento": "Positivo", "comentario": "A opção Vivo Easy pra mim é a melhor, não uso muito internet de dados, assim, considero justo pagar por uma quantidade e usá-la até o fim."},
                    {"sentimento": "Neutro", "comentario": "Estou usando há uns meses e não tive nenhum problema até agora, preço bem em conta"},
                    {"sentimento": "Positivo", "comentario": "Um bom plano, para as minhas necessidades"},
                    {"sentimento": "Positivo", "comentario": "Internet acumula isto, e ótimo"},
                    {"sentimento": "Neutro", "comentario": "Ok"}
                ],
                "16-11-2024": [
                    {"sentimento": "Negativo", "comentario": "Cadê as promoções de Black Friday vivo EASY para comprar GB de internet..."},
                    {"sentimento": "Neutro", "comentario": "Show de bola só me perco um pouco nos saldos...tinha que ser mais objetivo."}
                ],
                "16-11-2024": [
                    {"sentimento": "Negativo", "comentario": "As vezes fico sem telefone na rua"},
                    {"sentimento": "Neutro", "comentario": "Um bom plano, para as minhas necessidades"}
                ]
            },
            "Análise para o Conselho Executivo": "O sentimento geral é positivo com 5 comentários favoráveis. Existem 2 comentários negativos e 4 neutros. A principal queixa é sobre o sinal de internet ruim e a falta de promoções em Black Friday."
        }
        """
        
        # Template para o prompt do LLM
        prompt_template = create_prompt_template('analise_sentimento')
        formatted_comments = "\n".join([f"{comment['date']} - {comment['content']}" for comment in comments])
        
        # Adicionando o exemplo de saída esperada
        prompt = prompt_template.format(comments=formatted_comments) + exemplo_saida
        
        # Print o prompt para depuração
        print("Prompt being sent to the model:")
        print(prompt)
        
        response = ollama.chat(model='llama3:8b', messages=[{'role': 'user', 'content': prompt}])
        
        # Print a resposta para depuração
        print(response)
        
        # Extrair e limpar o conteúdo da resposta
        content = response['message']['content']
        
        # Verificar se o conteúdo é JSON ou precisa ser extraído
        json_content = extract_json(content)
        
        if not json_content:
            raise ValueError("Falha ao extrair o conteúdo JSON da resposta.")
        
        # Print o conteúdo limpo para depuração
        print("Cleaned content:")
        print(json_content)
        
        # Analisar a resposta JSON
        result = json.loads(json_content)
        
        # Formatar a resposta
        formatted_result = format_response(result)
        
        # Print o resultado para depuração
        print("Result to be saved:")
        print(json.dumps(formatted_result, ensure_ascii=False, indent=4))
        
        save_results(formatted_result, "android_analysis_sentiment")
    except json.JSONDecodeError as e:
        print(f"Erro ao analisar sentimentos: problema ao decodificar JSON - {e}")
    except ValueError as e:
        print(f"Erro ao analisar sentimentos: {e}")
    except Exception as e:
        print(f"Erro ao analisar sentimentos: {e}")

def print_header(header):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(header)
    print("=" * 80)

# Exemplo de uso
if __name__ == "__main__":
    print_header("Análise de Sentimento")
    json_file = "/mnt/data/17_11_2024_google_play_review.json"  # Substitua pelo nome do seu arquivo de comentários
    comments = load_comments(json_file)
    if comments:
        analyze_sentiment(comments)
