import pandas as pd
import sys
#from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

nltk.download('punkt')

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Usar a chave API carregada do arquivo de ambiente
llm = ChatOpenAI(api_key=openai_api_key, model='gpt-3.5-turbo')

def load_comments(csv_file, column_name):
    df = pd.read_csv(csv_file)
    return df[column_name].tolist()

def create_prompt_template(case_of_use):
    templates = {
        "analise_sentimento": "Analisar o sentimento dos seguintes comentários (positivo, negativo, neutro):\n{comments}. O resultado deve conter o comentário e o resultado, além de uma tabela com os sentimentos e a contagem total de cada um. Crie também um texto de no máximo 10 linhas que possa dar uma visão ao conselho executivo do cenário geral considerando a análise dos comentários.",
        "questoes_comuns": "Identificar temas comuns nos comentários negativos, focando em bugs, problemas de usabilidade ou problemas específicos de funcionalidades:\n -- INÍCIO DOS COMENTÁRIOS -- {comments}\n -- FIM DOS COMENTÁRIOS -- Para os temas encontrados, por favor exemplifique com comentários.",
        "avaliacao_funcionalidades": "Listar as funcionalidades mencionadas nos comentários e classificá-las com base na frequência e no sentimento associado (positivo ou negativo):\n -- INÍCIO DOS COMENTÁRIOS -- {comments} \n-- FIM DOS COMENTÁRIOS -- No resultado, por favor mostre quantos comentários estão relacionados à funcionalidade e o sentimento associado. Explique também como a funcionalidade atenderá ao comentário associado.",
        "monitorar_atualizacoes": "Comparar o sentimento dos comentários antes e depois da atualização:\nAntes:\n{comments_before}\nDepois:\n{comments_after}",
        "extrair_sugestoes": "Extrair sugestões dos comentários dos usuários para melhorias e novas funcionalidades que gostariam de ver:\n{comments}",
        "benchmark_concorrentes": "Comparar o sentimento dos comentários para nosso aplicativo e o aplicativo de um concorrente:\nNosso App:\n{comments_app}\nApp do Concorrente:\n{comments_competitor}",
        "segmentar_usuarios": "Categorizar os comentários com base nos seguintes segmentos: {segments}\nComentários:\n{comments}",
        "detectar_spam": "Identificar e filtrar comentários que são spam, ofensivos ou não relacionados ao aplicativo:\n{comments}",
        "analisar_tendencias": "Detectar mudanças nas opiniões dos usuários ao longo do tempo analisando tendências nos comentários:\n{comments}",
        "personalizar_respostas": "Automatizar e personalizar respostas aos comentários dos usuários, fornecendo respostas apropriadas e úteis com base no conteúdo e no sentimento de cada comentário:\n{comments}"
    }
    return templates.get(case_of_use, "")

def analyze_sentiment(comments):
    prompt_template = create_prompt_template('analise_sentimento')
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = prompt | llm
    response = llm_chain.invoke("\n".join(comments))
    return response.content

def identify_common_issues(comments):
    prompt_template = create_prompt_template('questoes_comuns')
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = prompt | llm
    response = llm_chain.invoke("\n".join(comments))
    return response.content

def evaluate_features(comments):
    prompt_template = create_prompt_template('avaliacao_funcionalidades')
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = prompt | llm
    response = llm_chain.invoke(comments)
    return response.content

def monitor_updates(comments_before, comments_after):
    return 'NOT IMPLEMENTED'

def extract_suggestions(comments):
    return 'NOT IMPLEMENTED'

def benchmark_competitors(comments_app, comments_competitor):
    return 'NOT IMPLEMENTED'

def segment_users(comments, segments):
    return 'NOT IMPLEMENTED'

def detect_spam(comments):
    return 'NOT IMPLEMENTED'

def analyze_trends(comments):
    return 'NOT IMPLEMENTED'

def personalize_responses(comments):
    responses = ["Thank you for your feedback!" for _ in comments]
    return responses

def print_header(header):
    print("\n" + "=" * 80)
    print(header)
    print("=" * 80)

def run_all_cases(comments):
    print_header("Análise de Sentimento")
    sentiments = analyze_sentiment(comments)
    print(sentiments)

    print_header("Questões Comuns")
    issues = identify_common_issues(comments)
    print(issues)

    print_header("Avaliação de Funcionalidades")
    features = evaluate_features(comments)
    print(features)

    print_header("Extrair Sugestões")
    suggestions = extract_suggestions(comments)
    print(suggestions)

    print_header("Segmentar Usuários")
    segments = ['segment1', 'segment2']  # Defina seus segmentos
    segmented_comments = segment_users(comments, segments)
    print(segmented_comments)

    print_header("Detectar Spam")
    spam_comments = detect_spam(comments)
    print(spam_comments)

    print_header("Analisar Tendências")
    trends = analyze_trends(comments)
    print(trends)

    print_header("Personalizar Respostas")
    responses = personalize_responses(comments)
    print(responses)

def main():
    if len(sys.argv) < 4:
        print("Uso: python app.py <caso_de_uso|todos> <google_play_csv_file> <app_store_csv_file>")
        sys.exit(1)

    case_of_use = sys.argv[1]
    google_play_csv_file = sys.argv[2]
    app_store_csv_file = sys.argv[3]

    google_play_comments = load_comments(google_play_csv_file, 'content')
    app_store_comments = load_comments(app_store_csv_file, 'review')

    if case_of_use == "todos":
        print_header("Google Play")
        run_all_cases(google_play_comments)
        print_header("App Store")
        run_all_cases(app_store_comments)
    elif case_of_use == "analise_sentimento":
        print_header("Análise de Sentimento - Google Play")
        google_play_sentiments = analyze_sentiment(google_play_comments)
        print(google_play_sentiments)

        print_header("Análise de Sentimento - App Store")
        app_store_sentiments = analyze_sentiment(app_store_comments)
        print(app_store_sentiments)
    elif case_of_use == "questoes_comuns":
        print_header("Questões Comuns - Google Play")
        google_play_issues = identify_common_issues(google_play_comments)
        print(google_play_issues)

        print_header("Questões Comuns - App Store")
        app_store_issues = identify_common_issues(app_store_comments)
        print(app_store_issues)
    elif case_of_use == "avaliacao_funcionalidades":
        print_header("Avaliação de Funcionalidades - Google Play")
        google_play_features = evaluate_features(google_play_comments)
        print(google_play_features)

        print_header("Avaliação de Funcionalidades - App Store")
        app_store_features = evaluate_features(app_store_comments)
        print(app_store_features)
    elif case_of_use == "monitorar_atualizacoes":
        print_header("Monitorar Atualizações - Google Play")
        google_play_comments_before = load_comments('google_play_comments_before.csv', 'content')
        google_play_comments_after = load_comments('google_play_comments_after.csv', 'content')
        google_play_sentiments_before, google_play_sentiments_after = monitor_updates(google_play_comments_before, google_play_comments_after)
        print("Antes:", google_play_sentiments_before)
        print("Depois:", google_play_sentiments_after)

        print_header("Monitorar Atualizações - App Store")
        app_store_comments_before = load_comments('app_store_comments_before.csv', 'review')
        app_store_comments_after = load_comments('app_store_comments_after.csv', 'review')
        app_store_sentiments_before, app_store_sentiments_after = monitor_updates(app_store_comments_before, app_store_comments_after)
        print("Antes:", app_store_sentiments_before)
        print("Depois:", app_store_sentiments_after)
    elif case_of_use == "extrair_sugestoes":
        print_header("Extrair Sugestões - Google Play")
        google_play_suggestions = extract_suggestions(google_play_comments)
        print(google_play_suggestions)

        print_header("Extrair Sugestões - App Store")
        app_store_suggestions = extract_suggestions(app_store_comments)
        print(app_store_suggestions)
    elif case_of_use == "benchmark_concorrentes":
        print_header("Benchmark de Concorrentes - Google Play")
        google_play_comments_competitor = load_comments('google_play_comments_competitor.csv', 'content')
        google_play_sentiments_app, google_play_sentiments_competitor = benchmark_competitors(google_play_comments, google_play_comments_competitor)
        print("Sentimentos do App:", google_play_sentiments_app)
        print("Sentimentos do Concorrente:", google_play_sentiments_competitor)

        print_header("Benchmark de Concorrentes - App Store")
        app_store_comments_competitor = load_comments('app_store_comments_competitor.csv', 'review')
        app_store_sentiments_app, app_store_sentiments_competitor = benchmark_competitors(app_store_comments, app_store_comments_competitor)
        print("Sentimentos do App:", app_store_sentiments_app)
        print("Sentimentos do Concorrente:", app_store_sentiments_competitor)
    elif case_of_use == "segmentar_usuarios":
        print_header("Segmentar Usuários - Google Play")
        google_play_segments = ['segment1', 'segment2']  # Defina seus segmentos
        google_play_segmented_comments = segment_users(google_play_comments, google_play_segments)
        print(google_play_segmented_comments)

        print_header("Segmentar Usuários - App Store")
        app_store_segments = ['segment1', 'segment2']  # Defina seus segmentos
        app_store_segmented_comments = segment_users(app_store_comments, app_store_segments)
        print(app_store_segmented_comments)
    elif case_of_use == "detectar_spam":
        print_header("Detectar Spam - Google Play")
        google_play_spam_comments = detect_spam(google_play_comments)
        print(google_play_spam_comments)

        print_header("Detectar Spam - App Store")
        app_store_spam_comments = detect_spam(app_store_comments)
        print(app_store_spam_comments)
    elif case_of_use == "analisar_tendencias":
        print_header("Analisar Tendências - Google Play")
        google_play_trends = analyze_trends(google_play_comments)
        print(google_play_trends)

        print_header("Analisar Tendências - App Store")
        app_store_trends = analyze_trends(app_store_comments)
        print(app_store_trends)
    elif case_of_use == "personalizar_respostas":
        print_header("Personalizar Respostas - Google Play")
        google_play_responses = personalize_responses(google_play_comments)
        print(google_play_responses)

        print_header("Personalizar Respostas - App Store")
        app_store_responses = personalize_responses(app_store_comments)
        print(app_store_responses)
    else:
        print("Caso de uso inválido")

if __name__ == "__main__":
    main()