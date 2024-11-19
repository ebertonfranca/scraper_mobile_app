# Scraper Android App

Este projeto fornece scripts para raspar análises de aplicativos da Google Play Store, analisar os sentimentos dessas análises e identificar questões comuns.

## Estrutura do Projeto

1. `scraper_android.py`: Script que faz a raspagem de dados no Google Play Store
2. `app.py`: Script principal para rodar análises de sentimento.
3. `utils.py`: Contém funções utilitárias para carregar comentários, analisar sentimentos e imprimir cabeçalhos.
4. `test.py`: Contém testes para as funções principais.

## Dependências

As dependências para este projeto incluem bibliotecas de terceiros como pandas, nltk, langchain, dotenv, entre outras.

## Instalação

1. Clone o repositório:
    ```bash
    git clone https://github.com/seu-usuario/seu-repositorio.git
    cd seu-repositorio
    ```

2. Crie um ambiente virtual (opcional, mas recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Para Linux e macOS
    .\venv\Scripts\activate   # Para Windows
    ```

3. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

1. Executar o Script de raspagem:
    ```bash
    python scraper_android.py
    ```

2. Executar o Script de análise de sentimentos:
    ```bash
    python app.py analise_sentimento google_play_reviews.json
    ```

    Nesta função você pode alterar a quantidade de reviews raspados da Google Play no `count=5`:
    ```python
    def scrape_google_play_reviews(app_id, lang, country, count=5):
    ```

3. Executar dashboard para visualização gráfica da análise de sentimento:
    ```bash
    streamlit run dashboard.py
    ```

4. Executar testes:
    Explicação das Funções de Teste
    - `test_load_comments()`: Cria dados JSON simulados, escreve esses dados em um arquivo temporário `mock_google_play_reviews.json`, carrega os comentários usando a função `load_comments` e verifica se os comentários carregados correspondem aos dados simulados.
    - `test_analyze_sentiment()`: Utiliza uma lista de comentários simulados e chama a função `analyze_sentiment` para analisar os sentimentos dos comentários.
    - `test_print_header()`: Testa a função `print_header` imprimindo um cabeçalho de teste.

    Execute o script `test.py` para rodar os testes:
    ```bash
    python test.py
    ```

## Alterando a LLM (OpenAI ou Llama3)

Para alterar a LLM utilizada no `app.py`, siga as instruções abaixo:

1. No arquivo `app.py`, você encontrará as importações dos módulos `openai_utils.py` e `llama_utils.py`. Dependendo da LLM que você deseja usar, altere a importação conforme necessário.

2. Para usar OpenAI:
    ```python
    from openai_utils import (load_comments, analyze_sentiment, print_header)
    ```

3. Para usar Llama3:
    ```python
    from llama_utils import (load_comments, analyze_sentiment, print_header)
    ```

4. Certifique-se de que as funções `load_comments`, `analyze_sentiment` e `print_header` estejam implementadas corretamente nos respectivos arquivos utilitários (`openai_utils.py` ou `llama_utils.py`).

## Output

================================================================================
Análise de Sentimento
================================================================================

Análise dos sentimentos:
- Positivo: 2
- Negativo: 2
- Neutro: 1

Negativos:
- Aplicativo patético. Não funciona de jeito nenhum, fica carregando infinito diversas vezes que você tentar entrar.
- Não consigo adicionar um novo cartão, já entrei em contato mas eles não conseguem resolver o problema.

Positivos:
- Ótimo aplicativo, funciona bem.
- A independência e os benefícios valem cada centavo investido.

Neutros:
- Opção boa para uso na área externa.

Análise para o Conselho Executivo:
Considerando a análise dos comentários dos usuários, podemos observar que o aplicativo possui uma divisão equilibrada entre comentários positivos e negativos. Enquanto alguns usuários elogiaram o funcionamento e os benefícios do aplicativo, outros relataram problemas de carregamento e dificuldades no atendimento ao cliente. É importante avaliar as áreas em que o aplicativo está se destacando positivamente e buscar soluções para os problemas apontados pelos usuários. Investir em melhorias na usabilidade e no suporte ao cliente pode contribuir para uma experiência mais satisfatória e fidelização dos usuários.

Resultados salvos em `android_sentiment_analysis.json`.
