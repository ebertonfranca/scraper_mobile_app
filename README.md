# scraper_mobile_app 

Agente de SW 3.0 para análise e insights sobre a opinião de usuários do VivoEasy na Google Play e App Store

Pré-requisitos - Instalar dependências

pip install pandas
pip install transformers
pip install nltk
pip install langchain
pip install langchain_openai
pip install python-dotenv
pip install tensorflow
pip install torch torchvision torchaudio
pip install flax

Inserir variável de ambiente para a Key da OpenAI

CMD:
setx OPENAI_API_KEY "your-key"

OU

Criar um arquivo .env e inserir o seguinte comando interno:
OPENAI_API_KEY=your-key

Como rodar

Gerar reviews da google play e app store salvando em .csv:

python scraper_android_ios.py

Fazer análise de sentimento:

python code.py <case_of_use|all> <google_play_csv_file> <app_store_csv_file>

Exemplo:
python sentiment_analysis.py analise_sentimento google_play_reviews.csv app_store_reviews.csv

Output:

===============================================================================
Sentiment Analysis
===============================================================================
### Sentiment Analysis of Comments:

1. Muito bom rápido prático amoooo - **Positive**
2. Aplicativo precisa de internet pra usar… agora como comprar internet… sem ter dados no celular? - **Negative**
3. Muito bom app - **Positive**
4. muito bom plano e app, pra cashback pode usar meu codigo RBR1975 - **Positive**
5. Não consigo renovar assinatura - **Negative**
6. Suporte Técnico Ajuda Atendimento.. não existe - **Negative**

Análise para o conselho executivo:
A análise dos comentários sobre o aplicativo Vivo Easy mostra que a maioria dos usuários tem tido experiências semelhantes...
