# 0.0. Fazendo previsões com ARIMA

ARIMA no português significa auto-regressivo integrado de médias móveis. Utilizado em análises temporais, este modelo estatístico utiliza o passado para prever o futuro através de autocorrelação e média móveis. Modelos ARIMA são utilizados em alguns casos em que os dados mostram evidências de não estacionariedade.

# 1.0. Dataset
O dataset escolhido foi retirado do Kaggle, ele simula uma produção de eletricidade durante o período de 1985-01-01 até 2018-01-01, possuindo 397 entradas. Este conjunto de dados é referente a produção elétrica através contabilizada mensalmente, mas como é um modelo para estudo não existe muitos detalhes sobre sua origem.
Como o objetivo aqui é apenas traçar um passo a passo para fazer predições, não irei me atentar a isso.

O dataset pode ser encontrado neste endereço: [KAGGLE]('https://www.kaggle.com/shenba/time-series-datasets')


## 1.1. Carregando o dataset

Antes de iniciar qualquer etapa, é necessário indexar a sequência com as datas. Indexar uma série temporal torna intuitiva a busca de dados ao decorrer de eventos, como dias, mês, horas etc.

`df = pd.read_csv('dataset/Electric_Production.csv', parse_dates=['DATE'], index_col='DATE')`

* `parse_dates` transforma a coluna no formato datetime;
* `index_col` faz com que essa coluna vire o index.

# 2.0. Inspecionando os Dados
A primeira coisa a ser feita depois de carregar os dados é inspeciona-lo, verificar se possui dados nulos, observações incomuns, padrões, mudanças ao longo do tempo e relações entre as variáveis. Podemos fazer isso com alguns comandos da biblioteca Pandas. Assim:

* O dataset não possui dados nulos;
* Valor máximo de 129.40;
* Valor mínimo de 55.32;
* Média de 88.84;
* Os dados estão em series temporais e está distribuída de mês em mês.

As primeiras 5 linhas do dataset:

![tabela01]('img/tabela01a.png')
          
Para facilitar a visualização, vamos visualizar o gráfico, permitindo uma melhor visualização:
![plot01]('img/fig01.png')

# 3.0. Conferindo estacionariedade com Dickey-Fuller Test

O teste de Dickey-Fuller foi criado para verificar se um modelo autoregressivo é ou não estacionário (se os parâmetros estão variando com o tempo), a vantagem de trabalhar com modelos estacionários é que teremos menos parâmetros para avaliar. Podemos fazer isso definindo uma hipótese nula e uma alternativa:
* Hipótese nula: *Esta série temporal não é estacionária, ou seja, não depende do tempo*
* Hipótese alternativa: *Esta série temporal é estacionária. Apresenta uma tendência dependente do tempo*

Não iremos abordar as definições para validar uma hipótese, mas queremos um valor **p** maior que o nível de significância 0.05.
* valor de p = 0.186
Como p > 0.05 podemos dizer que não temos razão para rejeitar a hipótese nula e que nosso modelo é não estacionário. Nosso próximo passo é torna-la estacionária.

## 3.1. Convertendo o dataset para estacionário
Para realizar essa transformação, seguiremos por duas etapas, iremos utilizar a biblioteca numpy para fazer uma transformação logarítmica e logo em seguida iremos fazer uma diferença entre o período de dois valores adjacentes.

![tabela02]('img/tabela02.png')

Ao fazer esta diferença, é preciso lembrar que ela deixa valores nulos (NaN values), sendo importante utilizar o **dropna()** para remove-los:
`df = df.dropna()`

Com isso obtemos um valor de p de 3.25e-09, o que é pequeno o suficiente para rejeitarmos a hipótese nula e finalmente termos uma série estacionária.

# 4.0. Decomposição Sazonal
Aqui iremos responder as seguintes questões
* Sazonalidade: os dados exibem um padrão periódico claro?
* Tendência: os dados seguem uma tendência consistente para cima ou inclinação para baixo?
* Ruído: existem pontos outlier ou valores ausentes que são não é consistente com o resto dos dados?
Utilizando a biblioteca statsmodels podemos utilizar a `seasonal_decompose` para visualizar essas questões, mostrado no figura abaixo:
![plot02]('img/fig02.png')

Temos um padrão cíclico, o que demonstra que certos períodos do mês temos uma alta produção energética e outros com baixa. Uma tendência para cima e poucos ruídos.

# 5.0. Fazendo previsões
Inicialmente vamos dividir o dataset em treino(train) e teste(test), para que possamos validar as nossas predições. No caso eu vou treinar com os dados do período de 1985 até 2016 e comparar os resultados obtidos com o período entre 2017 e 2018.
Ficamos com o treino no formato (383,6) e teste (13, 6), lembrando que está na ordem linha x coluna.

Vamos utilizar dois modelos estatísticos, o ARIMA e o SARIMA. O SARIMA é uma vertente do ARIMA e permite identificar e considerar a sazonalidade.
Utilizando a biblioteca statsmodel:
`from statsmodels.tsa.statespace.sarimax import SARIMAX`

Mais detalhes aqui:
https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

## 5.1. Termos ARIMA

Para encontrarmos os melhores modelos, precisamos conhecer os parâmetros, que s]ao chamados de "termos", as três principais são: p, d e q.

Existem diversas formas de encontrar estes termos e a quantidade de números, como funções ACF e PACF, mas aqui iremos aproveitar o poder de um programador e utilizar uma biblioteca para fazer esse teste pra gente.
`import pmdarima as pm`
Para quem é familiarizado com machine learning, essa biblioteca opera como uma espécie de gridsearch, você coloca os termos que deseja testar e ela vai fazendo combinações até encontrar a melhor resposta, no caso do nosso modelo ARIMA foi esse daqui:

![tabela03]('img/tabela01.png')


Para validar nosso modelo, iremos utilizar a métrica MAE (Mean Absolute Error) representa a média da diferença absoluta entre os valores reais e previstos no conjunto de dados. Ele mede a média dos resíduos no conjunto de dados.

The Mean Absolute Error for ARIMA: 0.028
The Mean Absolute Error for SARIMA: 0.033

Nosso modelo ARIMA mostrou melhor desempenho.

Com o modelo pronto podemos fazer previsões:

![plot04]('img/fig04.png') 

