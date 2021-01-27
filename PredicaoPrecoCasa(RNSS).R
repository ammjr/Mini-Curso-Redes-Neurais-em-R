# Descrição de dados

#1.crim: taxa de criminalidade per capita por cidade.
#2.zn: proporção de terrenos residenciais divididos em lotes com mais de 25.000 pés quadrados.
#3.indus: proporção de acres comerciais/não comerciais por cidade.
#4.chas: Variável fictícia Charles River (= 1 se o trecho limita o rio; 0 caso contrário).
#5.nox: concentração de óxidos de nitrogênio (partes por 10 milhões).
#6.rm: número médio de quartos por habitação.
#7.era: proporção de unidades ocupadas pelos proprietários construídas antes de 1940.
#8.dis: média ponderada das distâncias para cinco centros de emprego em Boston.
#9.rad: índice de acessibilidade às rodovias radiais.
#10.imposto: taxa de imposto sobre a propriedade de valor total por \ $ 10.000.
#11.ptratio: proporção aluno-professor por cidade.
#12. black: 1000 (Bk - 0,63) ^ 2 onde Bk é a proporção de negros por cidade.
#13.lstat: menor status da população (por cento).
#14.medv: valor médio das casas ocupadas pelos proprietários em \ $ 1000s.
#
#https://www.kaggle.com/c/boston-housing/datasets
#1978
#--------------------------------------------------------------------------------
# Autor: Antonio Mendes M. Jr
# e-mail: jrjpmg@hotmail.com
##--------------------------------------------------------------------------------
##
library(RSNNS)

# carrega o dados
dados <- read.csv(file = "C:/Users/jrjpm/Desktop/Cursos Ministrados/RNA v2.0/Datasets/Price-House/housing.data", 
                    header=FALSE, sep="") 

# toma uma "amostra" aleatória do tamanho do próprio dados 
# (na prática apenas embaralha os dados/linhas do dados/dataframe)
dados <- dados[sample(1:nrow(dados),length(1:nrow(dados))),1:ncol(dados)]

# Separa as variáveis de entrada 
# e a variável resposta
inputs <- dados[-14]
outputs <- dados[14]

# Devide os dados treino/teste
dataset <- splitForTrainingAndTest(inputs, outputs, ratio=0.15)

# Normaliza os valores (inclusive os valores alvo)
dataset <- normTrainingAndTestSet(dataset, dontNormTargets = F, type="0_1")

# "linOut= T" seta a função de ativação da saída como linear
model <- mlp(dataset$inputsTrain, dataset$targetsTrain, learnFunc = "Rprop",
             inputsTest = dataset$inputsTest, targetsTest = dataset$targetsTest,
             size=c(30), maxit=50, linOut=T)

#names(model)
#model$IterativeFitError
#model$fitted.values
#model$fittedTestValues

# Erro iterativo do treinamento
plotIterativeError(model)

# Erro da regressão
plotRegressionError(dataset$targetsTrain, model$fitted.values, main="Regression Plot Fit")
plotRegressionError(dataset$targetsTest, model$fittedTestValues, main="Regression Plot Test")
# Histograma dos erros
hist(model$fittedTestValues - dataset$targetsTest, col="lightblue", main="Error Histogram Test")

# Busca qual foi os parãmetros usados na normalização da variável resposta
par_norm <- getNormParameters(normalizeData(dados[14], type='0_1'))

# Calcula os valores resposta para os dados de teste
predictions <- predict(model, dataset$inputsTest)

# Faz a volta da normalização para ficar na mesma escala original
predic_denorm <- denormalizeData(predictions, par_norm)
valor_real <- denormalizeData(dataset$targetsTest, par_norm)

# Organiza e mostra os resultados
resultados <- cbind(as.data.frame(valor_real),
                    as.data.frame(predic_denorm))
colnames(resultados) <- c("Valor real","Valor predito")
resultados

# Soma dos erros quadrados 1) antes de reverter a normalização 
# 2) depois de reverter a normalização
(erro_test <- sum((predictions-dataset$targetsTest)^2))
(erro_test <- sum((predic_denorm-valor_real)^2))
