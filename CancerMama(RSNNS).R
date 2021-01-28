# Informações do conjunto de dados:
# Os recursos são calculados a partir de uma imagem digitalizada de um aspirado 
# por agulha fina (PAAF) de uma massa mamária. 
# Eles descrevem características dos núcleos celulares presentes na imagem. 

# Informações sobre atributos:
# 1) número de identificação
# 2) Diagnóstico (M = maligno, B = benigno)
# 3-32)
# Dez recursos com valor real são calculados para cada núcleo celular:
# a) raio (média das distâncias do centro aos pontos do perímetro)
# b) textura (desvio padrão dos valores da escala de cinza)
# c) perímetro
# d) área
# e) suavidade (variação local no comprimento do raio)
# f) compactação (perímetro ^ 2 / área - 1,0)
# g) concavidade (severidade das partes côncavas do contorno)
# h) pontos côncavos (número de partes côncavas do contorno)
# i) simetria
# j) dimensão fractal ("aproximação da costa" - 1)

# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
# Arquivo: wdbc.data
#--------------------------------------------------------------------------------
# Autor: Antonio Mendes M. Jr
# e-mail: ammjr@outlook.com
##--------------------------------------------------------------------------------

#set.seed(42)

library(RSNNS) #carrega o pacote 

# carrega o dataset
dataset <- read.csv(file = "C:/Users/jrjpm/Desktop/Cursos Ministrados/RNA v2.0/Datasets/Breast-Cancer/wdbc.data", header=FALSE) 
dataset <- dataset[-1] # elimina a primeira coluna (id do paciente)

# Estrutura dos dados
str(dataset)

# Toma uma "amostra" aleatória do tamanho do próprio dataset 
# (na prática apenas embaralha os dados/linhas do dataset/dataframe)
dataset <- dataset[sample(1:nrow(dataset),
                          length(1:nrow(dataset))),
                   1:ncol(dataset)]

# Separa o dataset, pegando apenas os valores 
# (não pegando a classificação da amostra)
dataValues  <- dataset[-1]

# Separa o dataset, pegando apenas os valores alvos (a classificação)
# decodeClassLabels é uma função que decodifica as classes em valores binarios
dataset[,1] <- as.numeric(dataset[,1])
dataTargets <- decodeClassLabels(dataset[,1])

# Divide o dataset entre dados de treinamento e teste
# ratio é a proporcao de dados para teste
dataset <- splitForTrainingAndTest(dataValues, dataTargets, ratio=0.15)

# normaliza os valores do dataset 
# (por default, targets NÃO são normalizados)
dataset <- normTrainingAndTestSet(dataset)


#RSNNS::getSnnsRFunctionTable()  -> retorna uma lista de funções do pacote RSNNS
# mlp é o tipo de rede neural (multilayer perceptron), o pacote RSNNS suporta vários modelos
# os parâmetros obrigatórios são: dados de treinamento, targets dos dados de treinamento,
# quantidade de neurônios da(s) camada(s) escondida(s).
#
# Alguns dos parâmetros opcionais são: máximo de iterações, função de ativação dos neurônios
# das camadas escondidas (geralmente sigmoide ou tangente hiperbolica) 
# e da camada de saída (geralmente linear/identidade para regressão e logistica ou softmax
# para classificação )
# algoritmo de aprendizagem (SCG=gradiente conjugado escalonado e Rprop = resiliente backpropagation)
# use "help mlp" para verificar mais parâmetros
model <- mlp(dataset$inputsTrain, dataset$targetsTrain, size=c(5),
             learnFunc="SCG", maxit=100, shufflePatterns=T, 
             hiddenActFunc = "Act_Logistic", outputActFunc = "Act_Logistic")

# plota o erro em função das iterações
plotIterativeError(model)

# valores preditos pela rede treinada 
# (usado para verificar os dados de teste e novos dados)
predictions <- predict(model, dataset$inputsTest)


library(pROC)
par(pty='s')
data_test_code <- encodeClassLabels(dataset$targetsTest)
resp_test_code <- encodeClassLabels(predictions)
pROC::roc(data_test_code ~ resp_test_code,
               plot=T, print.auc=TRUE,
               auc.polygon=T, grid=TRUE, legacy.axes=T,
                ci = T)


# plota as curvas ROC (número de verdadeiros positivos e falsos positivos)
plotROC(fitted.values(model), dataset$targetsTrain)
plotROC(predictions, dataset$targetsTest)

# matrizes de confusão (diagonal são os acertos)
(mc_treino <- RSNNS::confusionMatrix(dataset$targetsTrain, fitted.values(model)))
(mc_teste <- RSNNS::confusionMatrix(dataset$targetsTest, predictions))

# nível de acerto (dados de teste)
(nivel_acerto <- sum(diag(mc_teste))/sum(mc_teste))

# plota a representação da RNA treinada
# NÃO UTILIZE PARA REDES MUITO GRANDES
library(NeuralNetTools)
plotnet(model, alpha_val=1, 
        x_names= c(''), y_names = c('B', 'M'),
        node_labs = TRUE, var_labs = TRUE, max_sp = TRUE,
        circle_col = 'white', bord_col = 'black',  circle_cex = 6)

