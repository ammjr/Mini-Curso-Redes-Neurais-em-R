# Informações do conjunto de dados:
  
# O grupo examinado compreendeu grãos pertencentes a 3 variedades de trigo: Kama, Rosa e Canadense, 
# 70 elementos cada, selecionados aleatoriamente para o experimento. 
# Foi utilizada a técnica de raio-X para verificar o interior dos grãos

# O conjunto de dados pode ser usado para as tarefas de classificação e análise de cluster.

# Informações sobre atributos:
#Para construir os dados, sete parâmetros geométricos dos grãos de trigo foram medidos:
#1. área A
#2. perímetro P,
#3. compacidade C = 4 * pi * A / P ^ 2,
#4. comprimento do núcleo,
#5. largura do núcleo,
#6. coeficiente de assimetria
#7. comprimento do sulco do núcleo.
# Todos esses parâmetros são contínuos, com valores reais
#
# Dataset disponível em: https://archive.ics.uci.edu/ml/datasets/seeds
# #--------------------------------------------------------------------------------
# Autor: Antonio Mendes M. Jr
# e-mail: jrjpmg@hotmail.com
##--------------------------------------------------------------------------------

library(RSNNS)

# Lendo o dataset
dataset <- read.csv(file = "C:/Users/jrjpm/Desktop/Cursos Ministrados/RNA v2.0/Datasets/Sementes-Trigo/seeds_dataset.txt", header=F, sep = "")

# Embaralhando o dataset
dataset <- dataset[sample(1:nrow(dataset),length(1:nrow(dataset))),1:ncol(dataset)]

# Colocando como fator a veriável resposta
dataset$V8= as.factor(dataset$V8) 

# Separando o dataset em variáveis preditoras e variável resposta
dataValues <- dataset[-8] # valores sao todas as colunas, menos a última
dataTargets <- decodeClassLabels(dataset[,8]) # codificando (one-hot) a variável resposta

# Dividindo o dataset em treino/teste
dataset <- splitForTrainingAndTest(dataValues, dataTargets, ratio=0.15)

# Normalizando o dataset
dataset <- normTrainingAndTestSet(dataset, type = "0_1")

#RSNNS::getSnnsRFunctionTable()
model <- mlp(dataset$inputsTrain, dataset$targetsTrain, size=c(10,3),
             hiddenActFunc='Act_TanH', learnFunc = "Rprop",
             outputActFunc = "Act_Logistic", maxit=1000, linOut = FALSE)

summary(model)

plotIterativeError(model)

# Predições em relação aos dados de teste
predictions <- predict(model, dataset$inputsTest)

plotROC(fitted.values(model), dataset$targetsTrain)
plotROC(predictions, dataset$targetsTest)


library(pROC)
par(pty='s')
data_test_code <- encodeClassLabels(dataset$targetsTest)
resp_test_code <- encodeClassLabels(predictions)
pROC::roc(data_test_code ~ resp_test_code,
          plot=T, print.auc=TRUE,
          auc.polygon=T, grid=TRUE, legacy.axes=T,
          ci = T)

(mc_treino <- RSNNS::confusionMatrix(dataset$targetsTrain, fitted.values(model)))
(mc_teste <- RSNNS::confusionMatrix(dataset$targetsTest, predictions))


(nivel_acerto <- sum(diag(mc_teste))/sum(mc_teste))


# Plota a representação da RNA treinada
# NÃO UTILIZE PARA REDES MUITO GRANDES
library(NeuralNetTools)
plotnet(model, alpha_val=1, 
        x_names= c(''), y_names = c('Kama', 'Rosa', 'Canadian'),
        node_labs = TRUE, var_labs = TRUE, max_sp = TRUE,
        circle_col = 'white', bord_col = 'black',  circle_cex = 6)


