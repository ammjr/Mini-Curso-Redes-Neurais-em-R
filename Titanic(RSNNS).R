# Dados referentes as mortes no naufrágio do TITANIC
# 
# Descrição de dados
#
# Sobreviveu?: Informa se o passageiro sobreviveu ao desastre. 0 = Não; 1 = Sim
# Classe: Classe na qual o passageiro viajou. 1 = Primeira Classe; 2 = Segunda Classe; 3 = Terceira Classe
# Nome: Nome do passageiro
# Sexo: Sexo do passageiro
# Idade: Idade do passageiro
# Irmãos/Cônjuge: Informa a quantidade de irmãos e cônjuges que o paciente possuía na embarcação
# Pais/Crianças: Quantidade de crianças e idosos (pais) relativos ao passageiro
# Tarifa: Valor da passagem
# Embarque: Local onde o passageiro embarcou
# 
# dados disponível em: 
# https://www.udacity.com/api/nodes/5420148578/supplemental_media/titanic-datacsv/download

#--------------------------------------------------------------------------------
# Autor: Antonio Mendes M. Jr
# e-mail: jrjpmg@hotmail.com
#--------------------------------------------------------------------------------

# Pacote para trabalhar com RNAs
library(RSNNS)

# Ler os dados
dados <- read.csv(file = "C:/Users/jrjpm/Desktop/Cursos Ministrados/RNA v2.0/datasets/Titanic/titanic_data.csv", header=T, sep = ",")
# Remover as colunas 1, 4, 9 e 11 (não são importantes para esse problema)
dados <- dados[c(-1,-4,-9,-11)]
colnames(dados) <- c("Sobreviveu", "Classe", "Sexo", "Idade", 
                       "Imaos/Conjuge", "Pais/Crianças", "Tarifa", "Embarque")

str(dados)


# Remover as linhas em que não há informação do local 
# de embarque (existem duas amostras assim)
dados <- dados[dados$Embarque!="",]
str(dados)
unique(dados$Embarque)

# Preenche com a média da idade os dados de idade faltando
dados$Idade[is.na(dados$Idade)] <- round(mean(na.omit(dados$Idade)))
# Transforma para numéricas as variaveis "sexo" e "embarque"
dados$Sexo <- as.numeric(dados$Sexo)
dados$Embarque <- as.numeric(dados$Embarque)

str(dados)


# Embaralhar o dados
dados <- dados[sample(1:nrow(dados),length(1:nrow(dados))), ]


# Separa as variáveis de entrada (dataValues)
# e de saída (dataTargets)
dataValues <- dados[,-1] 
dataTargets <- dados[,1]

# Codifica os dados de saída (dataTargets)
dataTargets <- decodeClassLabels(dataTargets)
colnames(dataTargets)=c("Morreu","Sobreviveu")

# Dividir o dataset em treino e teste
dataset <- splitForTrainingAndTest(dataValues, dataTargets, ratio=0.15)

# Normalizar os dados
dataset <- normTrainingAndTestSet(dataset)

#RSNNS::getSnnsRFunctionTable()
model <- mlp(dataset$inputsTrain, dataset$targetsTrain, size=c(5,2),
             hiddenActFunc='Act_TanH', learnFunc = "Rprop",
             outputActFunc = "Act_Logistic", maxit=5000)

#summary(model)

# Erro de treinamento do modelo
plotIterativeError(model)

# valores preditos pela rede treinada 
# (usado para verificar os dados de teste e novos dados)
predictions <- predict(model, dataset$inputsTest)


# Plotar Curva ROC
library(pROC)
par(pty='s')
data_test_code <- encodeClassLabels(dataset$targetsTest)
resp_test_code <- encodeClassLabels(predictions)
pROC::roc(data_test_code ~ resp_test_code,
          plot=T, print.auc=TRUE,
          auc.polygon=T, grid=TRUE, legacy.axes=T,
          ci = T)

# Cálculo do AUC
library(ModelMetrics)
# encodeClassLabels faz o contrário da decodeClassLabels, 
# tranformando a matriz em fatores
auc(encodeClassLabels(dataset$targetsTest), encodeClassLabels(predictions))

# plota as curvas ROC (número de verdadeiros positivos e falsos positivos)
plotROC(fitted.values(model), dataset$targetsTrain)
plotROC(predictions, dataset$targetsTest)

# matrizes de confusão (diagonal são os acertos)
(mc_treino <- RSNNS::confusionMatrix(dataset$targetsTrain, fitted.values(model)))
(mc_teste <- RSNNS::confusionMatrix(dataset$targetsTest, predictions))

# nível de acerto (dados de teste)
(nivel_acerto <- sum(diag(mc_teste))/sum(mc_teste))

#--------------------------------------------------------------
#
# plota a representação da RNA treinada
# NÃO UTILIZE PARA REDES MUITO GRANDES
library(NeuralNetTools)
plotnet(model, alpha_val=1, 
        x_names= c(''), y_names = c('B', 'M'),
        node_labs = TRUE, var_labs = TRUE, max_sp = TRUE,
        circle_col = 'white', bord_col = 'black',  circle_cex = 6)


# ==================================================
# Utilizando novos dados
# ==================================================

#classe, sexo (1h 2m), idade, irmaos/conjuge, pais, 
# tarifa e embarque (2C 3Q 4S)
new_person <- matrix(c(2,1,30,0,2,10,2), ncol=7)

#voce <- normalizeData(voce)
# 1 morreu   2 sobreviveu
(result_new_person <- predict(model, new_person))
which.max(result_new_person)

# plota a representação da RNA treinada
# NÃO UTILIZE PARA REDES MUITO GRANDES
library(NeuralNetTools)
plotnet(model, alpha_val=1, 
        x_names= c(''), y_names = c('B', 'M'),
        node_labs = TRUE, var_labs = TRUE, max_sp = TRUE,
        circle_col = 'white', bord_col = 'black',  circle_cex = 6)

