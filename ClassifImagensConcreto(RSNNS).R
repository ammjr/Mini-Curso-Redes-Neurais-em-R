# Dataset composto por imagens de superfíceis de concreto c/ e sem rachaduras
# Dataset disponível em: https://www.kaggle.com/arunrk7/surface-crack-detection
# Autor: Antonio Mendes
# Data: 16/01/21
# =============================================================
# A abordagem adotada utiliza os valores dos pixels diretamente como variáveis de entrada.
# O ideal seria extrair informações das imagens (como texturas) e utilizar essas informações
# como entrada ou utilizar CNNs para a classificação de imagens. Entretanto o resultado obtido
# é satisfatório.
# ==============================================================


library(imager) # carrega o pacote "imager", necessário para fazer processamento de imagens
library(RSNNS) # Carrega o pacote p/ trabalhar com RNAs


# Diretorio Base onde estão as pastas
diret_base <- 'C:/Users/jrjpm/Desktop/Cursos Ministrados/RNA v2.0/Datasets/Concrete-Crack-Images-for-Classification/Treino'

# Distagem das subpastas que estão no diretório base
list_diret <- list.dirs(diret_base, recursive = F)
dim_image <- 2 # Dimensões  das imagens  (que se deseja trabalhar) (2 p/ PB e 3 p/ colorida)
porct_utilizada <- 0.05 # Porcentagem do dataset a ser utilizado


# auxiliares
targets <- c()
list_arqt <- c()

# Faz as leituras das pastas e lista os arquivos e 
# seus endereços (ATENTE-SE AO FORMATO DAS IMAGENS)
for (i in 1:length(list_diret)){   
   list_arq <- list.files(path = list_diret[i],
                          pattern = "*.jpg" ,
                          full.names = T)
   list_arqt <- c(list_arqt, list_arq[1:(length(list_arq)*porct_utilizada)])
   targets = c(targets, rep(i, length(list_arq)*porct_utilizada))
}


# Se quer a imagem for em PB
if (dim_image==2){
   # Lê e carrega as imagens, transforma pra escala de cinza e aplica redimensionamento 
   arquivos <- list_arqt %>% lapply(load.image) %>% lapply(grayscale) %>% lapply(resize, size_x = 22, size_y = 22)
   
   # Calcula a quantidade de variáveis de entrada
   qtde_var <- dim(arquivos[[1]])[1] * dim(arquivos[[1]])[2]
   
   # Calcula a quantidade de amostras
   qtde_amostras <- length(arquivos)
   
   # Cria um vetor auxliar
   dados <- matrix(nrow = qtde_amostras, ncol= qtde_var*1)
   
   for (i in 1:length(arquivos)) {
      #lê a imagem e "captura" os valores dos pixels, passando para uma matriz
      #se fosse uma imagem RGB seria [,,]
      aux_image = arquivos[[i]][ ,] 
      # transforma a matriz (de pixels) em um vetor (uma única linha)
      aux_image = t(as.vector(aux_image))
      # a primeira linha da matriz auxiliar "dados_b" 
      # recebe o vetor com todos os pixels da i-esima imagem
      dados[i, ] = aux_image
   }
} else { # Se for colorida
   # Lê e carrega as imagens e aplica redimensionamento 
   arquivos <- list_arqt %>% lapply(load.image) %>% lapply(resize, size_x = 22, size_y = 22)
   
   # Calcula a quantidade de variáveis de entrada
   qtde_var <- dim(arquivos[[1]])[1] * dim(arquivos[[1]])[2]
   
   # Calcula a quantidade de amostras
   qtde_amostras <- length(arquivos)
   
   # Cria um vetor auxliar
   dados <- matrix(nrow = qtde_amostras, ncol= qtde_var*3)
   
   for (i in 1:length(arquivos)) {
      #lê a imagem e "captura" os valores dos pixels, passando para uma matriz
      #se fosse uma imagem RGB seria [,,]
      aux_image = arquivos[[i]][ , ,] 
      # transforma a matriz (de pixels) em um vetor (uma única linha)
      aux_image = t(as.vector(aux_image))
      # a primeira linha da matriz auxiliar "dados_b" 
      # recebe o vetor com todos os pixels da i-esima imagem
      dados[i, ] = aux_image
   }
}

# Plota uma das imagens 
plot(arquivos[[1]])


# Limpa a memória deixando apenas as variaveis
# importantes no momento
rm(list=setdiff(ls(), c("dados","targets")))




# toma uma "amostra" aleatoria do tamanho do dataset utilizado
# na prática, isso serve apenas para embaralhar os dados
# a variavel indice receberá valores de 1 ao número de amostras, 
# dipostos de forma aleatoria
indices <- sample(1:nrow(dados), nrow(dados))


# pega os valores do dataset, utilizando a variavel "indices" como indice das linhas
# do dataset (embaralhando os dados)
dataValues <- dados[indices, ]

# decodeClassLabels é uma função que decodifica as classes em valores binarios
# pega os valores do vetor targets, utilizando a variavel "indices" como indice (embaralha)
targets <- decodeClassLabels(targets)
dataTargets <- targets[indices, ]

# divide o dataset entre dados de treinamento e teste
# ratio é a proporcao de dados para teste
dataset <- splitForTrainingAndTest(dataValues, dataTargets, ratio = 0.15)

# normaliza os valores do dataset (por default, targets NÃO são normalizados)
dataset <- normTrainingAndTestSet(dataset, type = "0_1")

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
model <- mlp(
   dataset$inputsTrain,
   dataset$targetsTrain,
   size = c(30),
   learnFunc = "Rprop",
   maxit = 1000,
   shufflePatterns = T,
   outputActFunc = "Act_Logistic"
)

#summary(model)
#model
#weightMatrix(model)
#extractNetInfo(model)

# plota o erro (SSE) em função das iterações
plotIterativeError(model)

# valores preditos pela rede treinada (usado para verificar os dados de teste e novos dados)
predictions <- predict(model, dataset$inputsTest)

# matrizes de confusão (diagonal são os acertos)
(mc_treino <- RSNNS::confusionMatrix(dataset$targetsTrain, fitted.values(model)))
(mc_teste <- RSNNS::confusionMatrix(dataset$targetsTest, predictions))

# nível de acerto (dados de treino)
(nivel_acerto <- sum(diag(mc_treino)) / sum(mc_treino))

# nível de acerto (dados de teste)
(nivel_acerto <- sum(diag(mc_teste)) / sum(mc_teste))

#===============================================================================
# Até este ponto o código é geral para imagens em PeB e coloridas
# e utilizando problemas de duas ou mais classes. A partir desse ponto
# deve-se observar a curva ROC para os casos de duas ou mais classes,
# bem como no momento de "chamar" uma nova imagem para ser classificada
# ==============================================================================

# plota as curvas ROC (número de verdadeiros positivos e falsos positivos)
plotROC(fitted.values(model), dataset$targetsTrain)
plotROC(predictions, dataset$targetsTest)

# Plota curva ROC com AUC
library(pROC)
par(pty='s')
data_test_code <- encodeClassLabels(dataset$targetsTest)
resp_test_code <- encodeClassLabels(predictions)
pROC::roc(data_test_code ~ resp_test_code,
          plot=T, print.auc=TRUE,
          auc.polygon=T, grid=TRUE, legacy.axes=T,
          ci = T)

#-----------------------------------------------------------
# Para problemas multiclasses
# Plota curva ROC com AUC
library(pROC)
par(pty='s')
data_test_code <- encodeClassLabels(dataset$targetsTest)
resp_test_code <- encodeClassLabels(predictions)
pROC::multiclass.roc(data_test_code ~ resp_test_code,
          plot=T, print.auc=TRUE,
          auc.polygon=T, grid=TRUE, legacy.axes=T)


#========================================================================
# Verificar uma nova imagem
endereco <- 'C:/Users/jrjpm/Desktop/Cursos Ministrados/RNA v2.0/Datasets/Concrete-Crack-Images-for-Classification/N2/10015.jpg'
endereco <- 'C:/Users/jrjpm/Desktop/Cursos Ministrados/RNA v2.0/Datasets/Concrete-Crack-Images-for-Classification/S2/10001_1.jpg'

# Plotando a imagem lida
plot(load.image(endereco))

# Fazendo as mesmas transformações que foram aplicadas às imagens de treino
# na imagem a ser classificada (ATENÇÃO a necessidade do uso "grayscale" ou não - 
# deverá ser usado apenas se tiver sido usado nas imagens de treino, ter escolhido imagem
# em PeB no início do código)
image_teste <- endereco  %>% lapply(load.image) %>% 
   lapply(grayscale) %>% lapply(resize, size_x = 22, size_y = 22)

# Processamento para transformar a imagem em um único vetor
qtde_var_image_teste <- dim(image_teste[[1]])[1] * dim(image_teste[[1]])[2]
qtde_amostras_image_teste <- length(image_teste)
# Colocar *1 para imagens em PeB e *3 para imagens colorids (3 canais de cores, tem o triplo
# de variáveis)
dados_image_teste <- matrix(nrow = qtde_amostras_image_teste, ncol= qtde_var_image_teste*1)

#se fosse uma imagem RGB seria [,,]
aux_image = image_teste[[1]][ , ] 
# transforma a matriz (de pixels) em um vetor (uma única linha)
aux_image = t(as.vector(aux_image))
# a primeira linha da matriz auxiliar "dados_b" 
# recebe o vetor com todos os pixels da i-esima imagem
dados_image_teste[1, ] = aux_image

(predict_img <- predict(model, dados_image_teste))
which.max(predict_img[1,])
