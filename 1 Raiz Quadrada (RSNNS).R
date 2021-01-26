# Autor: Antonio Mendes M Jr
# 20/01/21

# Setar uma semente para gerar sempre os mesmos
# números aleatórios
set.seed(5)

# Gerados dados para treinamento
dados_input <-  as.data.frame(round(runif(200, min=0, max=500)));
colnames(dados_input) <- c('Valores'); head(dados_input) 

# Raiz quadrada dos valores gerados
dados_output <- sqrt(dados_input)
colnames(dados_input) <- c('Raiz Quadrada'); head(dados_output)


# Biblioteca para RNAs
library(RSNNS)

# Divide o dataset entre dados de treinamento e teste
# ratio é a proporcao de dados para teste
dataset <- splitForTrainingAndTest(dados_input, dados_output, ratio=0.15)

# normaliza os valores do dataset 
# (por default, targets NÃO são normalizados)
#dataset <- normTrainingAndTestSet(dataset)


# Cria um modelo de RNAs MLP 
# "linOut= T" seta a função de ativação da saída como linear
model <- mlp(dataset$inputsTrain, dataset$targetsTrain, learnFunc = "Rprop",
             inputsTest = dataset$inputsTest, targetsTest = dataset$targetsTest,
             size=c(10), maxit=50, linOut=T)


plotIterativeError(model)


new_data <- as.data.frame((1:10)^2)
colnames(new_data) <- c('Novos valores'); head(new_data)

# Verifica as respostas para o conjunto de teste
resultados <- predict(model, new_data)


# Organiza e mostra os resultados
resultados <- cbind(new_data, sqrt(new_data), resultados)
colnames(resultados) <- c("Entrada","Saída esperada","Saída da rede")
resultados

(erro_test <- sum((sqrt(new_data)-resultados[3])^2))


library(NeuralNetTools)
plotnet(model, alpha_val=1, 
        x_names = c('Valor'), y_names = c('Raiz'),
        node_labs = TRUE, var_labs = TRUE,
        circle_col = 'white', bord_col = 'black',  circle_cex = 6)

