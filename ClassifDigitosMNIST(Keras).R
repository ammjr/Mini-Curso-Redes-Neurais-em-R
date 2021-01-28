# Utilização da biblioteca Keras para construir uma rede neural
# para classificação de digitos manuscritos do dataset MNIST
#

library(keras)

mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape (passar matrix para vetor)
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# rescale (normalizar os dados)
x_train <- x_train / 255
x_test <- x_test / 255

# transformar a classificação em categórica
# (vetores binários)
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


# Criação de um modelo sequencial com camadas densas (totalmente conectadas)
# intercaladas com camadas dropout
model <- keras_model_sequential() 
model %>% 
   layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
   layer_dropout(rate = 0.4) %>% 
   layer_dense(units = 128, activation = 'relu') %>%
   layer_dropout(rate = 0.3) %>%
   layer_dense(units = 10, activation = 'softmax')

summary(model)

model %>% compile(
   loss = 'categorical_crossentropy',
   optimizer = optimizer_rmsprop(),
   metrics = c('accuracy')
)


history <- model %>% fit(
   x_train, y_train, 
   epochs = 30, batch_size = 128, 
   validation_split = 0.2
)

plot(history)


model %>% evaluate(x_test, y_test)

classes <- model %>% predict_classes(x_test)

# Matriz de confusão
table(mnist$test$y, classes)


# Curva ROC
library(pROC)
dev.off()
plot.new()
par(pty='s')
multiclass.roc(mnist$test$y ~ classes,
    plot=TRUE,print.auc=TRUE,
    auc.polygon=TRUE, grid=TRUE, legacy.axes=T)

# Outra forma (mais completa) de obter a matriz de confusão
library(caret)
caret::confusionMatrix(as.factor(mnist$test$y),as.factor(classes))
