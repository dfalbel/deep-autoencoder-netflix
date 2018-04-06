library(keras)
library(tensorflow)
library(tfdatasets)
library(Matrix)

K <- backend()
source("tied-dense-layer.R")

# Loading data ------------------------------------------------------------

netflix3m <- readRDS("data/netflix3m.rds")


FLAGS <- flags(
  flag_string("activation", "relu", "One of 'selu', 'relu', 'tanh', 'sigmoid', 'elu', 'lrelu'")
)

activation <- switch(
  FLAGS$activation,
  selu = layer_activation(activation = 'selu'),
  relu = layer_activation(activation = 'relu'),
  tanh = layer_activation(activation = 'tanh'),
  sigmoid = layer_activation(activation = 'sigmoid'),
  elu = layer_activation_elu,
  lrelu = layer_activation_leaky_relu
)

input <- layer_input(shape = ncol(netflix3m$train$x))

dense_1 <- layer_dense(units = 128)
dense_2 <- layer_dense(units = 256)
dense_3 <- layer_dense(units = 256)

dense_1_transposed <- layer_tied_dense(master_layer = dense_1)
dense_2_transposed <- layer_tied_dense(master_layer = dense_2)
dense_3_transposed <- layer_tied_dense(master_layer = dense_3)

output <- input %>%
  dense_1() %>%
  layer_activation("selu") %>%
  dense_2() %>%
  layer_activation("selu") %>%
  dense_3() %>%
  layer_activation("selu") %>%
  layer_dropout(0.65) %>%
  dense_3_transposed() %>%
  layer_activation("selu") %>%
  dense_2_transposed() %>%
  layer_activation("selu") %>%
  dense_1_transposed() %>%
  layer_activation("selu")
  
model <- keras_model(input, output)

masked_mse <- function(y_true, y_pred) {
  mask_true <- k_cast(k_not_equal(y_true, 0), k_floatx())
  masked_squared_error <- k_square(mask_true * (y_true - y_pred))
  masked_mse <- k_sum(masked_squared_error)/k_sum(mask_true)
  masked_mse
}

rmse <- function(y_true, y_pred) {
  masked_mse(y_true, y_pred) ^ 0.5
}


model %>%
  compile(
    loss = masked_mse, 
    metrics = list(rmse = rmse), 
    optimizer = optimizer_sgd(lr = 0.001, momentum = 0.9)
  )

sparse_generator <- function(set, batch_size) {

  x <- tf$SparseTensorValue(
    cbind(set$x@i, set$x@j),
    set$x@x,
    dense_shape = dim(set$x)
  ) %>%
    tf$sparse_reorder()

  y <- tf$SparseTensorValue(
    cbind(set$y@i, set$y@j),
    set$y@x,
    dense_shape = dim(set$y)
  ) %>%
    tf$sparse_reorder()

  x <- tensor_slices_dataset(x) %>%
    dataset_map(tf$sparse_tensor_to_dense)
  y <- tensor_slices_dataset(y) %>%
    dataset_map(tf$sparse_tensor_to_dense)

  zip_datasets(x, y) %>%
    dataset_repeat() %>%
    dataset_shuffle(100) %>%
    dataset_batch(batch_size)
}

# sparse_generator <- function (set, batch_size) {
#   obs <- NULL
#   function () {
#     if(length(obs) < batch_size) obs <<- seq_len(nrow(set$x))
#     batch_obs <- sample(obs, size = batch_size)
#     obs <<- obs[!obs %in% batch_obs]
#     
#     x <- as.matrix(set$x[batch_obs, ])
#     y <- as.matrix(set$y[batch_obs, ])
#     
#     list(x, y)
#   }
# }

model %>%
  fit_generator(
    sparse_generator(netflix3m$train, 128),
    epochs = 100,
    steps_per_epoch = nrow(netflix3m$train$x)/128,
    callbacks = callback_tensorboard()
  )


evaluate_generator(model, sparse_generator(netflix3m$test, batch_size = 128), steps = 1000)
