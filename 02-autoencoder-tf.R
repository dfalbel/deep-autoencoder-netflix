library(tensorflow)
library(purrr)
library(R6)


resolve_activation <- function(activation) {
  switch (activation,
    relu = tf$nn$relu,
    leaky_relu = tf$nn$leaky_relu,
    selu = tf$nn$selu,
    elu = tf$nn$elu,
    sigmoid = tf$nn$sigmoid,
    tanh = tf$nn$tanh
  )
}

initialize_weights <- function(layer_sizes, n_input) {
  
  encoder <- list(
    input_shape = c(n_input, layer_sizes[-length(layer_sizes)]),
    output_shape = layer_sizes,
    name = 1:length(layer_sizes) - 1
  )
  
  w_enc <- pmap(encoder, function(input_shape, output_shape, name) {
    list(
      w = tf$get_variable(
        paste0("encoder_w" , name), 
        shape = shape(input_shape, output_shape), 
        initializer = tf$contrib$layers$xavier_initializer(uniform = TRUE)
      ),
      b = tf$get_variable(
        paste0("encoder_b" , name), 
        shape = shape(output_shape), 
        initializer = tf$zeros_initializer()
      )
    )
  })
  
  decoder <- list(
    input_shape = rev(layer_sizes),
    output_shape = c(rev(layer_sizes)[-1], n_input),
    name = 1:length(layer_sizes) - 1
  )
  
  w_dec <- pmap(decoder, function(input_shape, output_shape, name) {
    list(
      w = tf$get_variable(
        paste0("decoder_w" , name), 
        shape = shape(input_shape, output_shape), 
        initializer = tf$contrib$layers$xavier_initializer(uniform = TRUE)
      ),
      b = tf$get_variable(
        paste0("decoder_b" , name), 
        shape = shape(output_shape), 
        initializer = tf$zeros_initializer()
      )
    )
  })
  
  list(
    w_enc = w_enc,
    w_dec = w_dec
  )
}

encode <- function(x, weights, activation, dropout_prob, training) {
  for (i in 1:length(weights$w_enc)) {
      w_enc <- weights$w_enc[[i]]
      x <- activation(tf$matmul(x, w_enc$w) + w_enc$b) 
  }
  print("hi")
  if (dropout_prob > 0) {
    x <- tf$layers$dropout(x, rate = dropout_prob, training = training)
  }
  x
}

decode <- function(x, weights, activation, last_layer_activations) {
  for (i in 1:length(weights$w_dec)) {
    w_dec <- weights$w_dec[[i]]
    x <- tf$matmul(x, w_dec$w) + w_dec$b
    if(i != length(weights$w_dec) | last_layer_activations) {
      x <- activation(x)
    }
  }
  x
}

Autoencoder <- R6Class(
  "Autoencoder", 
  public = list(
    
    loss = NULL,
    optimizer = NULL,
    sess = NULL,
    x = NULL,
    output = NULL,
    training = NULL,
    train_step = NULL,
    
    initialize = function(
      n_input,
      layer_sizes, 
      activation = "selu", 
      is_constrained = TRUE, 
      dropout_prob = 0, 
      last_layer_activations = TRUE
    ) {
      # Inputs placeholders
      self$x <- tf$placeholder(tf$float32, shape = shape(NULL, n_input))
      self$training <- tf$placeholder(tf$bool, shape = shape())
      # Weights
      weigths <- initialize_weights(layer_sizes, n_input)
      # Activation
      activation_fun <- resolve_activation(activation)
      # Building Model  
      encoded <- encode(
        self$x, 
        weigths, 
        activation = activation_fun, 
        dropout_prob = dropout_prob, 
        training = training
      )
      decoded <- decode(
        encoded, 
        weigths, 
        activation = activation_fun, 
        last_layer_activations = last_layer_activations
      )
      
      self$output <- decoded
      # Loss Function
      print("Defining the loss")
      masking <- tf$cast(tf$greater(self$x, 0), dtype = tf$float32)
      print(self$x)
      print(self$output)
      self$loss <- tf$reduce_sum(tf$square((self$x - self$output) * masking))/tf$reduce_sum(masking)
      print("Defining the optim")
      # Optimizer
      self$optimizer <- tf$train$MomentumOptimizer(
        learning_rate = 0.001, 
        momentum = 0.9
        )
      self$train_step <- self$optimizer$minimize(self$loss)
      # Init Ops
      init <- tf$global_variables_initializer()
      self$sess <- tf$Session()
      self$sess$run(init)
    },
    
    partial_fit = function(X) {
      x <- self$x
      training <- self$training
      r <- self$sess$run(
        list(self$loss, self$train_step), 
        feed_dict=dict(x = X, training = TRUE)
      )
      r[[1]] # return the loss
    },
    
    fit = function(X, dense_refeeding = 0, epochs = 1, batch_size = 128) {
      for(i in 1:epochs) {
        indices <- 1:nrow(X)
        loss <- 0
        step <- 0
        while(length(indices) > batch_size) {
          spl <- sample(indices, size = batch_size)
          x_spl <- as.matrix(X[spl,])
          
          step_loss <- self$partial_fit(x_spl)
          
          if(dense_refeeding > 0) {
            for(i in 1:dense_refeeding) {
              x <- self$x
              training <- self$training
              x_spl <- self$sess$run(self$output, feed_dict = dict(x = x_spl, training = FALSE))
              step_loss <- self$partial_fit(x_spl)
            }
          }
          
          step <- step + 1
          loss <- step_loss + loss
        }
        
        cat(sprintf("epoch %02d - loss: %04f", epoch, loss/step))
  
      }
    }
  )
)

netflix3m <- readRDS("data/netflix3m.rds")
X <- netflix3m$train$x

tf$reset_default_graph()
ae <- Autoencoder$new(n_input =  ncol(X), layer_sizes = c(128, 128), 
                      is_constrained = TRUE, activation = "elu")


ae$fit(X, dense_refeeding = 1, epochs = 5, batch_size = 128)

loss <- 0
for (step in 1:30000) {
  spl <- sample(1:nrow(X), size = 128)
  r <- ae$partial_fit(as.matrix(X[spl,]))
  loss <- loss + r
  if (step %% 20 == 0)
    cat(step, "-", loss/step, "\n")
}
