TiedDenseLayer <- R6::R6Class(
  "TiedDenseLayer",
  inherit = KerasLayer,
  public = list(
    
    master_layer = NULL,
    W = NULL,
    b = NULL,
    output_dim = NULL,
    
    initialize = function(output_dim, master_layer) {
      self$master_layer <- master_layer
    },
    
    build = function(input_shape) {
      self$W <- k_transpose(self$master_layer$weights[[1]])
      self$output_dim <- self$W$shape$as_list()[[2]]
      
      self$b <- self$add_weight(
        name = 'bias',
        shape = list(self$output_dim),
        initializer = initializer_constant(0),
        trainable = TRUE
      )
      
    },
    
    call = function(x, mask = NULL) {
      k_dot(x, self$W) + self$b
    },
    
    compute_output_shape = function(input_shape) {
      list(input_shape[[1]], self$output_dim)
    }
    
  )
)

layer_tied_dense <- function(object, master_layer, name = NULL, trainable = TRUE) {
  create_layer(TiedDenseLayer, object, list(
    master_layer = master_layer,
    name = name,
    trainable = trainable
  ))
}