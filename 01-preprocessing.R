library(readr)
library(purrr)
library(lubridate)
library(dplyr)
library(Matrix)

# Untar dataset -----------------------------------------------------------

# untar("nf_prize_dataset.tar.gz", exdir = "data-raw")
# untar("data-raw/download/training_set.tar", exdir = "data-raw")

# Files -------------------------------------------------------------------

files <- dir("data-raw/training_set/", full.names = TRUE)

parse_movie <- function(file) {
  movie_id <- read_lines(file, n_max = 1) %>% parse_number()
  df <- read_csv(file, skip = 1, col_names = c("uid", "rating", "date"), 
                 col_types = cols(
                   uid = col_integer(),
                   rating = col_integer(),
                   date = col_date(format = "")
                 ))
  df$mid <- movie_id
  df
}

df <- map_df(files, parse_movie)

# Creating 3 datasets -----------------------------------------------------
# 1) Netflix - 3 months
# 2) Netflix - 6 months
# 3) Netflix - 1 year
# 4) Netflix - FULL

# defining functions to split data and to transform to a sparse matrix

split_netflix_data <- function(df, min_date, max_date) {
  
  x <- df %>%
    filter(between(date, min_date, max_date))
  
  y <- df %>%
    filter(
      between(date, max_date + days(1), max_date + months(1) - days(1)),
      uid %in% unique(x$uid), 
      mid %in% unique(x$mid)
    )
  
  # split test and validation randomly
  ind_validation <- sample.int(nrow(y), nrow(y)/2)
  
  y_test <- y[-ind_validation,]
  y_val <- y[ind_validation,]
  
  cat("Dataset Info \n")
  cat("min_date = ", as.character(min_date), " max_date = ", as.character(max_date), "\n")
  cat("Train -> #Users ", length(unique(x$uid)), " #Ratings ", nrow(x), "\n")
  cat("Test  -> #Users ", length(unique(y_test$uid)), " #Ratings ", nrow(y_test), "\n")
  cat("Valid -> #Users ", length(unique(y_val$uid)), " #Ratings ", nrow(y_val), "\n")
  
  list(x = x, y_val = y_val, y_test = y_test)
} 

to_sparse <- function(netflix_data) {
  
  x <- netflix_data$x
  y_val <- netflix_data$y_val
  y_test <- netflix_data$y_test
  
  
  uids <- data_frame(
    uid = unique(x$uid),
    user_id = row_number(uid)
  )
  
  mids <- data_frame(
    mid = unique(x$mid),
    movie_id = row_number(mid)
  )
  
  x <- x %>%
    left_join(uids, by = "uid") %>%
    left_join(mids, by = "mid")
  
  y_val <- y_val %>%
    left_join(uids, by = "uid") %>%
    left_join(mids, by = "mid")
  
  y_test <- y_test %>%
    left_join(uids, by = "uid") %>%
    left_join(mids, by = "mid")
  
  x <- sparseMatrix(x$user_id, x$movie_id, x = x$rating)
  
  # validation data
  y_val <- sparseMatrix(y_val$user_id, y_val$movie_id, x = y_val$rating)
  ind_ratings <- which(rowSums(y_val) > 0)
  
  y_val <- y_val[ind_ratings,]
  x_val <- x[ind_ratings,]
  
  # test data
  y_test <- sparseMatrix(y_test$user_id, y_test$movie_id, x = y_test$rating)
  ind_ratings <- which(rowSums(y_test) > 0)
  
  y_test <- y_test[ind_ratings,]
  x_test <- x[ind_ratings,]
  
  list(
    train = list(x = x, y = x),
    val = list(x = x_val, y = y_val),
    test = list(x = x_test, y = y_test)
  )
}

netflix3m <- split_netflix_data(df, ymd("2005-09-01"), ymd("2005-11-30"))
netflix3m <- to_sparse(netflix3m)

saveRDS(netflix3m, "data/netflix3m.rds")







