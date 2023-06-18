

eval_forecasts <- function(y, fc, add.naive = TRUE, n.ahead = 1) {
  # eval substitute to get the name of the forecast if only a vector is passed
  mfc <- eval(substitute(qDF(fc))) 
  lagy <- flag(y, n.ahead)
  if (add.naive) mfc <- c(list(Naive = lagy), mfc)
  if (!all(length(y) == lengths(mfc))) 
    stop("All supplied quantities must be of equal length")
  res <- vapply(mfc, function(fcy) {
    # Preparation
    cc <- complete.cases(y, fcy)
    y <- y[cc]
    fcy <- fcy[cc]
    lagycc <- lagy[cc]
    n <- sum(cc)
    # Undo Bessel's correction: (n-1) instead of n in denominator
    nobessel <- sqrt((n - 1) / n) 
    sdy <- sd(y) * nobessel
    sdfcy <- sd(fcy) * nobessel
    diff <- fcy - y
    # Calculate Measures
    bias <- sum(diff) / n          # Bias
    MSE <- sum(diff^2) / n         # Mean Squared Error
    BP <- bias^2 / MSE             # Bias Proportion
    VP <- (sdy - sdfcy)^2 / MSE    # Variance Proportion
    CP <- 1 - BP - VP              # Covariance Proportion
    # CP <- 2 * (1 - cor(y, fcy)) * sdy * sdfcy / MSE
    RMSE <- sqrt(MSE)              # Root MSE
    R2 <- 1 - MSE / sdy^2          # R-Squared
    SE <- sd(diff) * nobessel      # Standard Forecast Error
    MAE <- sum(abs(diff)) / n      # Mean Absolute Error
    MPE <- sum(diff / y) / n * 100 # Mean Percentage Error
    MAPE <- sum(abs(diff / y)) / n * 100 # Mean Absolute Percentage Error
    U1 <- RMSE / (sqrt(sum(y^2) / n) + sqrt(sum(fcy^2) / n))   # Theils U1
    U2 <- sqrt(fmean((diff/lagycc)^2) / fmean((y/lagycc-1)^2)) # Theils U2 
    # Output
    return(c(Bias = bias, MSE = MSE, RMSE = RMSE, `R-Squared` = R2, SE = SE,
             MAE = MAE, MPE = MPE, MAPE = MAPE, U1 = U1, U2 = U2,
             `Bias Prop.` = BP, `Var. Prop.` = VP, `Cov. Prop.` = CP))
  }, numeric(13))
  attr(res, "naive.added") <- add.naive
  attr(res, "n.ahead") <- n.ahead
  attr(res, "call") <- match.call()
  class(res) <- "eval_forecasts"
  return(res)
}

# Print method
print.eval_forecasts <- function(x, digits = 3, ...) print.table(round(x, digits))

small_eval_forecasts <- function(y, fc, ...) {
  res <- eval_forecasts(unattrib(y), fc, ...)
  copyMostAttrib(res[c("Bias", "MAE", "RMSE", "R-Squared", "U2", "Bias Prop.", "Var. Prop.", "Cov. Prop."), ], res)
}
