library(fastverse)
fastverse_extend(xts, africamonitor, dfms, vars, glmnet)

prepcode <- function(x) toupper(sub(" ", "", x, fixed = TRUE))

data_m <- fread("data/FRED/MD/vintage_2023_02.csv") %>% 
          fmutate(V1 = am_as_date(V1)) %>% 
          frename(prepcode) %>% 
          fsubset(V1 >= "1990-01-01") %>% as.xts()
plot(fscale(data_m), lwd = 1)

data_q <- fread("data/FRED/QD/vintage_2023_02.csv") %>% 
          fmutate(V1 = am_as_date(V1)) %>% 
          frename(prepcode) %>% 
          fsubset(V1 >= "1990-01-01") %>% as.xts()
plot(fscale(data_q), lwd = 1)

groups <- fread("data/FRED/FRED-MD Appendix/FRED-MD_updated_appendix.csv") %>% 
          fmutate(fred = prepcode(fred))

# Check
ckmatch(colnames(data_m), groups$fred)

# Global Factors
ic <- ICr(data_m)
plot(ic)
screeplot(ic) # 2 Global factors is ok

VARselect(ic$F_pca[, 1:9]) # 4 lags

mod <- DFM(data_m, 2, 4)
plot(mod)
summary(mod)
plot(predict(mod))

# Global Model (for comparison)
glob_mod <- DFM(data_m, 9, 4)
plot(glob_mod)
summary(glob_mod)
plot(predict(glob_mod))

factors_glob <- glob_mod$F_qml %>% copyMostAttrib(data_m)
plot(fscale(factors_glob), lwd = 1)

# By Groups
data_m_groups <- groups %>% rsplit(fred ~ group_name) %>% lapply(function(x) data_m[, x])
data_m_groups %>% sapply(ncol)
ic_groups <- data_m_groups %>% lapply(ICr)

screeplot(ic_groups$`Consumption, Orders, and Inventories`) # 2 Factors
VARselect(ic_groups$`Consumption, Orders, and Inventories`$F_pca[, 1:2]) # 4 Lags
mod_coi <- DFM(data_m_groups$`Consumption, Orders, and Inventories`, 2, 4)

screeplot(ic_groups$Housing)  # 1 Factor
VARselect(ic_groups$Housing$F_pca[, 1]) # 2 Lags
mod_h <- DFM(data_m_groups$Housing, 1, 2)

screeplot(ic_groups$`Interest and Exchange Rates`) # 1-2 Factors
VARselect(ic_groups$`Interest and Exchange Rates`$F_pca[, 1]) # 3 lags
mod_ie <- DFM(data_m_groups$`Interest and Exchange Rates`, 2, 3)

screeplot(ic_groups$`Labor Market`) # 1 Factor
VARselect(ic_groups$`Labor Market`$F_pca[, 1]) # 1 lag
mod_l <- DFM(data_m_groups$`Labor Market`, 1, 1)

screeplot(ic_groups$`Money and Credit`) # 2 Factors
VARselect(ic_groups$`Money and Credit`$F_pca[, 1:2]) # 4 lags
mod_mc <- DFM(data_m_groups$`Money and Credit`, 2, 2)

screeplot(ic_groups$`Output and Income`) # 1 Factor
VARselect(ic_groups$`Output and Income`$F_pca[, 1]) # 2 lags
mod_oi <- DFM(data_m_groups$`Output and Income`, 1, 2)

screeplot(ic_groups$Prices) # 1 Factor
VARselect(ic_groups$Prices$F_pca[, 1]) # 1 lag 
mod_p <- DFM(data_m_groups$Prices, 1, 1)

screeplot(ic_groups$`Stock Market`) # 1 Factor
VARselect(ic_groups$`Stock Market`$F_pca[, 1]) # 1 lag
mod_sm <- DFM(data_m_groups$`Stock Market`, 1, 1)

# Putting factors together
factors <- cbind(
  mod$F_qml %>% add_stub("gl_"), 
  mod_coi$F_qml %>% add_stub("coi_"),
  mod_h$F_qml %>% add_stub("h_"),
  mod_ie$F_qml %>% add_stub("ie_"),
  mod_l$F_qml %>% add_stub("l_"),
  mod_mc$F_qml %>% add_stub("mc_"),
  mod_oi$F_qml %>% add_stub("oi_"),
  mod_p$F_qml %>% add_stub("p_"),
  mod_sm$F_qml %>% add_stub("sm_")
) %>% copyMostAttrib(data_m)

dim(factors)
plot(fscale(factors), lwd = 1)

factors_agg <- apply.quarterly(factors, mean)
index(factors_agg) %<>% lubridate::`month<-`(month(.)-2L)
rgdp_factors_agg <- factors_agg %>% merge(data_q[, 1] %>% setColnames("rgdp_growth"))

factors_wide <- cbind(factors %>% ss(month(index(.)) %% 3L == 1L) %>% add_stub("m1_"),
                      factors %>% ss(month(index(.)) %% 3L == 2L) %>% add_stub("m2_") %>% unclass(),
                      factors %>% ss(month(index(.)) %% 3L == 0L) %>% add_stub("m3_") %>% unclass())
rgdp_factors_wide <- factors_wide %>% merge(data_q[, 1] %>% setColnames("rgdp_growth"))

# Same for global model
factors_glob_agg <- apply.quarterly(factors_glob, mean)
index(factors_glob_agg) %<>% lubridate::`month<-`(month(.)-2L)
rgdp_factors_glob_agg <- factors_glob_agg %>% merge(data_q[, 1] %>% setColnames("rgdp_growth"))

factors_glob_wide <- cbind(factors_glob %>% ss(month(index(.)) %% 3L == 1L) %>% add_stub("m1_"),
                           factors_glob %>% ss(month(index(.)) %% 3L == 2L) %>% add_stub("m2_") %>% unclass(),
                           factors_glob %>% ss(month(index(.)) %% 3L == 0L) %>% add_stub("m3_") %>% unclass())
rgdp_factors_glob_wide <- factors_glob_wide %>% merge(data_q[, 1] %>% setColnames("rgdp_growth"))


# Linear Model
gdp_lm <- lm(rgdp_growth ~., qDF(rgdp_factors_agg))
summary(gdp_lm)
# plot(gdp_lm)
ts.plot(rgdp_factors_agg[, "rgdp_growth"])
lines(fitted(gdp_lm), col = "red")

# LASSO
cv_lasso <- cv.glmnet(x = qM(rgdp_factors_agg[, -ncol(rgdp_factors_agg)]), 
                      y = unattrib(rgdp_factors_agg[, "rgdp_growth"]), 
                      nfolds = nrow(rgdp_factors_agg), grouped = FALSE)
plot(cv_lasso)
fit_lasso_min <- cbind(icpt = 1, factors_agg) %*% as.matrix(coef(cv_lasso, s = "lambda.min"))
fit_lasso_1se <- cbind(icpt = 1, factors_agg) %*% as.matrix(coef(cv_lasso, s = "lambda.1se"))

# Wide LASSO
cv_lasso_wide <- cv.glmnet(x = qM(rgdp_factors_wide[, -ncol(rgdp_factors_wide)]), 
                           y = unattrib(rgdp_factors_wide[, "rgdp_growth"]), 
                           nfolds = nrow(rgdp_factors_wide), grouped = FALSE)
plot(cv_lasso_wide)
fit_lasso_wide_min <- cbind(icpt = 1, factors_wide) %*% as.matrix(coef(cv_lasso_wide, s = "lambda.min"))
fit_lasso_wide_1se <- cbind(icpt = 1, factors_wide) %*% as.matrix(coef(cv_lasso_wide, s = "lambda.1se"))

# ts.plot(rgdp_factors_agg[, "rgdp_growth"])
# lines(fitted(gdp_lm), col = "red")
# lines(fit_lasso_min, col = "blue")
# lines(fit_lasso_1se, col = "green")
# lines(fit_lasso_wide_min, col = "blue", lty = 2)
# lines(fit_lasso_wide_1se, col = "green", lty = 2)

fit_all <- rgdp_factors_agg[, "rgdp_growth"] %>% 
  cbind(lm = fitted(gdp_lm), lasso_min = unattrib(fit_lasso_min), lasso_1se = unattrib(fit_lasso_1se),
        wide_lasso_min = unattrib(fit_lasso_wide_min), wide_lasso_1se = unattrib(fit_lasso_wide_1se))

fit_all %>% plot(legend.loc = "topleft", lwd = 1, main = "DFM Prediction from Blocked DFM")

# Same for global model
gdp_lm_glob <- lm(rgdp_growth ~., qDF(rgdp_factors_glob_agg))
summary(gdp_lm_glob)
# plot(gdp_lm)
ts.plot(rgdp_factors_glob_agg[, "rgdp_growth"])
lines(fitted(gdp_lm_glob), col = "red")

# LASSO
cv_lasso_glob <- cv.glmnet(x = qM(rgdp_factors_glob_agg[, -ncol(rgdp_factors_glob_agg)]), 
                           y = unattrib(rgdp_factors_glob_agg[, "rgdp_growth"]), 
                           nfolds = nrow(rgdp_factors_glob_agg), grouped = FALSE)
plot(cv_lasso_glob)
fit_lasso_glob_min <- cbind(icpt = 1, factors_glob_agg) %*% as.matrix(coef(cv_lasso_glob, s = "lambda.min"))
fit_lasso_glob_1se <- cbind(icpt = 1, factors_glob_agg) %*% as.matrix(coef(cv_lasso_glob, s = "lambda.1se"))

# Wide LASSO
cv_lasso_glob_wide <- cv.glmnet(x = qM(rgdp_factors_glob_wide[, -ncol(rgdp_factors_glob_wide)]), 
                                y = unattrib(rgdp_factors_glob_wide[, "rgdp_growth"]), 
                                nfolds = nrow(rgdp_factors_glob_wide), grouped = FALSE)
plot(cv_lasso_glob_wide)
fit_lasso_glob_wide_min <- cbind(icpt = 1, factors_glob_wide) %*% as.matrix(coef(cv_lasso_glob_wide, s = "lambda.min"))
fit_lasso_glob_wide_1se <- cbind(icpt = 1, factors_glob_wide) %*% as.matrix(coef(cv_lasso_glob_wide, s = "lambda.1se"))

fit_all_glob <- rgdp_factors_glob_agg[, "rgdp_growth"] %>% 
  cbind(lm = fitted(gdp_lm_glob), lasso_min = unattrib(fit_lasso_glob_min), lasso_1se = unattrib(fit_lasso_glob_1se),
        wide_lasso_min = unattrib(fit_lasso_glob_wide_min), wide_lasso_1se = unattrib(fit_lasso_glob_wide_1se))

fit_all_glob %>% plot(legend.loc = "topleft", lwd = 1, main = "DFM Prediction from Global DFM")


# Evaluation
metrics <- function(x, y) c(r_squared = cor(x, y)^2, MAE = mean(abs(x - y)))
sapply(qDF(fit_all), metrics, unattrib(rgdp_factors_agg[, "rgdp_growth"]))
sapply(qDF(fit_all_glob), metrics, unattrib(rgdp_factors_glob_agg[, "rgdp_growth"]))
# -> lm is better for global model, but lasso on blocked model is best

#
# Now the Nowcast / Forecast ----------------------
#
VARselect(factors, lag.max = 15)
# Taking 2 lags
factor_VAR <- VAR(factors, 2)
factor_fcst <- predict(factor_VAR, n.ahead = 12) # 1 year ahead
# plot(factor_fcst, plot.type = "single")

factor_fcst_mat <- factor_fcst$fcst %>% lapply(function(x) x[, "fcst"]) %>% do.call(what = cbind) %>% 
  xts(order.by = seq(last(index(factors)), length.out = 13, by = "month")[-1L], frequency = 12)

factor_fcst_wide <- cbind(factor_fcst_mat %>% ss(month(index(.)) %% 3L == 1L) %>% add_stub("m1_"),
                          factor_fcst_mat %>% ss(month(index(.)) %% 3L == 2L) %>% add_stub("m2_") %>% unclass(),
                          factor_fcst_mat %>% ss(month(index(.)) %% 3L == 0L) %>% add_stub("m3_") %>% unclass())

fcst_lasso_wide_min <- cbind(icpt = 1, factor_fcst_wide) %*% as.matrix(coef(cv_lasso_wide, s = "lambda.min")) %>% 
  setColnames("fcst_lasso_wide_min")

# gdp_ts <- tsbox::ts_ts(rgdp_factors_agg[, "rgdp_growth"])
# ts.plot(gdp_ts, xlim = c(start(gdp_ts)[1], end(gdp_ts)[1]+2))
# lines(copyAttrib(fit_lasso_wide_min, gdp_ts), col = "red")
# lines(`tsp<-`(ts(fcst_lasso_wide_min), tsp(tsbox::ts_ts(factor_fcst_wide))), col = "red", lty = 2)

fcst_data <- rgdp_factors_agg[, "rgdp_growth"] %>% 
  cbind(lasso_wide_min = unattrib(fit_lasso_wide_min),
        lm = unattrib(fitted(gdp_lm))) %>% 
  merge(fcst_lasso_wide_min %>% copyMostAttrib(factor_fcst_wide)) %>%
  merge(cbind(icpt = 1, apply.quarterly(factor_fcst_mat, mean)) %*% coef(gdp_lm) %>% 
          copyMostAttrib(factor_fcst_wide) %>% setColnames("fcst_lm")) %>%
  ss(j = .c(rgdp_growth, lasso_wide_min, fcst_lasso_wide_min, lm, fcst_lm))

last_nmiss <- whichNA(fcst_data[, "fcst_lasso_wide_min"]) %>% last()
fcst_data[last_nmiss, "fcst_lasso_wide_min"] <- fcst_data[last_nmiss, "lasso_wide_min"]
fcst_data[last_nmiss, "fcst_lm"] <- fcst_data[last_nmiss, "lm"]

fcst_data %>% plot(legend.loc = "topleft", lwd = 1, main = "US GDP Nowcast from Blocked DFM")

# Same for global model
factor_glob_fcst <- predict(glob_mod, 12) # 1 year ahead
plot(factor_glob_fcst, xlim = c(350, 410), ylim = frange(factor_glob_fcst$F_fcst))

factor_glob_fcst_mat <- factor_glob_fcst$F_fcst %>% 
  xts(order.by = seq(last(index(factors)), length.out = 13, by = "month")[-1L], frequency = 12)

factor_glob_fcst_wide <- cbind(factor_glob_fcst_mat %>% ss(month(index(.)) %% 3L == 1L) %>% add_stub("m1_"),
                               factor_glob_fcst_mat %>% ss(month(index(.)) %% 3L == 2L) %>% add_stub("m2_") %>% unclass(),
                               factor_glob_fcst_mat %>% ss(month(index(.)) %% 3L == 0L) %>% add_stub("m3_") %>% unclass())

fcst_lasso_glob_wide_min <- cbind(icpt = 1, factor_glob_fcst_wide) %*% as.matrix(coef(cv_lasso_glob_wide, s = "lambda.min")) %>% 
  setColnames("fcst_lasso_wide_min")

fcst_data_glob <- rgdp_factors_glob_agg[, "rgdp_growth"] %>% 
  cbind(lasso_wide_min = unattrib(fit_lasso_glob_wide_min),
        lm = unattrib(fitted(gdp_lm_glob))) %>% 
  merge(fcst_lasso_glob_wide_min %>% copyMostAttrib(factor_glob_fcst_wide)) %>%
  merge(cbind(icpt = 1, apply.quarterly(factor_glob_fcst_mat, mean)) %*% coef(gdp_lm_glob) %>% 
          copyMostAttrib(factor_glob_fcst_wide) %>% setColnames("fcst_lm")) %>%
  ss(j = .c(rgdp_growth, lasso_wide_min, fcst_lasso_wide_min, lm, fcst_lm))

last_nmiss <- whichNA(fcst_data_glob[, "fcst_lasso_wide_min"]) %>% last()
fcst_data_glob[last_nmiss, "fcst_lasso_wide_min"] <- fcst_data_glob[last_nmiss, "lasso_wide_min"]
fcst_data_glob[last_nmiss, "fcst_lm"] <- fcst_data_glob[last_nmiss, "lm"]

fcst_data_glob %>% plot(legend.loc = "topleft", lwd = 1, main = "US GDP Nowcast from Global DFM")

