###########################
# Nowcast Evaluation in R
###########################

library(fastverse)
fastverse_extend(xts, africamonitor, dfms, vars, glmnet, install = TRUE)

prepcode <- function(x) toupper(sub(" ", "", x, fixed = TRUE))

vintages_m <- list.files("data/FRED/MD")
if(!identical(vintages_m, list.files("data/FRED/QD"))) stop("mismatch of monthly and quarterly vintages")

series <- fread("data/FRED/FRED-MD Appendix/FRED-MD_updated_appendix.csv") %>% 
          fmutate(fred = prepcode(fred))

# Helper function to ensure date is first day of quarter
index_month_to_quarter <- function(x) {
  ix <- index(x)
  lubridate::month(ix) <- as.integer(ceiling(month(ix) / 3L)) * 3L - 2L
  index(x) <- ix
  return(x)
}

# Results
nc_data_diff <- list(gdp = list(), blocked = list(), global = list())

# Outcome variable
ynam <- "GDPC1"

# Nowcasting Loop
for (i in vintages_m[1:2]) {
  
  it <- substr(i, 9, 15)
  print(it)
  
  data <- fread(paste0("data/FRED/MD/", i)) %>% 
    fmutate(V1 = am_as_date(V1)) %>% 
    frename(prepcode) %>% 
    fsubset(V1 >= "1990-01-01") %>% as.xts()

  gdp <- fread(paste0("data/FRED/QD/", i)) %>% 
    fmutate(V1 = am_as_date(V1)) %>% 
    frename(prepcode) %>% 
    fsubset(V1 >= "1990-01-01", V1, GDPC1) %>% as.xts()
  
  nc_data_diff$gdp[[it]] <- gdp
  
  # Results
  dfm <- new.env()
  dfm$data <- data
  
  # Global Model
  dfm_glob <- new.env()
  dfm_glob$data <- dfm$data
  
  dfm_glob$mod <- DFM(data, 9, 4)
  dfm_glob$factors <- dfm_glob$mod$F_qml %>% copy() %>% copyMostAttrib(data)
  
  # By Broad Sectors
  dfm$data_groups <- series[fred %in% colnames(data)] %>% rsplit(fred ~ group_name) %>% 
                     lapply(function(x) data[, x])
  
  # Global Factors
  dfm$mod_gl <- DFM(data, 2, 4, max.missing = 1)
  
  # Consumption, Orders, and Inventories Factors
  dfm$mod_coi <- DFM(dfm$data_groups$`Consumption, Orders, and Inventories`, 2, 4, max.missing = 1)
  
  # Housing Factors
  dfm$mod_h <- DFM(dfm$data_groups$Housing, 1, 2, max.missing = 1)
  
  # Interest and Exchange Rates Factors
  dfm$mod_ie <- DFM(dfm$data_groups$`Interest and Exchange Rates`, 2, 3, max.missing = 1)
  
  # Labor Market Factors
  dfm$mod_l <- DFM(dfm$data_groups$`Labor Market`, 1, 1, max.missing = 1)
  
  # Money and Credit Factors
  dfm$mod_mc <- DFM(dfm$data_groups$`Money and Credit`, 2, 4, max.missing = 1)
  
  # Output and Income Factors
  dfm$mod_oi <- DFM(dfm$data_groups$`Output and Income`, 1, 2, max.missing = 1)
  
  # Prices Factors
  dfm$mod_p <- DFM(dfm$data_groups$Prices, 1, 1, max.missing = 1)
  
  # Stock Market Factors
  dfm$mod_sm <- DFM(dfm$data_groups$`Stock Market`, 1, 1, max.missing = 1)

  # Combining factors
  dfm$factors <- cbind(dfm$mod$F_qml %>% add_stub("gl_"), 
                       dfm$mod_coi$F_qml %>% add_stub("coi_"),
                       dfm$mod_h$F_qml %>% add_stub("h_"),
                       dfm$mod_ie$F_qml %>% add_stub("ie_"),
                       dfm$mod_l$F_qml %>% add_stub("l_"),
                       dfm$mod_mc$F_qml %>% add_stub("mc_"),
                       dfm$mod_oi$F_qml %>% add_stub("oi_"),
                       dfm$mod_p$F_qml %>% add_stub("p_"),
                       dfm$mod_sm$F_qml %>% add_stub("sm_")) %>% copyMostAttrib(dfm$data)

  for (x in list(dfm, dfm_glob)) {
    # Aggregating to quarterly frequency
    x$factors_agg <- apply.quarterly(x$factors, mean) %>% index_month_to_quarter()
    x$rgdp_factors_agg <- x$factors_agg %>% merge(gdp[, ynam], all = FALSE) 
    # Expanding wide (blocking)
    x$factors_wide <- cbind(x$factors %>% ss(month(index(.)) %% 3L == 1L) %>% add_stub("m1_") %>% index_month_to_quarter(),
                            x$factors %>% ss(month(index(.)) %% 3L == 2L) %>% add_stub("m2_") %>% index_month_to_quarter(),
                            x$factors %>% ss(month(index(.)) %% 3L == 0L) %>% add_stub("m3_") %>% index_month_to_quarter()) %>% na.omit()
    x$rgdp_factors_wide <- x$factors_wide %>% merge(gdp[, ynam], all = FALSE)
  }

  #
  # Estimating Predictive models on historical data 
  #

  for (x in list(dfm, dfm_glob)) {
    tm <- qM(x$rgdp_factors_agg)
    tmw <- qM(x$rgdp_factors_wide)
    # Linear Model
    x$lm <- lm(as.formula(paste0(ynam, "~.")), qDF(tm))
    # Elastic Net
    x$cv_lasso <- cv.glmnet(x = tm[, -ncol(tm)], 
                            y = unattrib(tm[, ynam]), 
                            nfolds = nrow(tm), alpha = 0.5, grouped = FALSE)
    # Wide Elastic Net
    x$cv_lasso_wide <- cv.glmnet(x = tmw[, -ncol(tmw)], 
                                 y = unattrib(tmw[, ynam]), 
                                 nfolds = nrow(tmw), alpha = 0.5, grouped = FALSE)
    # Computing fit of all models
    x$fit_all <- x$rgdp_factors_agg[, ynam] %>% 
      cbind(lm = unattrib(fitted(x$lm)), 
            lasso_min = unattrib(predict(x$cv_lasso, tm[, -ncol(tm)], s = "lambda.min")), 
            lasso_1se = unattrib(predict(x$cv_lasso, tm[, -ncol(tm)], s = "lambda.1se"))) %>% 
      merge(cbind(lasso_wide_min = unattrib(predict(x$cv_lasso_wide, tmw[, -ncol(tmw)], s = "lambda.min")), 
                  lasso_wide_1se = unattrib(predict(x$cv_lasso_wide, tmw[, -ncol(tmw)], s = "lambda.1se")) %>% 
                    copyMostAttrib(x$rgdp_factors_wide)))
    rm(tm, tmw)
  }

  #
  # Now the Nowcast / Forecast 
  #

  dfm$factor_VAR <- VAR(dfm$factors, 2)
  dfm$factor_fcst <- predict(dfm$factor_VAR, n.ahead = 15) # 1 year and 1 quarter ahead
  dfm$factor_fcst_mat <- dfm$factor_fcst$fcst %>% lapply(function(x) x[, "fcst"]) %>% do.call(what = cbind) 
  dfm_glob$factor_fcst_mat <- predict(dfm_glob$mod, h = 15)$F_fcst

  for (x in list(dfm, dfm_glob)) {
    # For blocking it is important to have the full history...
    x$factor_fcst_comb <- x$factors %>% rbind(x$factor_fcst_mat %>% 
      xts(order.by = seq(last(index(x$factors)), length.out = 16, by = "month")[-1L], frequency = 12))
    
    x$factor_fcst_wide <- x$factor_fcst_comb %>% {
      cbind(ss(., month(index(.)) %% 3L == 1L) %>% add_stub("m1_") %>% index_month_to_quarter(),
            ss(., month(index(.)) %% 3L == 2L) %>% add_stub("m2_") %>% index_month_to_quarter(),
            ss(., month(index(.)) %% 3L == 0L) %>% add_stub("m3_") %>% index_month_to_quarter()) %>% na.omit()
    }
    x$fcst_data <- with(x, rgdp_factors_agg[, ynam] %>% 
                          merge(cbind(lm = unattrib(predict(lm, qDF(factor_fcst_comb))), 
                                      lasso_min = unattrib(predict(cv_lasso, qM(factor_fcst_comb), s = "lambda.min")), 
                                      lasso_1se = unattrib(predict(cv_lasso, qM(factor_fcst_comb), s = "lambda.1se"))) %>% 
                                  copyMostAttrib(factor_fcst_comb) %>% apply.quarterly(mean) %>% index_month_to_quarter()) %>%
                          merge(cbind(lasso_wide_min = unattrib(predict(cv_lasso_wide, qM(factor_fcst_wide), s = "lambda.min")), 
                                      lasso_wide_1se = unattrib(predict(cv_lasso_wide, qM(factor_fcst_wide), s = "lambda.1se")) %>% 
                                        copyMostAttrib(factor_fcst_wide)))) %>% na_omit(cols = -1)
  }
  # Saving results
  nc_data_diff$blocked[[it]] <- dfm$fcst_data
  nc_data_diff$global[[it]] <- dfm_glob$fcst_data
}

saveRDS(nc_data_diff, "models/DFM/results/bridge_models_results.rds")

# Evaluation
metrics <- function(x, y) c(r_squared = cor(x, y, use = "complete.obs")^2, MAE = mean(abs(x - y), na.rm = TRUE))
# source("code/forecast_evaluation.R")
# load("results/QB_DFM_nc_data_diff_eval.RData")

nc_data_diff$blocked %>% plot(legend.loc = "topleft", main = "US GDP Nowcasts from Large Blocked DFM") 
nc_data_diff$blocked %>% qDF() %>% sapply(metrics, .[[1]])
# nc_data_diff$blocked %>% {small_eval_forecasts(.[, 1], .[, -1])}

nc_data_diff$global %>% plot(legend.loc = "topleft", main = "US GDP Nowcasts from Large Global DFM")
nc_data_diff$global %>% qDF() %>% sapply(metrics, .[[1]])
# nc_data_diff$global %>% {small_eval_forecasts(.[, 1], .[, -1])}

# save(nc_data_diff, file = "results/QB_DFM_nc_data_diff_eval.RData")
