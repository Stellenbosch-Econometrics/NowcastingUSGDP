library(fastverse)
library(zoo)
source("results/forecast_evaluation.R")

DFM = fread("results/DFM/ALL_DFM_results_long.csv")
DL = fread("results/deep_learning/tidy_results.csv") %>% 
     fmutate(vintage = sub("_", "-", substr(vintage_file, 9, 15)))
ML = fread("results/machine_learning/tidy_svr.csv") %>% 
     fmutate(vintage = sub("_", "-", substr(vintage_file, 9, 15)))
GDP = fread("data/FRED/QD/vintage_2023_02.csv") %>% 
      fselect(year_quarter = V1, GDPC1) %>% 
      fmutate(year_quarter = as.integer(substr(year_quarter, 1, 4)) + (as.integer(substr(year_quarter, 6, 6))-1) / 4)
  
nowcasts = DL %>% fselect(year_quarter, vintage_quarter, vintage, model = Model, value = Estimate) %>% 
  rbind(ML %>% ftransform(year_quarter = unattrib(as.yearqtr(ds)),
                          vintage_quarter = unattrib(as.yearqtr(as.Date(paste0(vintage, "-01")))), 
                          Model = sub("svr", "SVR", tolower(Model))) %>% 
          fselect(year_quarter, vintage_quarter, vintage, model = Model, value = Estimate)) %>% 
  rbind(DFM %>% fselect(year_quarter, vintage_quarter, vintage, model, value) %>% tfm(model = paste0("DFM_", model))) %>% 
  fmutate(trel = as.integer((year_quarter - vintage_quarter)*4)) %T>%
  with(print(varying(trel, model, any_group = FALSE))) %>% 
  fsubset(trel == 0)

nowcasts %<>% merge(GDP, by = "year_quarter")

# Selected stats, computing error for each nowcast 
nowcasts[!model %ilike% "1se|o_min_"] %>% 
  fmutate(model = gsub("_wide_min", "", model),
          bias = (value - GDPC1) * 100, 
          MAE = abs(bias), 
          RMSE = bias^2) %>% 
  fgroup_by(model) %>% 
  fselect(bias, MAE, RMSE) %>% fmean() %>% 
  fmutate(RMSE = sqrt(RMSE)) %>% 
  roworder(RMSE) %>% 
  tfmv(is.numeric, round, 4)

# More thorough: averaging nowcatss 
nowcasts_wide  = nowcasts[!model %ilike% "1se|o_min_"] %>% 
  fmutate(model = gsub("_wide_min", "", model)) %>% 
  roworder(model, vintage) %>% 
  fgroup_by(model, year_quarter) %>% 
  fselect(value, GDPC1) %>% {
    list(average = fmean(.), last = flast(.))
  } %>% lapply(dcast, year_quarter ~ model, value.var = "value")
  

outcome = GDP[ckmatch(nowcasts_wide$average$year_quarter, year_quarter)]

# Average nowcast (3 vintages per quarter)
eval_forecasts(outcome$GDPC1 * 100, fselect(nowcasts_wide$average, -year_quarter) %c*% 100) %>% t() %>% qDF() %>% 
  fselect(-`R-Squared`, -MSE, -SE, -MPE, -U1) %>% roworder(RMSE) %>% round(2) %>% 
  xtable::xtable() %>% # print(type = "html") # Or
  print(type = "latex", booktabs = TRUE)

  

# Latest nowcast (final vintage)
eval_forecasts(outcome$GDPC1 * 100, fselect(nowcasts_wide$last, -year_quarter) %c*% 100) %>% t() %>% qDF() %>% 
  fselect(-`R-Squared`, -MSE, -SE, -MPE, -U1) %>% roworder(RMSE) %>% round(2) %>% 
  xtable::xtable() %>% # print(type = "html") # Or
  print(type = "latex", booktabs = TRUE)
