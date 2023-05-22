library(fastverse)

#################
# Combining data
#################

if(FALSE) {

files <- list.files("models/DFM/results", full.names = TRUE)

blocked_res <- files[files %ilike% "blocked_dfm_long_"] %>% lapply(fread) %>% rbindlist() %>% roworder(vintage)
blocked_res %>% gv(1:2) %>% any_duplicated()
blocked_res %>% fwrite("models/DFM/results/blocked_dfm_long.csv")

global_res <- files[files %ilike% "global_dfm_long_"] %>% lapply(fread) %>% rbindlist() %>% roworder(vintage)
global_res %>% gv(1:2) %>% any_duplicated()
global_res %>% fwrite("models/DFM/results/global_dfm_long.csv")

}

blocked_res <- fread("models/DFM/results/blocked_dfm_long.csv") %>% 
               fsubset(as.integer(substr(year_month, 5, 7)) %% 3L == 0L)
global_res <- fread("models/DFM/results/global_dfm_long.csv") %>% 
              fsubset(as.integer(substr(year_month, 5, 7)) %% 3L == 0L)

qytodate <- function(x) as.Date(paste(substr(x, 1, 4), substr(as.integer(substr(x, 6, 6))*0.03, 3, 4), "01", sep = "-"))

res <- list(global = global_res, blocked = blocked_res) %>% 
       rbindlist(idcol = "model") %>% 
       ftransform(year_quarter = data.table::yearqtr(as.Date(paste0(year_month, "-01"))),
                  latest_gdp = data.table::yearqtr(qytodate(latest_gdp)), 
                  vintage_quarter = data.table::yearqtr(as.Date(paste0(sub("_", "-", vintage), "-01"))))

# Reduce and add other models
res %<>% fselect(model, year_quarter, vintage, vintage_quarter, latest_gdp, value = mean)

bridge_models_results <- readRDS("models/DFM/results/bridge_models_results.rds")

bridge_res <- bridge_models_results[-1] %>% 
  rapply2d(as.data.table) %>% 
  unlist2d(c("spec", "vintage"), DT = TRUE) %>% 
  fmutate(year_quarter = data.table::yearqtr(index),
          vintage_quarter = data.table::yearqtr(as.Date(paste0(sub("_", "-", vintage), "-01"))),
          index = replace(data.table::yearqtr(index), is.na(GDPC1), NA_real_),
          latest_gdp = fmax(index, list(spec, vintage), 1),
          index = NULL, GDPC1 = NULL) %>%
  melt(.c(spec, vintage, vintage_quarter, year_quarter, latest_gdp), 
       variable.name = "model") %>%
  ftransform(model = paste(model, spec, sep = "_"), 
             spec = NULL) %>% 
  as_character_factor()

res %<>% rbind(bridge_res, fill = TRUE)

# GDP ground truth Estimates
gdp <- fread("data/FRED/QD/vintage_2023_02.csv", select = 1:2) %>% 
       ftransform(year_quarter = data.table::yearqtr(qytodate(V1)), V1 = NULL)

res %<>% merge(gdp, by = "year_quarter") %>% fsubset(year_quarter >= 2000)

# res[, trel := as.integer((year_quarter - latest_gdp) * 4)]
res[, trel := as.integer((year_quarter - vintage_quarter) * 4)] # This is better (as discussed)
descr(res$trel)
qtab(res$model)

fwrite(res, "models/DFM/results/All_DFM_results_long.csv")


########################
# Performance Evaluation
########################

metrics <- function(x, y) list(r_squared = cor(x, y)^2, MAE_pgr = mean(abs(x - y)*100))

res[trel == -2L, metrics(value, GDPC1), by = model]
res[trel == -1L, metrics(value, GDPC1), by = model]
res[trel == 0L, metrics(value, GDPC1), by = model]
res[trel == 1L, metrics(value, GDPC1), by = model]
res[trel == 2L, metrics(value, GDPC1), by = model]
res[trel == 3L, metrics(value, GDPC1), by = model]

res[between(trel, -2L, 3L), c(list(N = .N), metrics(value, GDPC1)), by = .(model, trel)] %>% 
  dcast(trel + N ~ model, value.var = .c(r_squared, MAE_pgr))
