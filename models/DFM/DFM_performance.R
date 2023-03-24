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

########################
# Performance Evaluation
########################

blocked_res <- fread("models/DFM/results/blocked_dfm_long.csv") %>% 
               fsubset(as.integer(substr(year_month, 5, 7)) %% 3L == 0L)
global_res <- fread("models/DFM/results/global_dfm_long.csv") %>% 
              fsubset(as.integer(substr(year_month, 5, 7)) %% 3L == 0L)

qytodate <- function(x) as.Date(paste(substr(x, 1, 4), substr(as.integer(substr(x, 6, 6))*0.03, 3, 4), "01", sep = "-"))

res <- list(global = global_res, blocked = blocked_res) %>% 
       rbindlist(idcol = "model") %>% 
       ftransform(year_quarter = yearqtr(as.Date(paste0(year_month, "-01"))),
                  latest_gdp = yearqtr(qytodate(latest_gdp)))



# GDP ground truth Estimates
gdp <- fread("data/FRED/QD/vintage_2023_02.csv", select = 1:2) %>% 
       ftransform(year_quarter = yearqtr(qytodate(V1)), V1 = NULL)

res %<>% merge(gdp, by = "year_quarter")

res[, trel := as.integer((year_quarter - latest_gdp) * 4)]
descr(res$trel)


# Now the evaluation
metrics <- function(x, y) list(r_squared = cor(x, y)^2, MAE_pgr = mean(abs(x - y)*100))

res[trel == -2L, metrics(mean, GDPC1), by = model]
res[trel == -1L, metrics(mean, GDPC1), by = model]
res[trel == 0L, metrics(mean, GDPC1), by = model]
res[trel == 1L, metrics(mean, GDPC1), by = model]
res[trel == 2L, metrics(mean, GDPC1), by = model]
res[trel == 3L, metrics(mean, GDPC1), by = model]

res[between(trel, -2L, 3L), c(list(N = .N), metrics(mean, GDPC1)), by = .(model, trel)] %>% 
  dcast(trel + N ~ model, value.var = .c(r_squared, MAE_pgr))
