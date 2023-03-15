library(fastverse)

files <- list.files("data/FRED/blocked", full.names = TRUE) # %>% grep("vintage", ., value = TRUE)
blocked_datasets <- lapply(files, fread)

get_leading_series <- function(vintage) {
    d <- vintage %>% gvr("_m1$|_m2$|_m3$", invert = TRUE) %>% roworder(year_quarter) 
    gdp_end_nas <- intersect(whichNA(d$GDPC1), tail(seq_row(d)))
    names(which(fnobs(fsubset(d, gdp_end_nas, -year_quarter, -GDPC1)) > 0L))
}

leading_series <- lapply(blocked_datasets, get_leading_series)

sapply(leading_series, identical, character(0))
# -> Essentially no quarterly series is published before GDP
# -> This calls into question the usefulness of quarterly data for forecasting GDP