# Script to add a group_name variable (for DFM which uses blocks)
library(data.table)

fred_md = fread("data/FRED/FRED-MD Appendix/FRED-MD_updated_appendix.csv")

groups = c("Output and Income", "Labor Market", "Housing", 
 "Consumption, Orders, and Inventories", "Money and Credit",
 "Interest and Exchange Rates", "Prices", "Stock Market")

fred_md$group_name = groups[fred_md$group]

fwrite(fred_md, "data/FRED/FRED-MD Appendix/FRED-MD_updated_appendix.csv")