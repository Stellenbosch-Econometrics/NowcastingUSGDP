# %% 
# Import libaries
import numpy as np
import pandas as pd

# %% See Appendix at: https://s3.amazonaws.com/real.stlouisfed.org/wp/2015/2015-012.pdf
# Function to transform data
def transform(column, transforms):
    transformation = transforms[column.name]
    # 1 => No transformation
    if transformation == 1:
        pass
    # 2 => First difference
    elif transformation == 2:
        column = column.diff()
    # 3 => Second difference
    elif transformation == 3:
        column = column.diff().diff()
    # 4 => Log
    elif transformation == 4:
        column = np.log(column)
    # 5 => Log first difference
    elif transformation == 5:
        column = np.log(column).diff()
    # 6 => Log second difference
    elif transformation == 6:
        column = np.log(column).diff().diff()
    # 7 => Difference of the exact percentage change
    elif transformation == 7:
        column = ((column / column.shift(1)) - 1.0).diff()
    return column

# %% 
# Main function to download and processs data
def create_fredmd_blocked(vintage):
    base_url = 'https://files.stlouisfed.org/files/htdocs/fred-md'
    # 1. Download FRED-MD
    orig_m = pd.read_csv(f'{base_url}/monthly/{vintage}.csv').dropna(how='all')
    # 2. Extract transformation information
    transform_m = orig_m.iloc[0, 1:].astype(int)
    orig_m = orig_m.iloc[1:]
    # 3. Extract the date as an index
    orig_m.index = pd.PeriodIndex(orig_m.sasdate.tolist(), freq='M')
    orig_m.drop('sasdate', axis=1, inplace=True)
    # 4. Generate month and year_quarter variables
    orig_m["month"] = "m" + pd.Series(orig_m.index.month % 3).replace(0, 3).astype(str).values
    orig_m["year_quarter"] = orig_m.index.year.astype(str) + "Q" + orig_m.index.quarter.astype(str)
    # 5. Reshape
    orig_m_wide =  orig_m.pivot(index="year_quarter", columns="month")
    orig_m_wide.columns = ['_'.join(col) for col in orig_m_wide.columns.values]
    orig_m_wide.index = pd.PeriodIndex(orig_m_wide.index, freq="Q")
    # 6. Download FRED-QD
    orig_q = pd.read_csv(f'{base_url}/quarterly/{vintage}.csv').dropna(how='all')
    # 7. Extract factors and transformation information
    factors_q = orig_q.iloc[0, 1:].astype(int)
    transform_q = orig_q.iloc[1, 1:].astype(int)
    orig_q = orig_q.iloc[2:]
    # 8. Extract the date as an index
    orig_q.index = pd.PeriodIndex(orig_q.sasdate.tolist(), freq='Q')
    orig_q.drop('sasdate', axis=1, inplace=True)
    # 9. Merging
    final_mq = orig_m_wide.merge(orig_q, how='outer', left_index=True, right_index=True)
    # 10. Transforming
    transform_mq = transform_m.repeat(3) 
    transform_mq.index = orig_m_wide.columns
    transform_mq = pd.concat([transform_mq, transform_q])
    if not all(transform_mq.index == final_mq.columns):
        raise Exception("column transformation mismatch") 
    final_mq_transformed = final_mq.apply(transform, axis=0, transforms=transform_mq).iloc[1:]
    # 11. Returning results
    return dict(orig_m = orig_m, orig_q = orig_q, orig_m_wide = orig_m_wide, 
                final_mq = final_mq, final_mq_transformed = final_mq_transformed, transform_mq = transform_mq)


# %%
# Load Current Vintage
current_vintage = create_fredmd_blocked("current")
# %%
# Saving Current Vintage
current_vintage["final_mq_transformed"].to_csv("data/FRED/blocked/vintage_current.csv", index_label="year_quarter")
# %%
# Load Other Vintages
vintage_names = pd.date_range(start = "2019-01-01", end="2023-01-01", freq="M") \
                         .to_period().astype(str).to_list()

older_vintages = {v: create_fredmd_blocked(v) for v in vintage_names}
# %%
# Saving older vintages
for k, v in older_vintages.items():
    v["final_mq_transformed"].to_csv("data/FRED/blocked/vintage_{}.csv".format(k.replace("-", "_")), index_label="year_quarter")


# %%
