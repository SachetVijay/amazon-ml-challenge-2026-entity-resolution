import collections, re
import pyarrow.parquet as pq
pf = pq.ParquetFile("artifacts/train/records.parquet")
cnt = collections.Counter()
for rg in range(0, pf.num_row_groups, 3):                       # every 3rd row group is plenty
    t = pf.read_row_group(rg, columns=["business_address", "country"]).to_pandas()
    for addr in t.business_address[t.country == "India"]:
        for comp in str(addr).split(","):
            comp = comp.strip()
            if comp and re.search(r"[^\x00-\x7F]", comp):
                cnt[comp] += 1
for comp, n in cnt.most_common(45):
    print(n, comp)
