import liopandas as lp
import pandas as pd
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("LioPandas v{}".format(lp.__version__))
print("=" * 60)

# ── 1.  Series ────────────────────────────────────────────────
print("\n▸ 1. Series basics")
s = lp.Series([10, 20, 30, 40, 50], index=["a", "b", "c", "d", "e"], name="scores")
print(s)
print(f"  sum={s.sum()}, mean={s.mean()}, std={s.std():.2f}")

# ── 2.  DataFrame ────────────────────────────────────────────
print("\n▸ 2. DataFrame creation")
df = lp.DataFrame({
    "name":   ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "age":    [28, 34, 22, 45, 31],
    "salary": [72000, 93000, 55000, 110000, 85000],
    "dept":   ["Eng", "Eng", "Sales", "Sales", "Eng"],
})
print(df)

# ── 3.  Filtering ─────────────────────────────────────────────
print("\n▸ 3. Boolean filtering  (salary > 70 000)")
rich = df[df["salary"] > 70000] 
print(rich)

# ── 4.  Column arithmetic ─────────────────────────────────────
print("\n▸ 4. Add computed column  (bonus = salary * 0.1)")
df["bonus"] = df["salary"] * 0.1 # type: ignore
print(df)

# ── 5.  GroupBy ────────────────────────────────────────────────
print("\n▸ 5. GroupBy dept → mean")
grouped = df.groupby("dept").mean()
print(grouped)

# ── 6.  Sort ───────────────────────────────────────────────────
print("\n▸ 6. Sort by salary descending")
print(df.sort_values("salary", ascending=False))

# ── 7.  Describe ───────────────────────────────────────────────
print("\n▸ 7. describe()")
print(df[["age", "salary"]].describe())

# ── 8.  Merge ──────────────────────────────────────────────────
print("\n▸ 8. Merge / Join")
dept_info = lp.DataFrame({
    "dept": ["Eng", "Sales", "HR"],
    "location": ["NYC", "LA", "Chicago"],
})
merged = lp.merge(df, dept_info, on="dept", how="left")
print(merged)

# ── 9.  CSV round-trip ────────────────────────────────────────
print("\n▸ 9. CSV I/O round-trip")
tmp = os.path.join(tempfile.gettempdir(), "liopandas_demo.csv")
df.to_csv(tmp, index=False)
df_loaded = lp.read_csv(tmp)
print(f"  Wrote {tmp}")
print(f"  Re-loaded shape: {df_loaded.shape}")
print(df_loaded)

# 10. compare df
comparedf = df.compare(rich) # type: ignore
print(comparedf)

# 11. to_numpy
df = lp.DataFrame({
    "name":   ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "age":    [28, 34, 22, 45, 31],
    "salary": [72000, 93000, 55000, 110000, 85000],
    "dept":   ["Eng", "Eng", "Sales", "Sales", "Eng"],
})
nuparr = df.to_numpy()

print(nuparr)

# 12. to_pandas

# Liopandas to Pandas
ldf = lp.DataFrame({"A": [1, 2], "B": [3, 4]})
pdf = ldf.to_pandas()
print(type(pdf))
# Pandas to Liopandas
ldf_new = lp.DataFrame.from_pandas(pdf)
print(type(ldf_new))

print("\n All tests passed!\n")

