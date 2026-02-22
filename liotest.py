import liopandas as lp

df = lp.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Alice"],
    "age": [35, 30, 35, 35],
    "salary": [70000, 60000, 70000, 70000],
})

df = df[df["age"] > 25]

print(df.groupby("age").mean().drop(columns=["name"]))

df = df.drop(columns=["age"])

print(df)

df = df.drop_duplicates(subset=["name"])

print(df)






