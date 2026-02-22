import liopandas as lp

df = lp.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [35, 30, 35],
    "salary": [50000, 60000, 70000],
})

df = df[df["age"] > 25]

print(df.groupby("age").mean().drop(columns=["name"]))

df = df.drop(columns=["age"])

df.to_csv("test.csv", index=False)

print(df.loc[1])



