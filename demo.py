import liopandas as lp
import pandas as pd
import geopandas as gpd
import shapely.geometry as sg
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("LioPandas v{}".format(lp.__version__))
print("=" * 60)

# 1.  Series
print("\n▸ 1. Series basics")
s = lp.Series([10, 20, 30, 40, 50], index=["a", "b", "c", "d", "e"], name="scores")
print(s)
print(f"  sum={s.sum()}, mean={s.mean()}, std={s.std():.2f}")

# 2.  DataFrame
print("\n▸ 2. DataFrame creation")
df = lp.DataFrame({
    "name":   ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "age":    [28, 34, 22, 45, 31],
    "salary": [72000, 93000, 55000, 110000, 85000],
    "dept":   ["Eng", "Eng", "Sales", "Sales", "Eng"],
})
print(df)

# 3.  Filtering 
print("\n▸ 3. Boolean filtering  (salary > 70 000)")
rich = df[df["salary"] > 70000] 
print(rich)

# 4.  Column arithmetic
print("\n▸ 4. Add computed column  (bonus = salary * 0.1)")
df["bonus"] = df["salary"] * 0.1 
print(df)

# 5.  GroupBy 
print("\n▸ 5. GroupBy dept → mean")
grouped = df.groupby("dept").mean()
print(grouped)

# 6.  Sort 
print("\n▸ 6. Sort by salary descending")
print(df.sort_values("salary", ascending=False))

# 7.  Describe
print("\n▸ 7. describe()")
print(df[["age", "salary"]].describe())

# 8.  Merge 
print("\n▸ 8. Merge / Join")
dept_info = lp.DataFrame({
    "dept": ["Eng", "Sales", "HR"],
    "location": ["NYC", "LA", "Chicago"],
})
merged = lp.merge(df, dept_info, on="dept", how="left")
print(merged)

# 9.  CSV round-trip 
print("\n▸ 9. CSV I/O round-trip")
tmp = os.path.join(tempfile.gettempdir(), "liopandas_demo.csv")
df.to_csv(tmp, index=False)
df_loaded = lp.read_csv(tmp)
print(f"  Wrote {tmp}")
print(f"  Re-loaded shape: {df_loaded.shape}")
print(df_loaded)

# 10. Compare df
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


# 13. Offline Map Generation (No Internet Required)

data = {
    'city': [
        'Gaza City - Center',
        'Gaza - Al Shati Camp',
        'Gaza - Al Rimal',
        'Gaza - Sheikh Radwan',
        'Gaza - Tel al-Hawa'
    ],
    'population': [590481, 590481, 590481, 590481, 590481],
    'geometry': [
        sg.Point(34.4668, 31.5017),  # Gaza City Center
        sg.Point(34.4505, 31.5225),  # Al Shati (Beach Camp)
        sg.Point(34.4750, 31.5150),  # Al Rimal
        sg.Point(34.4620, 31.5300),  # Sheikh Radwan
        sg.Point(34.4500, 31.4900)   # Tel al-Hawa
    ]
}
gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

print("\n1. Created GeoPandas GeoDataFrame:")
print(gdf[['city', 'population']])

# Convert to LioPandas DataFrame
# LioPandas has a built-in from_geopandas method
ldf = lp.DataFrame.from_geopandas(gdf)

print("\n Converted to LioPandas DataFrame:")
print(ldf)

# 3. Generate static maps using LioPandas

print("\n Generating detailed map with SIDE-BY-SIDE context view...")
ldf.plot_static(output_path='With_side_map.png', show_city_labels=True, show_labels=False, show_context=True)

print(" Generating map WITHOUT city labels")
ldf.plot_static(output_path='Without_city_labels.png', show_city_labels=False, show_labels=True)

print("\n Maps generated successfully!")

print("\n All tests passed!\n")
